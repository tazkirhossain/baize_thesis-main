"""Bounded Flower RPC transfer of model parameters through private local files.

Large Fit/Evaluate parameter lists are staged in ordered, checksum-verified
chunks. Only serialized NumPy model tensors are allowed, never pickled objects.
This changes transport framing, not FedAvg semantics or model precision.
"""
import hashlib
import struct
import tempfile
import uuid
from pathlib import Path

import flwr as fl
from flwr.common import Parameters, FitIns, EvaluateIns, GetPropertiesIns, GetPropertiesRes, Status, Code
from flwr.common import serde
from flwr.server.client_proxy import ClientProxy

DEFAULT_LIMITS = {"chunk_bytes": 4*1024**2, "max_transfer_bytes": 16*1024**3}
MARKER = "chunked_numpy_v1"
OK = Status(Code.OK, "")
REF_KEYS = {"_chunk_id", "_chunk_size", "_chunk_sha256"}


def server_workspace(prefix):
    root = Path(__file__).resolve().parents[1]/"data/transport-runtime"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=root)


def parameter_size(parameters):
    return sum(len(value) for value in parameters.tensors)


def framed_size(parameters):
    return parameter_size(parameters) + 8*(1+len(parameters.tensors))


def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def write_parameters(path, parameters):
    if parameters.tensor_type != "numpy.ndarray":
        raise ValueError("only NumPy model parameters can be staged")
    with path.open("xb") as stream:
        stream.write(struct.pack(">Q", len(parameters.tensors)))
        for tensor in parameters.tensors:
            stream.write(struct.pack(">Q", len(tensor)))
            stream.write(tensor)
    return path.stat().st_size, checksum(path)


def read_parameters(path, maximum):
    if path.stat().st_size > maximum:
        raise ValueError("parameter transfer exceeds the configured limit")
    with path.open("rb") as stream:
        def length():
            value = stream.read(8)
            if len(value) != 8:
                raise ValueError("truncated parameter framing")
            return struct.unpack(">Q", value)[0]
        count = length()
        if count > 1_000_000:
            raise ValueError("invalid parameter tensor count")
        tensors = []
        for _ in range(count):
            size = length()
            if size > maximum or size > path.stat().st_size-stream.tell():
                raise ValueError("invalid tensor framing")
            tensors.append(stream.read(size))
        if stream.read(1):
            raise ValueError("unexpected parameter framing suffix")
    return Parameters(tensors, "numpy.ndarray")


def reference(identifier, size, digest):
    return {"_chunk_id": identifier, "_chunk_size": size, "_chunk_sha256": digest}


class ChunkedClient(fl.client.Client):
    def __init__(self, endpoint, *, contract, directory, limits=None):
        self.endpoint, self.contract = endpoint, contract
        self.directory, self.limits = Path(directory), limits or DEFAULT_LIMITS
        self.workspace, self.incoming, self.outgoing = None, {}, {}

    def _path(self, identifier):
        if not isinstance(identifier, str) or len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("invalid model-transfer identifier")
        if self.workspace is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.workspace = tempfile.TemporaryDirectory(prefix="model-transfer-", dir=self.directory)
        return Path(self.workspace.name) / identifier

    def get_properties(self, ins):
        cfg = ins.config
        if "_chunk_op" not in cfg:
            result = self.endpoint.get_properties(ins)
            result.properties["parameter_transport"] = MARKER
            return result
        if cfg.get("contract") != self.contract:
            raise ValueError("model-transfer contract mismatch")
        op, identifier = cfg["_chunk_op"], cfg.get("id")
        path = self._path(identifier)
        if op == "begin":
            if set(cfg) != {"_chunk_op", "contract", "id", "size", "sha256"}:
                raise ValueError("invalid model-transfer header")
            if self.incoming or identifier in self.outgoing or path.exists():
                raise ValueError("duplicate or concurrent model upload")
            size, digest = cfg["size"], cfg["sha256"]
            if type(size) is not int or not 8 <= size <= self.limits["max_transfer_bytes"]:
                raise ValueError("invalid model-transfer size")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid model-transfer checksum")
            path.touch(exist_ok=False)
            self.incoming[identifier] = dict(size=size, sha256=digest, offset=0, ready=False)
            return GetPropertiesRes(OK, {})
        if op == "write":
            record = self.incoming.get(identifier)
            block, offset = cfg.get("data"), cfg.get("offset")
            if (set(cfg) != {"_chunk_op", "contract", "id", "data", "offset"} or record is None or record["ready"] or
                    type(offset) is not int or offset != record["offset"] or type(block) is not bytes or
                    not 0 < len(block) <= self.limits["chunk_bytes"] or offset+len(block) > record["size"]):
                raise ValueError("invalid or out-of-order model chunk")
            with path.open("ab") as stream:
                stream.write(block)
            record["offset"] += len(block)
            return GetPropertiesRes(OK, {})
        if op == "seal":
            record = self.incoming.get(identifier)
            if (set(cfg) != {"_chunk_op", "contract", "id"} or record is None or record["ready"] or
                    record["offset"] != record["size"] or checksum(path) != record["sha256"]):
                raise ValueError("incomplete or corrupted model upload")
            record["ready"] = True
            return GetPropertiesRes(OK, {})
        if op == "read":
            record = self.outgoing.get(identifier)
            offset = cfg.get("offset")
            if (set(cfg) != {"_chunk_op", "contract", "id", "offset"} or record is None or
                    type(offset) is not int or offset != record["offset"] or offset >= record["size"]):
                raise ValueError("invalid or out-of-order model download")
            with path.open("rb") as stream:
                stream.seek(offset)
                block = stream.read(self.limits["chunk_bytes"])
            record["offset"] += len(block)
            return GetPropertiesRes(OK, {"data": block})
        if op == "release":
            if set(cfg) != {"_chunk_op", "contract", "id"}:
                raise ValueError("invalid model release")
            self.incoming.pop(identifier, None)
            self.outgoing.pop(identifier, None)
            path.unlink(missing_ok=True)
            return GetPropertiesRes(OK, {})
        raise ValueError("unknown model-transfer operation")

    def _restore(self, parameters, config):
        config = dict(config)
        present = REF_KEYS & config.keys()
        if not present:
            if parameter_size(parameters) > self.limits["chunk_bytes"]:
                raise ValueError("large model parameters require chunked transfer")
            return parameters, config
        if present != REF_KEYS or parameters.tensor_type != MARKER or parameters.tensors:
            raise ValueError("invalid staged model reference")
        identifier = config.pop("_chunk_id")
        record = self.incoming.get(identifier)
        size, digest = config.pop("_chunk_size"), config.pop("_chunk_sha256")
        if record is None or not record["ready"] or record["size"] != size or record["sha256"] != digest:
            raise ValueError("missing or mismatched staged model")
        path = self._path(identifier)
        parameters = read_parameters(path, self.limits["max_transfer_bytes"])
        path.unlink()
        del self.incoming[identifier]
        return parameters, config

    def get_parameters(self, ins):
        # The strategy always supplies server initialization, so this is unused
        # for large states. Fail rather than send an oversized accidental RPC.
        result = self.endpoint.get_parameters(ins)
        if parameter_size(result.parameters) > self.limits["chunk_bytes"]:
            raise ValueError("large models must use the strategy's explicit server initialization")
        return result

    def fit(self, ins):
        parameters, config = self._restore(ins.parameters, ins.config)
        result = self.endpoint.fit(FitIns(parameters, config))
        if parameter_size(result.parameters) > self.limits["chunk_bytes"]:
            if self.outgoing:
                raise ValueError("previous model update was not released")
            identifier = uuid.uuid4().hex
            path = self._path(identifier)
            if framed_size(result.parameters) > self.limits["max_transfer_bytes"]:
                raise ValueError("model update exceeds configured transfer limit")
            size, digest = write_parameters(path, result.parameters)
            if size > self.limits["max_transfer_bytes"]:
                path.unlink()
                raise ValueError("model update exceeds configured transfer limit")
            self.outgoing[identifier] = dict(size=size, offset=0)
            result.parameters = Parameters([], MARKER)
            result.metrics.update(reference(identifier, size, digest))
        return result

    def evaluate(self, ins):
        parameters, config = self._restore(ins.parameters, ins.config)
        return self.endpoint.evaluate(EvaluateIns(parameters, config))

    def close(self):
        if self.workspace is not None:
            self.workspace.cleanup()
            self.workspace = None
        self.incoming.clear()
        self.outgoing.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Explicit close is preferred; interpreter shutdown is best effort.


class ChunkedProxy(ClientProxy):
    def __init__(self, endpoint, *, contract, limits=None):
        super().__init__(endpoint.cid)
        self.endpoint, self.contract = endpoint, contract
        self.limits = limits or DEFAULT_LIMITS
        self.traffic = {}

    def get_properties(self, ins, timeout, group_id):
        return self.endpoint.get_properties(ins, timeout, group_id)

    def get_parameters(self, ins, timeout, group_id):
        return self.endpoint.get_parameters(ins, timeout, group_id)

    def reconnect(self, ins, timeout, group_id):
        return self.endpoint.reconnect(ins, timeout, group_id)

    def _rpc(self, cfg, timeout, group_id, traffic):
        ins = GetPropertiesIns(dict(cfg, contract=self.contract))
        response = self.endpoint.get_properties(ins, timeout, group_id)
        if response.status.code != Code.OK:
            raise RuntimeError("model chunk transfer failed")
        traffic[0] += serde.get_properties_ins_to_proto(ins).ByteSize()
        traffic[1] += serde.get_properties_res_to_proto(response).ByteSize()
        return response.properties

    def _request(self, ins, timeout, group_id, traffic):
        if parameter_size(ins.parameters) <= self.limits["chunk_bytes"]:
            return ins
        if framed_size(ins.parameters) > self.limits["max_transfer_bytes"]:
            raise ValueError("model exceeds configured transfer limit")
        with server_workspace("flower-model-upload-") as temporary:
            path = Path(temporary)/"parameters"
            size, digest = write_parameters(path, ins.parameters)
            if size > self.limits["max_transfer_bytes"]:
                raise ValueError("model exceeds configured transfer limit")
            identifier = uuid.uuid4().hex
            self._rpc({"_chunk_op": "begin", "id": identifier, "size": size, "sha256": digest}, timeout, group_id, traffic)
            with path.open("rb") as stream:
                offset = 0
                for block in iter(lambda: stream.read(self.limits["chunk_bytes"]), b""):
                    self._rpc({"_chunk_op": "write", "id": identifier, "offset": offset, "data": block}, timeout, group_id, traffic)
                    offset += len(block)
            self._rpc({"_chunk_op": "seal", "id": identifier}, timeout, group_id, traffic)
        return type(ins)(Parameters([], MARKER), dict(ins.config, **reference(identifier, size, digest)))

    def fit(self, ins, timeout, group_id):
        traffic = [0, 0]
        request = self._request(ins, timeout, group_id, traffic)
        traffic[0] += serde.fit_ins_to_proto(request).ByteSize()
        result = self.endpoint.fit(request, timeout, group_id)
        traffic[1] += serde.fit_res_to_proto(result).ByteSize()
        if result.parameters.tensor_type == MARKER:
            if result.parameters.tensors or not REF_KEYS <= result.metrics.keys():
                raise ValueError("invalid chunked model response")
            identifier = result.metrics.pop("_chunk_id")
            size, digest = result.metrics.pop("_chunk_size"), result.metrics.pop("_chunk_sha256")
            if type(size) is not int or not 8 <= size <= self.limits["max_transfer_bytes"]:
                raise ValueError("invalid response transfer size")
            with server_workspace("flower-model-response-") as temporary:
                path = Path(temporary)/"parameters"
                with path.open("xb") as stream:
                    offset = 0
                    while offset < size:
                        response = self._rpc({"_chunk_op": "read", "id": identifier, "offset": offset}, timeout, group_id, traffic)
                        block = response.get("data")
                        if (set(response) != {"data"} or type(block) is not bytes or
                                not 0 < len(block) <= min(self.limits["chunk_bytes"], size-offset)):
                            raise ValueError("invalid model response chunk")
                        stream.write(block)
                        offset += len(block)
                if checksum(path) != digest:
                    raise ValueError("corrupted model response")
                result.parameters = read_parameters(path, self.limits["max_transfer_bytes"])
            self._rpc({"_chunk_op": "release", "id": identifier}, timeout, group_id, traffic)
        elif parameter_size(result.parameters) > self.limits["chunk_bytes"]:
            raise ValueError("large model response was not chunked")
        self.traffic["fit"] = tuple(traffic)
        return result

    def evaluate(self, ins, timeout, group_id):
        traffic = [0, 0]
        request = self._request(ins, timeout, group_id, traffic)
        traffic[0] += serde.evaluate_ins_to_proto(request).ByteSize()
        result = self.endpoint.evaluate(request, timeout, group_id)
        traffic[1] += serde.evaluate_res_to_proto(result).ByteSize()
        self.traffic["evaluate"] = tuple(traffic)
        return result
