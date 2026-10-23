"""Real serialization, bounded RPC frames, corruption and ordering rejection."""
from pathlib import Path
import hashlib
import uuid
import numpy as np
import pytest
fl = pytest.importorskip("flwr")
from flwr.common import FitIns, EvaluateIns, GetPropertiesIns, ndarrays_to_parameters, parameters_to_ndarrays, serde
from src.transport import ChunkedClient, ChunkedProxy


class TensorClient(fl.client.NumPyClient):
    def get_properties(self, config):
        return {"client_id": 0}

    def fit(self, parameters, config):
        assert config == {"round": 1}
        return [value+1 for value in parameters], 7, {"training_count": 7}

    def evaluate(self, parameters, config):
        return 0.5, 4, {"tn": 2, "tp": 2}


class WireProxy:
    cid = "0"
    def __init__(self, endpoint):
        self.endpoint, self.sizes = endpoint, []

    def get_properties(self, ins, timeout, group_id):
        res = self.endpoint.get_properties(ins)
        self.sizes += [serde.get_properties_ins_to_proto(ins).ByteSize(), serde.get_properties_res_to_proto(res).ByteSize()]
        return res

    def fit(self, ins, timeout, group_id):
        self.sizes.append(serde.fit_ins_to_proto(ins).ByteSize())
        res = self.endpoint.fit(ins)
        self.sizes.append(serde.fit_res_to_proto(res).ByteSize())
        return res

    def evaluate(self, ins, timeout, group_id):
        self.sizes.append(serde.evaluate_ins_to_proto(ins).ByteSize())
        res = self.endpoint.evaluate(ins)
        self.sizes.append(serde.evaluate_res_to_proto(res).ByteSize())
        return res


def test_chunked_model_roundtrip_counts_actual_bounded_wire_bytes(tmp_path):
    limits = {"chunk_bytes": 1024, "max_transfer_bytes": 1024*1024}
    endpoint = ChunkedClient(TensorClient().to_client(), contract="contract", directory=tmp_path, limits=limits)
    wire = WireProxy(endpoint)
    proxy = ChunkedProxy(wire, contract="contract", limits=limits)
    arrays = [np.arange(12000, dtype=np.float32).reshape(100, 120), np.ones(2000, dtype=np.float32)]
    parameters = ndarrays_to_parameters(arrays)
    try:
        res = proxy.fit(FitIns(parameters, {"round": 1}), None, 1)
        for original, actual in zip(arrays, parameters_to_ndarrays(res.parameters)):
            np.testing.assert_array_equal(actual, original+1)
        assert res.metrics == {"training_count": 7}
        assert sum(proxy.traffic["fit"]) == sum(wire.sizes)
        assert max(wire.sizes) < 2048
        assert not endpoint.incoming and not endpoint.outgoing
        assert not list(Path(endpoint.workspace.name).iterdir())
        wire.sizes.clear()
        assert proxy.evaluate(EvaluateIns(parameters, {}), None, 1).num_examples == 4
        assert sum(proxy.traffic["evaluate"]) == sum(wire.sizes)
        assert max(wire.sizes) < 2048
    finally:
        endpoint.close()
    assert not list(tmp_path.iterdir())


def test_corrupt_out_of_order_replayed_and_oversized_chunks_fail(tmp_path):
    limits = {"chunk_bytes": 1024, "max_transfer_bytes": 4096}
    endpoint = ChunkedClient(TensorClient().to_client(), contract="contract", directory=tmp_path, limits=limits)
    identifier = uuid.uuid4().hex
    def request(op, **kwargs):
        return endpoint.get_properties(GetPropertiesIns(dict(_chunk_op=op, contract="contract", id=identifier, **kwargs)))
    try:
        request("begin", size=12, sha256=hashlib.sha256(b"abcdefghijkl").hexdigest())
        with pytest.raises(ValueError, match="duplicate"):
            request("begin", size=12, sha256="0"*64)
        with pytest.raises(ValueError, match="out-of-order"):
            request("write", offset=1, data=b"wrong")
        with pytest.raises(ValueError):
            request("write", offset=0, data=b"x"*1025)
        request("write", offset=0, data=b"xxxxxxxxxxxx")
        with pytest.raises(ValueError, match="corrupted"):
            request("seal")
        request("release")
        with pytest.raises(ValueError, match="size"):
            request("begin", size=4097, sha256="0"*64)
        with pytest.raises(ValueError, match="contract"):
            endpoint.get_properties(GetPropertiesIns({"_chunk_op": "release", "id": identifier, "contract": "wrong"}))
        with pytest.raises(ValueError, match="identifier"):
            endpoint.get_properties(GetPropertiesIns({"_chunk_op": "release", "id": "../unsafe", "contract": "contract"}))
    finally:
        endpoint.close()
