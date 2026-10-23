"""Client-local split-reader protection and integrity checks."""
import csv

import pytest

from src.data.preprocessing import PreprocessingConfig, prepare_clients, split_client
from src.client.data import read_local_split
from src.data.partitioning import partition_prepared


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("balanced-client")
    source = root / "raw.csv"
    with source.open("w",newline="") as stream:
        writer = csv.writer(stream)
        for domain in range(1000):
            for page in range(2):
                label = "Adult" if domain%10 == 0 else "Arts"
                marker = "private-adult-path" if label == "Adult" else "public-info"
                writer.writerow([f"{domain:05}-{page}",f"https://site{domain}.com/{marker}/{page}",label])
    prepared = root / "prepared"
    prepare_clients(source,prepared,PreprocessingConfig(n_features=64))
    iid = root / "iid"
    partition_prepared(prepared,iid)
    directory = iid / "client_000"
    split_client(directory)
    return directory


def test_public_reader_protects_holdout_labels_and_checks_file_hashes(client,tmp_path):
    assert read_local_split(client,"val").labels.protected
    assert read_local_split(client,"test").labels.protected
    assert not read_local_split(client,"train").labels.flags.writeable
    import shutil
    copy = tmp_path / "client"
    shutil.copytree(client,copy)
    with (copy / "splits/test.jsonl").open("a") as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError,match="checksum"):
        read_local_split(copy,"test")

