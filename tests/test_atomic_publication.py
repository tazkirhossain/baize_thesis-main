"""Bounded Windows sharing-lock retries preserve atomic no-overwrite publication."""
from pathlib import Path

import pytest

from src.data.preprocessing import _publish_directory


def test_transient_windows_sharing_lock_is_retried(monkeypatch, tmp_path):
    ready, output = tmp_path / "ready", tmp_path / "output"
    ready.mkdir()
    (ready / "data.txt").write_text("unchanged")
    original = Path.rename
    attempts = []
    def locked_once(path, destination):
        attempts.append(1)
        if len(attempts) == 1:
            error = PermissionError("transient sharing violation")
            error.winerror = 32
            raise error
        return original(path, destination)
    monkeypatch.setattr(Path, "rename", locked_once)
    monkeypatch.setattr("src.data.preprocessing.time.sleep", lambda _:None)
    _publish_directory(ready, output)
    assert len(attempts) == 2
    assert (output / "data.txt").read_text() == "unchanged"
    assert not ready.exists()


def test_existing_output_is_never_replaced(tmp_path):
    ready, output = tmp_path / "ready", tmp_path / "output"
    ready.mkdir()
    output.mkdir()
    with pytest.raises(FileExistsError):
        _publish_directory(ready, output)
    assert ready.exists()


@pytest.mark.parametrize("winerror,expected_attempts", [(32,9), (None,1)])
def test_permanent_lock_or_other_permission_failure_propagates(monkeypatch, tmp_path, winerror, expected_attempts):
    ready, output = tmp_path / "ready", tmp_path / "output"
    ready.mkdir()
    attempts = []
    def fail(path, destination):
        attempts.append(1)
        error = PermissionError("blocked")
        if winerror is not None:
            error.winerror = winerror
        raise error
    monkeypatch.setattr(Path, "rename", fail)
    monkeypatch.setattr("src.data.preprocessing.time.sleep", lambda _:None)
    with pytest.raises(PermissionError):
        _publish_directory(ready, output)
    assert len(attempts) == expected_attempts
    assert not output.exists()
