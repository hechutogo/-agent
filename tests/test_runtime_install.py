"""Real archive verification/extraction with a local fake download boundary."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest
import zstandard


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/install_macos_runtime.py"
    spec = importlib.util.spec_from_file_location("install_macos_runtime", script)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    content = b"verified runtime payload"
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode="w") as archive:
        member = tarfile.TarInfo("lib/runtime.dylib")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("pkg-runtime.tar.zst",
                         zstandard.ZstdCompressor().compress(tar.getvalue()))
    payload = package.getvalue()
    monkeypatch.setattr(installer, "ROOT", tmp_path / "runtime")
    monkeypatch.setattr(installer, "PACKAGES", {
        "runtime.conda": hashlib.sha256(payload).hexdigest(),
    })
    monkeypatch.setattr(installer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(installer.platform, "machine", lambda: "arm64")
    installer.ROOT.mkdir()
    return installer, payload, content


@pytest.mark.parametrize("cached", [False, True])
def test_verified_download_is_atomically_promoted_and_reused(runtime, monkeypatch, cached):
    installer, payload, content = runtime
    archive = installer.ROOT / "runtime.conda"
    if cached:
        archive.write_bytes(b"old interrupted download")
    downloads, replacements = [], []
    replace = Path.replace

    def download(url, destination):
        destination = Path(destination)
        downloads.append(destination)
        assert destination != archive
        assert destination.parent == archive.parent
        assert archive.read_bytes() == b"old interrupted download" if cached else not archive.exists()
        destination.write_bytes(payload)

    def promote(source, destination):
        assert source.read_bytes() == payload
        assert Path(destination) == archive
        replacements.append(source)
        return replace(source, destination)

    monkeypatch.setattr(installer.urllib.request, "urlretrieve", download)
    monkeypatch.setattr(Path, "replace", promote)
    installer.main()
    assert len(downloads) == len(replacements) == 1
    assert archive.read_bytes() == payload
    assert (installer.ROOT / "lib/runtime.dylib").read_bytes() == content
    assert sorted(path.name for path in installer.ROOT.iterdir()) == ["lib", "runtime.conda"]
    installer.main()
    assert len(downloads) == 1


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("failure", ["interrupted", "checksum"])
def test_failed_download_is_cleaned_and_next_invocation_recovers(runtime, monkeypatch, cached, failure):
    installer, payload, content = runtime
    archive = installer.ROOT / "runtime.conda"
    if cached:
        archive.write_bytes(b"old partial archive")
    downloads = []

    def fail_download(url, destination):
        downloads.append(Path(destination))
        Path(destination).write_bytes(b"partial or corrupt download")
        if failure == "interrupted":
            raise OSError("download interrupted")

    monkeypatch.setattr(installer.urllib.request, "urlretrieve", fail_download)
    error_type = OSError if failure == "interrupted" else RuntimeError
    reason = "download interrupted" if failure == "interrupted" else "Checksum mismatch"
    with pytest.raises(error_type, match=reason):
        installer.main()
    assert len(downloads) == 1  # Bounded fresh attempt, even with a bad cache.
    assert not (installer.ROOT / "lib").exists()  # Never extract unverified data.
    assert sorted(path.name for path in installer.ROOT.iterdir()) == (
        ["runtime.conda"] if cached else [])
    if cached:
        assert archive.read_bytes() == b"old partial archive"

    def good_download(url, destination):
        downloads.append(Path(destination))
        Path(destination).write_bytes(payload)

    monkeypatch.setattr(installer.urllib.request, "urlretrieve", good_download)
    installer.main()
    assert len(downloads) == 2
    assert archive.read_bytes() == payload
    assert (installer.ROOT / "lib/runtime.dylib").read_bytes() == content
    assert sorted(path.name for path in installer.ROOT.iterdir()) == ["lib", "runtime.conda"]


def test_atomic_replace_failure_cleans_temporary_archive(runtime, monkeypatch):
    installer, payload, _ = runtime
    archive = installer.ROOT / "runtime.conda"

    def download(url, destination):
        Path(destination).write_bytes(payload)

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(installer.urllib.request, "urlretrieve", download)
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        installer.main()
    assert not archive.exists()
    assert list(installer.ROOT.iterdir()) == []
