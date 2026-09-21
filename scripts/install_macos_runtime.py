"""Fetch verified conda-forge runtime archives without a global conda install."""
import hashlib
import io
from pathlib import Path
import platform
import tarfile
import tempfile
import urllib.request
import zipfile

import zstandard

ROOT = Path(__file__).resolve().parents[1] / ".runtime"
PACKAGES = {
    "moltenvk-1.4.2-h407b865_0.conda":
        "8e489cff52952e7599b8bc20fbe4530a97780224aad882be7d16b1ade8511fb2",
    "libcxx-19.1.7-ha82da77_2.conda":
        "07e4efaea695d807135f4d2d217bfec586bf1be0c81b69be8088d5e2cf8a606f",
}


def main():
    if platform.system() != "Darwin":
        return
    if platform.machine() != "arm64":
        raise SystemExit("Bundled runtime supports Apple Silicon; configure Vulkan for Intel Mac separately.")
    ROOT.mkdir(exist_ok=True)
    for name, expected in PACKAGES.items():
        archive = ROOT / name
        if (not archive.exists()
                or hashlib.sha256(archive.read_bytes()).hexdigest() != expected):
            print(f"Downloading {name}", flush=True)
            with tempfile.NamedTemporaryFile(dir=ROOT, prefix=f".{name}.",
                                             suffix=".tmp", delete=False) as download:
                temporary = Path(download.name)
            try:
                urllib.request.urlretrieve(
                    f"https://conda.anaconda.org/conda-forge/osx-arm64/{name}", temporary)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Checksum mismatch: {archive}")
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
        with zipfile.ZipFile(archive) as package:
            for member in package.namelist():
                if member.startswith("pkg-") and member.endswith(".tar.zst"):
                    data = zstandard.ZstdDecompressor().decompress(package.read(member))
                    with tarfile.open(fileobj=io.BytesIO(data)) as content:
                        content.extractall(ROOT, filter="data")
    print("Project-local MoltenVK runtime installed.")


if __name__ == "__main__":
    main()
