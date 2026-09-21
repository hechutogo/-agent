"""Project-local macOS Vulkan runtime setup, before importing SAPIEN."""
import os
from pathlib import Path
import sys


def configure():
    if sys.platform == "darwin":
        root = Path(__file__).resolve().parents[2]
        icd = root / ".runtime/share/vulkan/icd.d/MoltenVK_icd.json"
        if icd.exists():
            os.environ.setdefault("VK_ICD_FILENAMES", str(icd))
        os.environ.setdefault("MVK_CONFIG_LOG_LEVEL", "1")
