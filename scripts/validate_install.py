"""Validate imports and report optional CUDA acceleration."""

import importlib
import sys


REQUIRED = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "accelerate": "accelerate",
    "trl": "trl",
    "deepspeed": "deepspeed",
    "qwen_vl_utils": "qwen-vl-utils",
    "cv2": "opencv-python-headless",
    "rdkit": "rdkit",
    "molnextr": "MolNexTR vendored module",
    "sam3": "sam3",
}


def main():
    failures = []
    for module_name, package_name in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"OK  {module_name:<18} {version}")
        except Exception as error:
            failures.append((module_name, package_name, error))
            print(f"ERR {module_name:<18} {type(error).__name__}: {error}")

    if not failures:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
        return 0

    print("\nMissing or broken packages:")
    for module_name, package_name, _ in failures:
        print(f"  {module_name}: install {package_name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())