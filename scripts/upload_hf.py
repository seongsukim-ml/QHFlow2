"""Upload QHFlow2 pre-trained checkpoints to HuggingFace."""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", "ckpt"))

# Mapping: (local_dir, repo_id, path_in_repo)
UPLOADS = {
    "QH9": {
        "repo": "ksusu/QHFlow2-QH9",
        "base": CKPT_ROOT / "QH9",
    },
    "MD17": {
        "repo": "ksusu/QHFlow2-MD17",
        "base": CKPT_ROOT / "md17",
    },
    "rMD17": {
        "repo": "ksusu/QHFlow2-rMD17",
        "base": CKPT_ROOT / "rmd17",
    },
}


def upload_dataset(name: str, info: dict, api: HfApi):
    repo_id = info["repo"]
    base = info["base"]

    # Find all .ckpt files
    ckpts = sorted(base.rglob("*.ckpt"))
    print(f"\n{'='*60}")
    print(f"[{name}] {repo_id}: {len(ckpts)} checkpoints")
    print(f"{'='*60}")

    for ckpt in ckpts:
        # Build path: split/config/filename
        rel = ckpt.relative_to(base)
        size_gb = ckpt.stat().st_size / 1e9
        print(f"  Uploading {rel} ({size_gb:.1f} GB)...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(ckpt),
                path_in_repo=str(rel),
                repo_id=repo_id,
                repo_type="model",
            )
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")


def main():
    api = HfApi(token=TOKEN)

    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        for name, info in UPLOADS.items():
            upload_dataset(name, info, api)
    elif target in UPLOADS:
        upload_dataset(target, UPLOADS[target], api)
    else:
        print(f"Unknown target: {target}. Use: all, QH9, MD17, rMD17")
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
