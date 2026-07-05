"""Push chosen checkpoint files to a private Hugging Face Hub repo.

Run on the GPU instance after training:
    uv run python scripts/push_checkpoint.py checkpoints/my_ckpt.pt
    uv run python scripts/push_checkpoint.py --datasets-only

Auth comes from the HF_TOKEN env var (write token from
https://huggingface.co/settings/tokens) or a cached `hf auth login`.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "cs336_basics" / "data"
DATASET_FILES = [DATA_DIR / f"ts-{split}-dataset.npy" for split in ("train", "valid")]


def upload_file(api: HfApi, repo_id: str, path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"not a file: {path}")
    print(f"Uploading {path} -> {repo_id}/{path.name}")
    api.upload_file(path_or_fileobj=path, path_in_repo=path.name, repo_id=repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="*", type=Path, help="checkpoint file(s) to upload")
    parser.add_argument(
        "--repo",
        default=None,
        help="target repo id (default: <your-hf-username>/lmfromscratch-checkpoints)",
    )
    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help="upload only ts-train-dataset.npy and ts-valid-dataset.npy",
    )
    args = parser.parse_args()

    if not args.datasets_only and not args.checkpoints:
        parser.error("provide at least one checkpoint, or use --datasets-only")

    api = HfApi()
    # Default the repo to the token owner's namespace so no config is needed.
    repo_id = args.repo or f"{api.whoami()['name']}/lmfromscratch-checkpoints"

    # Idempotent: creates the private repo on first use, no-op afterwards.
    api.create_repo(repo_id, private=True, exist_ok=True)

    paths = DATASET_FILES if args.datasets_only else [*args.checkpoints, *DATASET_FILES]
    for path in paths:
        upload_file(api, repo_id, path)

    print(f"\nDone: https://huggingface.co/{repo_id}")
    print("Download on your local machine with:")
    print(f"  uv run hf download {repo_id} <file> --local-dir checkpoints/")


if __name__ == "__main__":
    main()
