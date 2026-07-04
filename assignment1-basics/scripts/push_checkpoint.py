"""Push chosen checkpoint files to a private Hugging Face Hub repo.

Run on the GPU instance after training:
    uv run python scripts/push_checkpoint.py checkpoints/my_ckpt.pt

Auth comes from the HF_TOKEN env var (write token from
https://huggingface.co/settings/tokens) or a cached `hf auth login`.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path, help="checkpoint file(s) to upload")
    parser.add_argument(
        "--repo",
        default=None,
        help="target repo id (default: <your-hf-username>/lmfromscratch-checkpoints)",
    )
    args = parser.parse_args()

    api = HfApi()
    # Default the repo to the token owner's namespace so no config is needed.
    repo_id = args.repo or f"{api.whoami()['name']}/lmfromscratch-checkpoints"

    # Idempotent: creates the private repo on first use, no-op afterwards.
    api.create_repo(repo_id, private=True, exist_ok=True)

    for ckpt in args.checkpoints:
        if not ckpt.is_file():
            raise SystemExit(f"not a file: {ckpt}")
        print(f"Uploading {ckpt} -> {repo_id}/{ckpt.name}")
        api.upload_file(path_or_fileobj=ckpt, path_in_repo=ckpt.name, repo_id=repo_id)

    print(f"\nDone: https://huggingface.co/{repo_id}")
    print("Download on your local machine with:")
    print(f"  uv run hf download {repo_id} <file> --local-dir checkpoints/")


if __name__ == "__main__":
    main()
