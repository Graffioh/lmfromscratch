# Remote GPU workflow

Works on any provider (RunPod, Prime Intellect, Lambda, ...): shell access via
the provider's SSH command or web terminal, code and the canonical BPE
tokenizer artifacts (committed in outputs/) go up via git, checkpoints go to a
private HF Hub repo. No file transfer runs over SSH.

## One-time setup

1. Add `~/.ssh/id_ed25519.pub` to the provider account (e.g. RunPod: Settings -> SSH Public Keys).
2. Create the instance, on container providers using the custom image from
   `docker/Dockerfile.gpu`.
3. Create a HF write token at https://huggingface.co/settings/tokens.

## Each session

```
# 1. connect with the SSH command from the provider dashboard (or web terminal)
#    e.g. RunPod: ssh POD-ID@ssh.runpod.io -i ~/.ssh/id_ed25519

# 2. get the code (first time: clone; later: pull)
cd /workspace                       # persistent volume on container providers
git clone https://github.com/Graffioh/lmfromscratch.git   # or: cd lmfromscratch && git pull
cd lmfromscratch/assignment1-basics

# 3. keys
export WANDB_API_KEY=...
export HF_TOKEN=...

# 4. setup + train (downloads the dataset if missing; BPE artifacts come
#    from git, retrained only as a fallback if they were deleted)
bash scripts/setup_gpu.sh --train

# long runs: survive disconnects with
nohup bash scripts/setup_gpu.sh --train > train.log 2>&1 &
tail -f train.log
```

> If you ever retrain the tokenizer (e.g. new vocab size), commit the new
> outputs/ pickles together with the config change: checkpoints only decode
> with the exact vocab/merges they were trained with.

## Save the checkpoints you want to keep (on the GPU instance)

Pushes to a private repo `<hf-username>/lmfromscratch-checkpoints` (created on
first use). Do this before stopping the pod: unpushed checkpoints die with it.

```
uv run python scripts/push_checkpoint.py checkpoints/my_ckpt.pt
```

## Download a checkpoint (on the local machine)

```
uv run hf download <hf-username>/lmfromscratch-checkpoints <file> --local-dir checkpoints/
```
