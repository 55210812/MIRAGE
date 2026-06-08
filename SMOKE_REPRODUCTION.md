# MIRAGE Smoke Reproduction

Target server path:

```bash
/mnt/data2/zyc/mirage
```

The current smoke path uses AgentDojo's local DeepSeek HF weights at `/home/intern/models/DeepSeek-R1-Distill-Qwen-14B`. MIRAGE attribution needs local model internals and gradients, so the smoke test loads the Hugging Face model directory directly rather than using the vLLM/OpenAI-compatible service.

## Setup

Copy runtime secrets locally on the server, without committing them:

```bash
cp /mnt/data2/zyc/agentdojo/.env /mnt/data2/zyc/mirage/.env
```

Create the project environment:

```bash
cd /mnt/data2/zyc/mirage
bash scripts/setup_zqllms_overlay.sh
```

The overlay setup script reuses AgentDojo's `/home/hqdeng7/.conda/envs/zqllms/bin/python` and installs only the packages missing from that environment into `.envs/zqllms-overlay`. It retries failed network commands through `127.0.0.1:7890`, then through `127.0.0.1:17890` for an SSH reverse proxy from the local machine.

The full official Conda environment remains available through `scripts/setup_env.sh` if an isolated Python 3.9 environment is required.

## Smoke Test

```bash
cd /mnt/data2/zyc/mirage
bash scripts/run_smoke_deepseek.sh
```

The smoke test runs:

1. dependency import and CUDA checks
2. bundled one-item ELI5 smoke fixture, or ALCE ELI5 data download if the fixture is removed
3. one-sample DeepSeek Qwen 14B generation
4. MIRAGE internal attribution
5. MIRAGE citation generation
6. lightweight citation evaluation

The older Llama 3.2 smoke script is retained in `scripts/run_smoke_llama32.sh`, but the default reproduction target is DeepSeek because it is already present on the AgentDojo server in Hugging Face format.
