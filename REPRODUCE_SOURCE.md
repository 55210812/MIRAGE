# MIRAGE Reproduction Source

This repository is based on the official MIRAGE reproduction code:

- Upstream: https://github.com/Betswish/MIRAGE-reproduce
- Upstream commit: `a9201b69d015ad0a5e830b36f7e855da74fe827f`
- Paper: https://aclanthology.org/2024.emnlp-main.347/

Local changes are limited to making DeepSeek and Llama smoke reproduction paths runnable on `hqdeng-server`, adding proxy-aware setup/test scripts, and excluding local secrets, downloaded datasets, model caches, and generated outputs from git.
