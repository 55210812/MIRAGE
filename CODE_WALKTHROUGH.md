# MIRAGE 代码细粒度讲解

本文档用于汇报 `55210812/MIRAGE` 仓库的代码逻辑。重点覆盖本次复现真正跑通的 Section 5 LongQA / DeepSeek smoke 链路，而不是逐行复述所有论文实验脚本。

## 0. 一句话总览

MIRAGE 的核心思想是：先让 RAG 模型基于检索文档生成答案，再用模型内部归因判断答案每句话主要依赖哪些检索文档，最后把这些依赖关系转换成显式 citation，例如 `[1]`、`[2]`。

这和普通 RAG 的区别在于，引用不是完全依赖模型自己“说它引用了什么”，而是通过模型内部 token attribution 计算出来。

## 1. 仓库结构

关键目录如下：

```text
.
├── MIRAGE.yaml
├── REPRODUCE_SOURCE.md
├── SMOKE_REPRODUCTION.md
├── CODE_WALKTHROUGH.md
├── scripts/
│   ├── setup_env.sh
│   ├── setup_zqllms_overlay.sh
│   ├── run_smoke_deepseek.sh
│   └── run_smoke_llama32.sh
└── sec5_longQA/
    ├── configs/
    │   └── eli5_deepseek_qwen14b_shot0_ndoc2_bm25_selfcitation_smoke.yaml
    ├── smoke_data/
    │   └── eli5_eval_bm25_top100_smoke.json
    ├── run.py
    ├── mirage_attribute.py
    ├── mirage_cite.py
    ├── eval.py
    ├── utils.py
    └── searcher.py
```

各文件角色：

- `MIRAGE.yaml`：官方 Conda 环境描述，本次移除了上游个人机器路径里的 `prefix`。
- `REPRODUCE_SOURCE.md`：记录上游仓库、上游 commit 和论文链接。
- `SMOKE_REPRODUCTION.md`：记录如何在服务器上复现 smoke 测试。
- `scripts/setup_env.sh`：创建完整独立 Conda 环境。
- `scripts/setup_zqllms_overlay.sh`：复用 AgentDojo 的 `zqllms` 环境，只补装 MIRAGE 缺少的包。
- `scripts/run_smoke_deepseek.sh`：端到端 smoke 入口。
- `sec5_longQA/run.py`：生成 LongQA 答案。
- `sec5_longQA/mirage_attribute.py`：调用 `inseq` 做 MIRAGE 内部归因。
- `sec5_longQA/mirage_cite.py`：把归因结果转换成引用。
- `sec5_longQA/eval.py`：评估答案质量和引用格式。
- `sec5_longQA/utils.py`：prompt 构造、模型加载、文本规范化等公共函数。
- `sec5_longQA/searcher.py`：交互式检索模式下的文档内搜索器。

## 2. 端到端执行链路

本次跑通的命令是：

```bash
cd /mnt/data2/zyc/mirage
bash scripts/run_smoke_deepseek.sh
```

它对应的执行链路是：

```text
setup_zqllms_overlay.sh
        |
        v
run_smoke_deepseek.sh
        |
        +--> run.py
        |       读取 config + smoke data
        |       拼接 prompt
        |       加载 DeepSeek HF 模型
        |       生成答案 JSON
        |
        +--> mirage_attribute.py
        |       读取答案 JSON
        |       重新加载同一个 DeepSeek 模型
        |       用 inseq 计算 token attribution
        |       写 internal_selfcitation/*.json
        |
        +--> mirage_cite.py
        |       读取 attribution JSON
        |       按句子聚合 token attribution
        |       选择相关文档编号
        |       写 *.mirage_cite_CTI_1_CCI_-5
        |
        +--> eval.py
                计算长度、ROUGE、citation regex 指标
                写 *.score
```

输出文件在服务器上生成，但不提交到 Git：

```text
sec5_longQA/result/selfcitation/*.json
sec5_longQA/result/selfcitation/*.mirage_cite_CTI_1_CCI_-5
sec5_longQA/result/selfcitation/*.score
sec5_longQA/internal_selfcitation/*.json
```

## 3. 复现配置文件

文件：`sec5_longQA/configs/eli5_deepseek_qwen14b_shot0_ndoc2_bm25_selfcitation_smoke.yaml`

```yaml
prompt_file: prompts/eli5_default_llama2_selfcitation.json
eval_file: smoke_data/eli5_eval_bm25_top100_smoke.json
quick_test: 1
shot: 0
ndoc: 2
dataset_name: eli5
tag: bm25-smoke
model: /home/intern/models/DeepSeek-R1-Distill-Qwen-14B
temperature: 0.6
top_p: 0.95
do_sample: false
max_new_tokens: 64
```

字段解释：

- `prompt_file`：使用 self-citation 风格 prompt。虽然文件名里有 `llama2`，但这里主要复用其 prompt 模板，不表示模型是 Llama2。
- `eval_file`：使用仓库内置的一条 ELI5 smoke fixture，避免每次 smoke 都下载完整 ALCE 数据。
- `quick_test: 1`：只跑 1 条样本，目标是快速验证链路。
- `shot: 0`：不使用 in-context demonstration。
- `ndoc: 2`：每个问题只放前 2 篇检索文档。
- `dataset_name: eli5`：输出文件名的一部分。
- `tag: bm25-smoke`：输出文件名的一部分，用来标记这次是 smoke。
- `model`：本地 Hugging Face 模型目录。MIRAGE attribution 需要梯度和内部状态，所以这里不能只填 vLLM API 地址。
- `temperature`、`top_p`：生成参数。
- `do_sample: false`：本次新增的稳定性参数。禁用采样后 smoke 结果更可复现。
- `max_new_tokens: 64`：限制答案长度，减少 smoke 测试显存和时间消耗。

## 4. 环境脚本：`scripts/setup_env.sh`

这个脚本用于创建完整独立 Conda 环境，适合想和现有 Python 环境隔离时使用。

### 4.1 Shell 安全设置

位置：`scripts/setup_env.sh:1-6`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-$ROOT_DIR/.envs/mirage-py39}"
CONDA_BIN="${CONDA_BIN:-}"
```

含义：

- `set -Eeuo pipefail`：让脚本在命令失败、变量未定义、管道失败时立刻退出，避免后续步骤在错误环境里继续跑。
- `ROOT_DIR`：自动定位仓库根目录。
- `ENV_PREFIX`：默认把 Conda 环境建到 `.envs/mirage-py39`，不污染系统环境。
- `CONDA_BIN`：允许外部指定 conda 路径。

### 4.2 代理 fallback

位置：`scripts/setup_env.sh:8-28`

这个函数按三段式执行网络命令：

1. 先直连。
2. 失败后走远端 `127.0.0.1:7890`。
3. 仍失败后走 SSH 反向代理 `127.0.0.1:17890`。

这样实现了用户要求的网络策略：下载依赖时先尝试正常网络，不行再尝试代理，而且代理只在当前命令子 shell 中生效，不写入 Git。

### 4.3 查找 Conda

位置：`scripts/setup_env.sh:30-49`

`find_conda()` 的逻辑：

- 如果用户设置了 `CONDA_BIN`，直接用它。
- 否则查 `PATH` 里的 `conda`。
- 再尝试几个常见路径，例如 `$HOME/miniconda3/bin/conda`。
- 找不到就返回失败。

这让脚本在不同服务器上有一定可移植性。

### 4.4 已存在环境则复用

位置：`scripts/setup_env.sh:51-55`

如果 `$ENV_PREFIX/bin/python` 已存在，就打印 Python 版本并退出。这样重复运行脚本不会反复重建环境。

### 4.5 移除 Conda prefix 并创建环境

位置：`scripts/setup_env.sh:57-68`

关键逻辑：

- 从 `MIRAGE.yaml` 中过滤掉 `prefix:`。
- 用 `conda env create -p "$ENV_PREFIX"` 创建环境。

为什么要过滤 `prefix:`：

- 上游 `MIRAGE.yaml` 可能带有作者本机路径。
- 如果保留，会导致环境被创建到不存在或不合适的位置。
- 本仓库统一使用项目本地 `.envs/`，便于复现和清理。

## 5. Overlay 环境脚本：`scripts/setup_zqllms_overlay.sh`

这个脚本是本次服务器复现真正使用的方案。

背景是：AgentDojo 已经有 `/home/hqdeng7/.conda/envs/zqllms/bin/python`，里面已有 PyTorch、Transformers、CUDA 等大依赖。重新建完整环境成本高，所以本次只把 MIRAGE 缺少的小包装到项目本地 overlay。

### 5.1 默认 Python 和 overlay 目录

位置：`scripts/setup_zqllms_overlay.sh:4-7`

- `PYTHON` 默认指向 AgentDojo 的 `zqllms` Python。
- `OVERLAY` 默认是 `.envs/zqllms-overlay`。

这表示运行时会复用原 Python 解释器，但额外从 overlay 目录加载补装包。

### 5.2 代理 fallback

位置：`scripts/setup_zqllms_overlay.sh:8-28`

逻辑和 `setup_env.sh` 一样：直连、远端 7890、本地反向代理 17890。

### 5.3 Python 存在性检查

位置：`scripts/setup_zqllms_overlay.sh:30-33`

如果找不到 `/home/hqdeng7/.conda/envs/zqllms/bin/python`，脚本直接退出。这样可以避免把包安装到错误 Python 环境里。

### 5.4 安装缺失依赖

位置：`scripts/setup_zqllms_overlay.sh:35-49`

安装方式：

```bash
pip install --target "$OVERLAY" --no-cache-dir --no-deps ...
```

含义：

- `--target "$OVERLAY"`：包安装到项目目录，不写入 AgentDojo 原环境。
- `--no-deps`：不自动升级已有大依赖，避免破坏 PyTorch / Transformers / CUDA 组合。
- 安装的主要包：
  - `inseq`：模型归因核心库。
  - `captum`：归因算法依赖。
  - `nltk`：句子切分。
  - `rouge-score`：ROUGE 评估。
  - `jsonlines`、`jaxtyping`、`typeguard` 等：运行依赖。

### 5.5 导入验证

位置：`scripts/setup_zqllms_overlay.sh:51-62`

脚本最后用 `PYTHONPATH="$OVERLAY"` 启动 Python，检查：

- `torch`
- `transformers`
- `inseq`
- `nltk`
- `jsonlines`
- `captum`

同时打印 `torch.cuda.is_available()`，确认 GPU 可用。

## 6. Smoke 入口脚本：`scripts/run_smoke_deepseek.sh`

这个脚本把整个 MIRAGE DeepSeek smoke 串起来，是汇报时最重要的入口。

### 6.1 初始化路径

位置：`scripts/run_smoke_deepseek.sh:1-14`

核心变量：

- `ROOT_DIR`：仓库根目录。
- `ENV_PREFIX`：完整 Conda 环境路径。
- `OVERLAY`：zqllms overlay 路径。
- `PYTHON`：实际使用的 Python。
- `CONFIG`：DeepSeek smoke 配置。
- `OUT`：生成答案 JSON 路径。
- `INTERNAL`：内部归因 JSON 路径。

判断 Python 的逻辑：

- 如果 AgentDojo `zqllms` Python 存在，并且 overlay 目录存在，就用它。
- 否则使用完整环境 `.envs/mirage-py39/bin/python`。
- 用户也可以通过环境变量 `PYTHON=/path/to/python` 覆盖。

### 6.2 设置 PYTHONPATH 和 NLTK_DATA

位置：`scripts/run_smoke_deepseek.sh:16-20`

如果 overlay 存在，就把 overlay 加到 `PYTHONPATH` 前面。这样 Python 会优先从 `.envs/zqllms-overlay` 找 `inseq` 等包。

`NLTK_DATA` 指向项目本地 `.nltk_data`，避免把 NLTK 数据写到用户全局目录。

### 6.3 代理 fallback 函数

位置：`scripts/run_smoke_deepseek.sh:22-42`

和 setup 脚本一样，负责下载数据或依赖时的网络重试：

1. 直连。
2. 远端 `127.0.0.1:7890`。
3. 本地反向代理 `127.0.0.1:17890`。

### 6.4 加载 `.env`

位置：`scripts/run_smoke_deepseek.sh:44-50`

如果仓库根目录存在 `.env`，脚本通过 `source .env` 加载环境变量。

注意：

- `.env` 不提交到 Git。
- 它主要保存 token、代理、API key 等运行时私密配置。

### 6.5 Python 环境检查

位置：`scripts/run_smoke_deepseek.sh:52-66`

检查两件事：

- Python 可执行文件是否存在。
- 能否 import `torch`、`transformers`、`inseq`，以及 CUDA 是否可用。

如果 `torch.cuda.is_available()` 是 false，脚本直接失败，因为 MIRAGE attribution 加载 14B 模型需要 GPU。

### 6.6 模型目录检查

位置：`scripts/run_smoke_deepseek.sh:68`

```bash
test -d /home/intern/models/DeepSeek-R1-Distill-Qwen-14B
```

这里明确要求 DeepSeek 本地 Hugging Face 模型目录存在。MIRAGE 不能只使用 vLLM 服务，因为归因需要访问模型内部梯度。

### 6.7 数据准备

位置：`scripts/run_smoke_deepseek.sh:72-74`

如果完整 ALCE 数据和 smoke fixture 都不存在，就运行 `0_download_data.sh`。

本仓库已经提交了 `smoke_data/eli5_eval_bm25_top100_smoke.json`，所以 smoke 默认不需要下载大数据。

### 6.8 生成答案

位置：`scripts/run_smoke_deepseek.sh:76-81`

如果 `OUT` 不存在，或者设置了 `FORCE_GENERATE=1`，就执行：

```bash
python run.py --config "$CONFIG"
```

生成结果写入：

```text
sec5_longQA/result/selfcitation/eli5-DeepSeek-R1-Distill-Qwen-14B-bm25-smoke-shot0-ndoc2-42-quick_test1.json
```

### 6.9 计算 MIRAGE attribution

位置：`scripts/run_smoke_deepseek.sh:84-89`

如果 `INTERNAL` 不存在，或者设置了 `FORCE_ATTRIBUTION=1`，就执行：

```bash
python mirage_attribute.py --f "$OUT"
```

输出：

```text
sec5_longQA/internal_selfcitation/_home_intern_models_deepseek-r1-distill-qwen-14b-shot0-seed42-0.json
```

### 6.10 生成 citation

位置：`scripts/run_smoke_deepseek.sh:92-97`

执行：

```bash
python mirage_cite.py --f "$OUT" --CTI 1 --CCI -5
```

含义：

- `CTI 1`：只保留比平均 CTI 高 1 个标准差以上的输出 token。
- `CCI -5`：保留上下文贡献分数 top 5% 范围。

输出：

```text
*.mirage_cite_CTI_1_CCI_-5
```

### 6.11 轻量评估

位置：`scripts/run_smoke_deepseek.sh:100-103`

执行：

```bash
python eval.py --f "$OUT.mirage_cite_CTI_1_CCI_-5" --citations --citation_regex_only
```

这里使用 `--citation_regex_only`，只检查引用格式和引用编号范围，不加载很大的 AutoAIS 模型。目标是 smoke 快速验收，不是完整论文评测。

## 7. 答案生成：`sec5_longQA/run.py`

`run.py` 负责把问题和检索文档拼成 prompt，然后调用模型生成答案。

### 7.1 日志和依赖

位置：`sec5_longQA/run.py:1-19`

主要依赖：

- `transformers.AutoTokenizer`：本地模型 tokenizer。
- `yaml`：读取配置文件。
- `nltk.sent_tokenize`：句子处理。
- `utils`：prompt 构造和模型加载。
- `SearcherWithinDocs`：交互式检索分支使用。

### 7.2 `remove_citations`

位置：`sec5_longQA/run.py:21-22`

作用是从一句话里删除 `[1]`、`[2]` 这种 citation 标记，便于后续计算或重新加引用。

### 7.3 `LLM.__init__`

位置：`sec5_longQA/run.py:24-54`

这个类统一封装两类模型：

- API 模型：OpenAI / Azure OpenAI。
- 本地模型：Hugging Face causal LM。

本次 DeepSeek 复现走的是本地模型分支：

```python
self.model, self.tokenizer = load_model(args.model)
```

这里的 `args.model` 是：

```text
/home/intern/models/DeepSeek-R1-Distill-Qwen-14B
```

### 7.4 `LLM.generate` 的长度保护

位置：`sec5_longQA/run.py:56-65`

如果 prompt 太长导致剩余生成 token 数小于等于 0，就返回空字符串；如果剩余 token 少于 50，会打印 warning。

这样做是为了避免超过模型上下文长度。

### 7.5 API 模型生成分支

位置：`sec5_longQA/run.py:66-130`

这部分支持 OpenAI / Azure：

- Chat API：`ChatCompletion.create`
- Completion API：`Completion.create`

本次 DeepSeek smoke 不走这里。保留它是为了兼容上游代码。

### 7.6 本地模型生成分支

位置：`sec5_longQA/run.py:131-151`

主要步骤：

1. 用 tokenizer 把 prompt 转成 tensor。
2. 构造 stop token。
3. 针对 Llama、Zephyr、Mistral、Mamba、DeepSeek、Qwen 清理 `unk_token_id`。
4. 调用 `model.generate()`。
5. 只 decode 新生成的 token，不包含 prompt。

本次新增/适配点：

- `deepseek` 和 `qwen` 被纳入 stop token 清理逻辑。
- `do_sample=args.do_sample` 被加入 `generate()`，便于 smoke 中禁用采样。

### 7.7 CLI 参数定义

位置：`sec5_longQA/run.py:154-220`

参数分为几组：

- prompt 和数据：`prompt_file`、`eval_file`、`quick_test`
- ICL 设置：`ndoc`、`shot`、`seed`
- 模型：`model`、`openai_api`、`azure`
- 解码：`temperature`、`top_p`、`do_sample`、`max_new_tokens`
- 交互模式：`interactive`、`interactive_query`、`retriever`
- prompt 模式：`standard`

### 7.8 读取 YAML 配置

位置：`sec5_longQA/run.py:222-229`

逻辑是：

1. 先解析命令行，拿到 `--config`。
2. 如果有 config，就用 `yaml.safe_load()` 读入配置。
3. `parser.set_defaults(**config)` 把 YAML 的字段变成默认参数。
4. 再 parse 一次，使 YAML 和命令行统一到 `args`。

所以 smoke 配置文件里的字段最终都会变成 `args.xxx`。

### 7.9 设置模型最大上下文长度

位置：`sec5_longQA/run.py:231-256`

根据模型名设置 `args.max_length`：

- GPT / turbo / gpt-4：按 API 模型上下文设定。
- Llama2：4096。
- Zephyr / Mistral：32768。
- Llama3：4096。
- DeepSeek / Qwen：4096。

本次适配点是增加 DeepSeek/Qwen 分支，避免默认上下文长度不明确。

### 7.10 加载 prompt 和 eval 数据

位置：`sec5_longQA/run.py:260-268`

执行：

- 初始化 LLM。
- 设置随机种子。
- 读取 prompt JSON。
- 读取 eval JSON。

### 7.11 构造 few-shot demonstration

位置：`sec5_longQA/run.py:269-285`

如果 `shot > 0`，会从 prompt 文件里的 demos 随机抽样，拼到 `head_prompt`。

本次 smoke 配置是 `shot: 0`，所以这里不会加入示例，只保留任务 instruction 和当前问题文档。

### 7.12 quick test 抽样

位置：`sec5_longQA/run.py:286-290`

如果 `quick_test` 不为空，就从 eval 数据中随机抽样指定数量的样本。

本次是 `quick_test: 1`，只跑一条，降低 smoke 成本。

### 7.13 给每条样本构造 prompt

位置：`sec5_longQA/run.py:291-309`

对每个 eval item：

1. 调用 `make_demo(..., test=True)` 构造测试 prompt。
2. 截取前 `ndoc` 篇文档。
3. 把实际使用的文档写回 `eval_data[idx]['docs']`。
4. 如果文档不足，记录 warning。

这一步决定了模型真正能看到哪些检索文档。

### 7.14 交互检索准备

位置：`sec5_longQA/run.py:310-315`

如果打开 interactive search，并且使用 GTR dense retriever，就加载 sentence-transformer。

本次 smoke 不使用 interactive 模式。

### 7.15 主生成循环

位置：`sec5_longQA/run.py:316-423`

对每条样本执行生成。

非交互模式下，本次实际走的是：

```python
output_array.append(llm.generate(prompt, min(args.max_new_tokens, args.max_length-prompt_len)))
item['prompt'] = prompt
```

也就是：

- 输入：拼好的 RAG prompt。
- 输出：DeepSeek 生成的答案。
- 存储：写回 `item['output']`。

交互模式代码保留了上游的 Check/Search/Output/End 多轮行为，但 smoke 没启用。

### 7.16 输出结果命名和保存

位置：`sec5_longQA/run.py:427-482`

输出文件名由这些字段组成：

```text
{dataset_name}-{model_name}-{tag}-shot{shot}-ndoc{ndoc}-{seed}
```

本次得到：

```text
eli5-DeepSeek-R1-Distill-Qwen-14B-bm25-smoke-shot0-ndoc2-42-quick_test1.json
```

因为 `standard: false`，结果写入：

```text
result/selfcitation/
```

结果 JSON 结构：

```json
{
  "args": { "...": "运行参数" },
  "data": [
    {
      "question": "...",
      "docs": [...],
      "prompt": "...",
      "output": "模型生成答案"
    }
  ]
}
```

## 8. 内部归因：`sec5_longQA/mirage_attribute.py`

这个文件是 MIRAGE 的核心。它把“生成答案”变成“答案 token 对检索上下文 token 的依赖分数”。

### 8.1 依赖和 inseq

位置：`sec5_longQA/mirage_attribute.py:1-17`

主要依赖：

- `torch`：模型和 GPU。
- `inseq`：归因库。
- `AttributeContextArgs` / `attribute_context_with_model`：MIRAGE attribution 入口。
- `utils.load_model`：加载本地 HF 模型。

### 8.2 读取生成结果并选择保存目录

位置：`sec5_longQA/mirage_attribute.py:22-35`

脚本参数：

```bash
python mirage_attribute.py --f <generation_json>
```

逻辑：

- 读取 `run.py` 生成的 JSON。
- 如果是 self-citation 模式，保存到 `internal_selfcitation/`。
- 如果是 standard 模式，保存到 `internal_standard/`。

本次是 self-citation。

### 8.3 加载 prompt 和模型

位置：`sec5_longQA/mirage_attribute.py:36-46`

步骤：

1. 读取 prompt 模板。
2. 用 `load_model()` 加载 DeepSeek HF 模型。
3. 用 `inseq.load_model(model, "saliency", ...)` 包装模型。

这里的 `saliency` 表示使用基于梯度的 saliency attribution 方法。

为什么不能用 vLLM：

- vLLM 只提供推理服务。
- MIRAGE 需要模型内部梯度和 token-level attribution。
- 所以必须加载本地 HF 模型对象。

### 8.4 构造 stop token

位置：`sec5_longQA/mirage_attribute.py:48-58`

逻辑：

- 把换行相关 token 加入 stop 列表。
- 转成 token id。
- 加上模型 `eos_token_id`。
- 对 Llama / Zephyr / Mistral / DeepSeek / Qwen 移除 `unk_token_id`。

本次适配点：

- 使用公开 API `tokenizer.convert_tokens_to_ids()`，替代私有 `_convert_token_to_id()`。
- 增加 DeepSeek/Qwen 分支，避免把无效 unknown token 当成 stop token。

### 8.5 选择 decoder 输入输出分隔符

位置：`sec5_longQA/mirage_attribute.py:59-73`

`decoder_input_output_separator` 用来告诉 inseq：prompt 和答案之间如何分隔。

不同模型 tokenization 习惯不同：

- Zephyr：使用 `'\n '`。
- Llama2 / Llama3 / Mistral / DeepSeek / Qwen：使用 `' '`。

本次新增：

- Llama3 支持。
- DeepSeek/Qwen 支持。

### 8.6 遍历每条生成样本

位置：`sec5_longQA/mirage_attribute.py:77-87`

对 `data['data']` 中每个 item：

- 如果 `output` 为空，跳过。
- 清理输出首尾空格。
- 合并过多换行。
- 取出 `doc_list`。

### 8.7 构造 attribution 输入

位置：`sec5_longQA/mirage_attribute.py:89-94`

这里把一个 RAG 例子拆成 MIRAGE 需要的几个部分：

- `input_context_text`：所有检索文档拼起来的上下文。
- `input_current_text`：当前问题。
- `input_template`：包含 `{context}` 和 `{current}` 的 prompt 模板。
- `contextless_input_current_text`：没有检索文档时的问题 prompt。
- `output_current_text`：模型生成答案，去掉原始 citation。

MIRAGE 要比较“带上下文”和“不带上下文”的行为差异，所以这里需要同时准备 context 和 contextless 输入。

### 8.8 首条样本打印调试信息

位置：`sec5_longQA/mirage_attribute.py:95-114`

只在 `idx == 0` 时打印：

- 文档上下文。
- 问题。
- prompt 模板。
- 无上下文 prompt。
- 输出答案。
- decoder separator。

这对检查 attribution 输入是否拼错非常重要。

### 8.9 attribution 输出路径

位置：`sec5_longQA/mirage_attribute.py:115`

路径由模型名、shot、seed、样本编号组成。

本次 DeepSeek smoke 输出类似：

```text
internal_selfcitation/_home_intern_models_deepseek-r1-distill-qwen-14b-shot0-seed42-0.json
```

### 8.10 `AttributeContextArgs`

位置：`sec5_longQA/mirage_attribute.py:116-148`

这是 MIRAGE attribution 的核心参数对象。

关键字段：

- `model_name_or_path`：模型路径。
- `input_context_text`：检索文档。
- `input_current_text`：问题。
- `output_current_text`：生成答案。
- `input_template`：prompt 结构。
- `contextless_input_current_text`：去掉上下文后的输入。
- `attributed_fn="contrast_prob_diff"`：用有上下文和无上下文的概率差异做归因目标。
- `attribution_method="saliency"`：使用 saliency attribution。
- `attribution_kwargs={"logprob": True}`：对 log probability 做归因。
- `save_path`：归因 JSON 保存路径。
- `model_kwargs`：使用 FP16、自动分配 GPU、按剩余显存设置 max memory。
- `generation_kwargs`：生成参数和 stop token。
- `decoder_input_output_separator`：模型 prompt 和输出之间的分隔符。

可以汇报成：这一步计算的是“答案 token 对上下文 token 的贡献”，而不是简单文本匹配。

### 8.11 执行 attribution

位置：`sec5_longQA/mirage_attribute.py:150`

```python
attribute_context_with_model(lm_rag_prompting_example, model_mirage)
```

这行会实际运行 inseq，并把结果写到 `save_path`。

归因结果里后续最重要的字段包括：

- `input_context_tokens`：检索上下文 token。
- `cti_scores`：输出 token 的 context-sensitive 分数。
- `cci_scores`：每个输出 token 对各上下文 token 的贡献分布。

## 9. 引用生成：`sec5_longQA/mirage_cite.py`

这个文件把 MIRAGE 的 token attribution 结果转成文档级引用。

### 9.1 `mirage_cite()` 函数

位置：`sec5_longQA/mirage_cite.py:23-54`

输入：

- `res_mirage`：某条样本的 attribution JSON。
- `cti_threshold`：筛选输出 token 的阈值。
- `start_pos_sent` / `end_pos_sent`：当前句子在输出 token 序列中的范围。
- `topk_CCI`：上下文 token 贡献筛选策略。
- `doc_seps`：哪些 token 是文档分隔符。

处理逻辑：

1. 遍历 `res_mirage['cci_scores']`。
2. 只处理属于当前句子范围内的输出 token。
3. 如果该 token 的 CTI 分数超过阈值，就继续看它对上下文 token 的 CCI 分数。
4. 根据 `topk_CCI` 筛掉低贡献上下文 token。
5. 把剩余贡献按文档分隔符聚合成文档级分数。
6. 返回每篇文档对应的贡献值数组。

直观解释：

- CTI 判断“这个答案 token 是否真的受上下文影响”。
- CCI 判断“如果受上下文影响，它具体依赖上下文里的哪些 token”。
- 文档级聚合就是把 token-level 贡献汇总到 document-level citation。

### 9.2 CLI 参数

位置：`sec5_longQA/mirage_cite.py:57-78`

参数：

- `--f`：`run.py` 生成的答案 JSON。
- `--CTI`：CTI 阈值策略。
- `--CCI`：CCI 阈值策略。

本次 smoke 使用：

```bash
--CTI 1 --CCI -5
```

### 9.3 选择 attribution 目录

位置：`sec5_longQA/mirage_cite.py:80-88`

如果是 self-citation 模式，从 `internal_selfcitation/` 读取 attribution；否则从 `internal_standard/` 读取。

prefix 由模型名、shot、seed 构造，和 `mirage_attribute.py` 的保存路径保持一致。

### 9.4 加载 tokenizer

位置：`sec5_longQA/mirage_cite.py:89-95`

加载同一个模型的 tokenizer，目的是把输出句子切成 token，确定每句话对应哪些输出 token。

这一步必须和 attribution 阶段 tokenizer 对齐，否则句子 token 范围会错。

### 9.5 遍历每条样本

位置：`sec5_longQA/mirage_cite.py:97-109`

清理输出文本，跳过空输出。

这里先去掉原始 citation，因为后面要重新根据 MIRAGE attribution 加 citation。

### 9.6 读取 attribution JSON

位置：`sec5_longQA/mirage_cite.py:111-114`

读取当前样本对应的内部归因结果。

### 9.7 计算 CTI 阈值

位置：`sec5_longQA/mirage_cite.py:116-119`

如果 `CTI >= 0`：

```text
cti_threshold = mean(cti_scores) + CTI * std(cti_scores)
```

当 `CTI=1` 时，只有比平均值高一个标准差以上的 token 会被认为是强上下文依赖 token。

### 9.8 切分答案句子

位置：`sec5_longQA/mirage_cite.py:121-124`

普通数据集用 `sent_tokenize(output)` 切句。

如果是 QAMPARI，则按逗号切实体答案。

本次 ELI5 smoke 走普通句子切分。

### 9.9 找文档分隔 token

位置：`sec5_longQA/mirage_cite.py:125-129`

`input_context_tokens` 中的换行 token 被视为文档分隔符。不同 tokenizer 会把换行编码成不同 token，例如 Llama 常见 `<0x0A>`，DeepSeek/Qwen 这次结果里是 `Ċ`。代码会同时识别 `<0x0A>`、`Ċ` 和普通 `\n`，后面再按这些分隔符把上下文 token contribution 汇总成每篇文档的分数。

### 9.10 对每个句子生成引用

位置：`sec5_longQA/mirage_cite.py:131-176`

对每个句子：

1. 记录句子原来的引用。
2. 用 tokenizer 计算当前句子的 token 范围。
3. 调用 `mirage_cite()` 得到每篇文档的贡献分数。
4. 过滤掉贡献为 0 的文档。
5. 按贡献分数从高到低排序。
6. 把文档编号转成 `[1]`、`[2]`。
7. 把新 citation 放在句子开头。

也就是说，最终 citation 是由 attribution 结果决定的，不是由模型原始输出决定的。

### 9.11 保存加引用后的结果

位置：`sec5_longQA/mirage_cite.py:181-190`

输出文件名在原生成结果后追加：

```text
.mirage_cite_CTI_1_CCI_-5
```

这个文件结构仍然和原结果 JSON 一样，只是 `data[i].output` 已经被改写成带 MIRAGE citation 的答案。

## 10. 评估：`sec5_longQA/eval.py`

`eval.py` 负责对答案和 citation 做评估。本次 smoke 使用的是轻量评估。

### 10.1 依赖和模型常量

位置：`sec5_longQA/eval.py:1-35`

主要内容：

- 下载 NLTK `punkt`。
- 引入 ROUGE、Transformers pipeline。
- 定义 QA 模型 `gaotianyu1350/roberta-large-squad`。
- 定义 AutoAIS 模型 `google/t5_xxl_true_nli_mixture`。

AutoAIS 很大，所以 smoke 默认不加载它。

### 10.2 基础文本指标函数

位置：`sec5_longQA/eval.py:38-192`

包括：

- `compute_f1()`：两个字符串的 token F1。
- `compute_exact()`：规范化后是否完全相等。
- `exact_presence()`：短答案是否出现在上下文。
- `compute_rouge()`：计算 ROUGE-Lsum。
- `compute_str_em()`：ASQA 风格短答案 exact match。
- `compute_len()`：平均输出长度。

本次 ELI5 smoke 主要用到：

- `compute_len`
- `compute_rouge`
- `compute_str_em` 返回 0，因为 smoke 数据没有 ASQA `qa_pairs`。

### 10.3 QA / MAUVE / Claims 评估

位置：`sec5_longQA/eval.py:195-301`

这些是上游完整实验保留的评估模块：

- `compute_qa()`：加载 QA pipeline 评估答案是否覆盖短答案。
- `compute_mauve()`：计算文本分布相似度。
- `compute_claims()`：用 NLI 判断 claims 是否被答案支持。

本次 smoke 不启用这些，因为它们会额外下载或加载模型。

### 10.4 AutoAIS citation 评估

位置：`sec5_longQA/eval.py:304-433`

`compute_autoais()` 使用 NLI 模型判断：

- 每个带 citation 的句子是否被引用文档支持。
- 多引用场景下是否存在 over-citation。

这是更接近论文完整评测的 citation 指标，但需要加载 `google/t5_xxl_true_nli_mixture`，显存和下载成本很高。

本次 smoke 不跑它，避免把验收变成大模型评估任务。

### 10.5 轻量 citation regex 评估

位置：`sec5_longQA/eval.py:436-464`

这是本次新增的 smoke 评估函数。

它不判断语义支持，只检查 citation 格式：

- 总句子数。
- 有 citation 的句子数。
- citation 编号是否在文档范围内。
- 平均每句 citation 数。

输出字段：

- `citation_regex_sentences`
- `citation_regex_cited_sentences`
- `citation_regex_valid_sentences`
- `citation_regex_rec`
- `citation_regex_valid_rate`
- `citation_regex_avg_citations`

这适合作为快速 smoke：证明 citation 生成链路跑通，且生成的编号合法。

### 10.6 参数定义

位置：`sec5_longQA/eval.py:510-524`

关键参数：

- `--f`：待评估 JSON。
- `--no_rouge`：不算 ROUGE。
- `--qa`：启用 QA 评估。
- `--mauve`：启用 MAUVE。
- `--citations`：启用 citation 评估。
- `--citation_regex_only`：只做轻量 citation regex 检查。
- `--claims_nli`：启用 claims NLI。

本次 smoke 用：

```bash
--citations --citation_regex_only
```

### 10.7 数据读取和清理

位置：`sec5_longQA/eval.py:525-559`

逻辑：

- 读取 JSON。
- 取出 `data`。
- 如果是 QAMPARI，自动关闭部分指标。
- 清理输出里的空白、换行和 `<|im_end|>`。
- 对非 citation 指标，复制一份去掉 citation 的 `normalized_data`。

这样 ROUGE 等内容指标不会被 `[1]` 这种引用符号干扰。

### 10.8 汇总指标并保存

位置：`sec5_longQA/eval.py:560-581`

按参数启用指标：

- 永远计算 `length`。
- 计算 `str_em` 和 `str_hit`。
- 默认计算 `rougeLsum`。
- 如果 `--citations`：
  - 有 `--citation_regex_only` 就跑 regex 检查。
  - 否则跑 AutoAIS。

最终写入：

```text
<input_file>.score
```

本次 smoke 得到的 `.score` 包含：

```json
{
  "length": 51.0,
  "str_em": 0,
  "str_hit": 0,
  "rougeLsum": 18.666666666666668,
  "citation_regex_sentences": 2,
  "citation_regex_cited_sentences": 1,
  "citation_regex_valid_sentences": 1,
  "citation_regex_rec": 50.0,
  "citation_regex_valid_rate": 50.0,
  "citation_regex_avg_citations": 0.5
}
```

## 11. 公共工具：`sec5_longQA/utils.py`

`utils.py` 提供多个主流程共享的函数。

### 11.1 `normalize_answer`

位置：`sec5_longQA/utils.py:12-27`

用于评估时规范化答案：

- 转小写。
- 删除标点。
- 删除英文冠词 `a/an/the`。
- 合并空白。

这样可以减少格式差异对 exact match / F1 的影响。

### 11.2 `remove_citations`

位置：`sec5_longQA/utils.py:29-30`

删除 `[1]`、`[2]` citation 标记。

用途：

- 评估答案内容时去掉引用。
- attribution 前去掉模型原始 citation。
- citation 重写前清理旧引用。

### 11.3 `get_max_memory`

位置：`sec5_longQA/utils.py:33-39`

逻辑：

- 查询当前 GPU 空闲显存。
- 每张卡预留 6GB。
- 返回 Transformers `from_pretrained(..., max_memory=...)` 能使用的字典。

作用是避免加载 14B 模型时把 GPU 显存全部占满。

### 11.4 `make_doc_prompt`

位置：`sec5_longQA/utils.py:42-52`

把单篇文档填入 prompt 模板。

输入文档通常有：

- `title`
- `text`

模板里会替换：

- `{ID}`：文档编号，从 1 开始。
- `{T}`：标题。
- `{P}`：正文。

### 11.5 `get_shorter_text`

位置：`sec5_longQA/utils.py:55-70`

如果使用 summary 或 extraction 模式，这个函数会挑选更短的文档字段。

逻辑：

- 如果文档没有对应字段，至少保留第一篇全文文档。
- 如果字段里标记 irrelevant，就跳过。
- 直到收集到 `ndoc` 篇。

本次 smoke 不使用 summary/extraction。

### 11.6 `make_demo`

位置：`sec5_longQA/utils.py:73-99`

这是 prompt 构造的核心函数。

它把一个样本填入 prompt 模板：

- `{INST}` 替换成任务 instruction。
- `{Q}` 替换成问题。
- `{D}` 替换成检索文档。
- `{A}` 替换成答案。

如果 `test=True`，表示当前是要让模型回答的问题，所以 `{A}` 会被删掉，留给模型生成。

如果 `test=False`，表示 few-shot demonstration，所以会把标准答案填进去。

### 11.7 `load_model`

位置：`sec5_longQA/utils.py:102-133`

加载 Hugging Face causal LM 和 tokenizer。

关键参数：

- `device_map='auto'`：自动把模型分配到 GPU。
- `torch_dtype=torch.float16`：FP16 加载，降低显存。
- `max_memory=get_max_memory()`：限制每张 GPU 使用量。
- `load_in_8bit=int8`：可选 int8，本次不用。
- `use_fast=False`：使用 slow tokenizer，和 attribution/token 对齐更稳。
- `padding_side="left"`：左 padding，适配 causal LM 批处理习惯。

本次 DeepSeek 生成和归因都通过这个函数加载模型。

## 12. 文档内搜索器：`sec5_longQA/searcher.py`

`searcher.py` 主要服务于 `run.py` 的 interactive 模式。虽然 smoke 不启用，但它是上游 LongQA 代码的一部分。

### 12.1 文档转文本

位置：`sec5_longQA/searcher.py:8-12`

- `doc_to_text_tfidf()`：把标题和正文用空格拼接，给 TF-IDF 用。
- `doc_to_text_dense()`：把标题和正文用句号拼接，给 dense retriever 用。

### 12.2 `SearcherWithinDocs.__init__`

位置：`sec5_longQA/searcher.py:15-28`

初始化文档内检索器。

支持两类 retriever：

- `tfidf`：用 `TfidfVectorizer` 建文档向量。
- `gtr`：用 sentence-transformer 编码文档向量。

如果传入未知 retriever，抛 `NotImplementedError`。

### 12.3 `search`

位置：`sec5_longQA/searcher.py:30-44`

输入 query，返回最相关的一篇文档编号。

- TF-IDF 分支：计算 query 和每篇文档的 cosine similarity。
- GTR 分支：计算 query embedding 和文档 embedding 的点积。

本次 smoke 不走这条路径，因为不是 interactive RAG。

## 13. 安全和 Git 管理

文件：`.gitignore`

关键忽略项：

- `.env`、`.env.*`：避免提交 token 和 API key。
- `.envs/`：本地环境。
- `.nltk_data/`、`.pip-cache/`、`.conda_pkgs/`：缓存。
- `sec5_longQA/data/`：完整下载数据。
- `sec5_longQA/internal_selfcitation/`、`sec5_longQA/internal_standard/`：归因结果。
- `sec5_longQA/result/`：生成答案、citation 和 score。
- `models/`、`hf_cache/`、`.huggingface/`：模型和 HF 缓存。

这样 GitHub 仓库只包含代码、配置和 smoke fixture，不包含密钥、大模型、下载数据或实验产物。

## 14. 本次复现通过了什么

在 `hqdeng-server` 的 `/mnt/data2/zyc/mirage` 上已经验证：

1. Python 能 import `torch`、`transformers`、`inseq`。
2. CUDA 可用。
3. DeepSeek HF 模型目录存在。
4. `run.py` 能生成一条 ELI5 smoke 答案。
5. `mirage_attribute.py` 能生成 `internal_selfcitation/*.json`。
6. `mirage_cite.py --CTI 1 --CCI -5` 能生成带 MIRAGE citation 的 JSON。
7. `eval.py --citations --citation_regex_only` 能生成 `.score`。

验收结果中的 citation regex 指标说明：

- 共有 2 个句子。
- 其中 1 个句子带 citation。
- 这个 citation 编号合法。
- citation 覆盖率是 50%。

## 15. 汇报时可以这样讲

可以直接使用下面这段作为口头汇报：

> 我们复现的是 MIRAGE 在 LongQA 场景下的 answer attribution 链路。普通 RAG 生成答案后，引用往往来自模型自己生成，可信度有限。MIRAGE 的做法是加载同一个生成模型，通过 inseq 对模型内部 token 概率做 saliency attribution，计算答案 token 对检索上下文 token 的依赖。然后代码把 token 级别的贡献按句子和文档聚合，自动生成 `[1]`、`[2]` 这样的 citation。本次适配把官方 Llama/Zephyr 代码扩展到 AgentDojo 服务器已有的 DeepSeek-R1-Distill-Qwen-14B，本地 HF 模型直接参与梯度归因。smoke 测试已经跑通生成、内部归因、citation 重写和轻量评估四个阶段。

## 16. 限制和后续工作

当前仓库支持的是快速复现 smoke，而不是完整论文实验。

主要限制：

- smoke 只跑 1 条 ELI5 样本。
- citation 评估使用 regex-only，不加载完整 AutoAIS。
- 没跑 Section 5 全量数据、多 seed、多模型对比。
- 依赖服务器已有 DeepSeek HF 模型目录和 CUDA。

后续如果要做正式实验，可以扩展：

1. 删除或替换 smoke fixture，下载完整 ALCE 数据。
2. 把 `quick_test` 调大或去掉。
3. 跑完整 `mirage_attribute.py` 和 `mirage_cite.py`。
4. 配好 AutoAIS 模型缓存后，去掉 `--citation_regex_only`，跑完整 citation precision/recall。
5. 对不同 `CTI` / `CCI` 参数做网格实验。
