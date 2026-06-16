# data_1 DeepSeek MIRAGE v2 修复流水线

v2 不再使用 `data_1/*/history_report/历史成果.txt` 中重复且错配的旧报告。原始 `data_1` 保持只读，新报告写入 `runs/data1-deepseek-mirage-v2/generated_history/`。

## 运行

```bash
cd /mnt/data2/zyc/mirage
bash scripts/data1_deepseek_v2_watchdog.sh
bash scripts/data1_deepseek_v2_status.sh
```

默认只处理排序后的第一个可用 workdir。运行目录是 `runs/data1-deepseek-mirage-v2/`，tmux session 是 `mirage-data1-deepseek-v2`。

默认 CTI 模式是 token 级 inseq saliency。需要提速时使用句子级 CTI：

```bash
cd /mnt/data2/zyc/mirage
CTI_MODE=sentence_logprob FORCE_RERUN=1 bash scripts/data1_deepseek_v2_watchdog.sh
```

快速 smoke 可在 runner 后追加 Python 参数，例如 `--sentence-limit 1 --top-sensitive-sentences 1 --paragraph-doc-topk 1`；默认 `--sentence-limit 0`，正式运行仍处理全部内容句。

安装 crontab watchdog：

```bash
cd /mnt/data2/zyc/mirage
bash scripts/install_data1_deepseek_v2_cron.sh
```

停止当前 v2 任务：

```bash
cd /mnt/data2/zyc/mirage
bash scripts/data1_deepseek_v2_stop.sh
```

## v2 改动

- 用 `BAAI/bge-m3` 做中英文语义 embedding 检索，替换字符 TF-IDF。
- 用 DeepSeek 14B 基于 workdir 资料生成对应主题的中文 `历史成果.txt` 副本。
- manifest 记录原始 history 的 MD5/标题/重复数，以及 generated history 的路径、标题、MD5、标题-主题相似度。
- CTI 前过滤非事实性结构句，包括 Markdown 标题、编号加粗小标题和短冒号承接句；跳过清单写入 `embedding_debug/<alias>/skipped_answer_sentences.json`。
- 每个内容句保留目标句之前的原始答案前文，上下文字段写入 `answer_prefix_chars`、`answer_prefix_excerpt` 和 `answer_prefix_before`。
- `--cti-mode token_saliency` 保留 inseq token CTI，输出 `cti_scores/cci_scores`，并用聚合后的 `sentence_cti` 排序。
- `--cti-mode sentence_logprob` 新增句子级 CTI，用 `question + Top5 资料块 + 答案前文` 对比 `question + 答案前文` 的整句 logprob delta。
- `sentence_cti.jsonl` 只写入成功句；失败句写入 `cti_failed.jsonl`，不参与 Top100 和后续 citation。
- 文档级和段落级扰动都保留答案前文上下文；文档级新增 `delta_vs_contextless_with_answer_prefix`，并暂时同步写旧字段 `delta_vs_question_only`。
- 段落扰动按 Markdown/标题分节，并在节内每 5 句组成一个 chunk，不再把整篇资料当作单段。
- DeepSeek-R1 生成报告时使用 no-think 预填，并且报告生成只用 EOS stop token，避免标题换行处被旧 attribution stop token 截断。
- 运行脚本会优先使用本机已缓存完整的 `BAAI/bge-m3` snapshot；未缓存时仍按直连、远端 `7890`、反向代理 `17890` 的顺序下载。
- v2 watchdog 默认把 tmux 子进程限制到 `CUDA_VISIBLE_DEVICES=3`；需要换 GPU 时可在调用 watchdog 前覆盖该环境变量。

## 产物

- `generated_history/workdir-1/历史成果.txt`
- `manifest.json`
- `embedding_debug/workdir-1/generation_chunks.json`
- `embedding_debug/workdir-1/sentence_contexts.json`
- `embedding_debug/workdir-1/skipped_answer_sentences.json`
- `sentence_cti.jsonl`
- `cti_failed.jsonl`
- `doc_perturbation.jsonl`
- `paragraph_perturbation.jsonl`
- `summary.md`

## 验证要点

- `manifest.json` 中 27 个原始 history 会被标记为同一 MD5；`workdir-1` 的原始标题是“生成式人工智能在美军指挥控制领域的发展现状”。
- `generated_history/workdir-1/历史成果.txt` 应以 `## “星链”系统在俄乌冲突中的军事作用与暴露的问题` 开头，正文围绕星链、俄乌冲突、军事通信、无人机、指挥控制和暴露问题。
- `manifest.json` 中 `validation_errors` 应为空，`title_topic_similarity` 应明显高于阈值。
- token 模式下，`internal_cti/workdir-1/sentence-*.json` 中 `record.cti_method` 必须是 `inseq_saliency`。
- sentence 模式下，`sentence_cti.jsonl` 中 `cti_method` 必须是 `sentence_logprob_delta_with_answer_prefix`，且每条有 `context_avg_logprob/contextless_avg_logprob`。
- 两种模式下，每条 `sentence_cti.jsonl` 都必须包含 `answer_prefix_chars` 和 `answer_prefix_excerpt`。
- `sentence_cti.jsonl` 只在全部有效内容句 CTI 完成后重写；运行中先看 `internal_cti/` 文件数和 status heartbeat。
