# data_1 DeepSeek MIRAGE 两级归因流水线

这个新增流水线面向 `/mnt/data2/zyc/mirage/data_1`。它不会重命名或改写原始 workdir，而是在 `runs/data1-deepseek-mirage/manifest.json` 中把中文目录稳定映射为 `workdir-1`、`workdir-2` 等别名。

## 运行入口

```bash
cd /mnt/data2/zyc/mirage
bash scripts/data1_deepseek_watchdog.sh
bash scripts/data1_deepseek_status.sh
```

默认只跑排序后的第一个可用 workdir，但仍执行 Top100 敏感句逻辑；如果报告不足 100 句，就处理全部句子。后续全量运行可以设置：

```bash
WORKDIR_LIMIT=27 bash scripts/run_data1_deepseek_attribution.sh
```

固定本地模型为 `/home/intern/models/DeepSeek-R1-Distill-Qwen-14B`，因为 CTI 和扰动打分都需要本地模型 logits/梯度能力，不能只用 API。

## 产物

- `manifest.json`：27 个 workdir 的 alias、原目录、历史成果路径、最新 search-results 目录和前 100 个资料映射。
- `sentence_cti.jsonl`：每个 answer 句子的 token CTI、sentence CTI、Top100 标记和所用 Top5 上下文资料。
- `doc_perturbation.jsonl`：每个敏感句子对前 100 篇资料的扰动分数排序。
- `paragraph_perturbation.jsonl`：每个敏感句子的 Top3 文档内段落扰动排序。
- `summary.md`：中文汇报摘要，列出 Top 敏感句、推荐引用资料和关键段落摘录。
- `.running/.heartbeat/.done/.failed`：长跑状态标记。
- `run.log/watchdog.log`：主任务和 watchdog 日志。

## 代码逻辑

1. 扫描 `data_1`：读取每个 `history_report/历史成果.txt`，选择时间戳最新的 `search-results-*`，按 `资料N.txt` 的 N 排序取前 100 篇。
2. 切分 answer：按中英文句末标点和换行切句，得到待归因句子。
3. 句子级 CTI：先用字符 n-gram TF-IDF 从前 100 篇资料中为每个句子检索 Top5，构造紧凑上下文；每篇 CTI 上下文资料默认截断到 600 字，避免 14B saliency 在长上下文上卡住。默认尝试 DeepSeek + inseq saliency 得到 token CTI。若单句 saliency 异常，会降级为 DeepSeek logprob 差分并在该行 `cti_method` 标明。
4. 文档级扰动：对 Top100 敏感句逐篇计算 `avg_logprob(sentence | question + doc_j) - avg_logprob(sentence | question_only)`，delta 越高，表示该资料越能支持该句。
5. 段落级扰动：对每句 Top3 文档逐段移除，计算 `avg_logprob(sentence | full_doc) - avg_logprob(sentence | full_doc_without_paragraph_k)`，importance 越高，表示该段越关键。
6. 长跑管理：watchdog 用 `flock` 防重复，用 tmux session `mirage-data1-deepseek` 跑主任务；`.done` 或 `.failed` 存在时不重复启动。

## crontab

安装脚本只替换自己的标记块，不删除已有 crontab：

```bash
bash scripts/install_data1_deepseek_cron.sh
```

安装后会存在：

```cron
# mirage-data1-deepseek-watchdog-BEGIN
@reboot /bin/bash -lc 'sleep 120; cd /mnt/data2/zyc/mirage && bash scripts/data1_deepseek_watchdog.sh >> runs/data1-deepseek-mirage/watchdog.log 2>&1'
*/10 * * * * /bin/bash -lc 'cd /mnt/data2/zyc/mirage && bash scripts/data1_deepseek_watchdog.sh >> runs/data1-deepseek-mirage/watchdog.log 2>&1'
# mirage-data1-deepseek-watchdog-END
```
