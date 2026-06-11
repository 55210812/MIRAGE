#!/usr/bin/env python
"""Run MIRAGE-style attribution over the local data_1 workdirs.

This pipeline treats each data_1 history report as a fixed answer and the
corresponding search-result files as candidate evidence. It does not edit the
raw data directories; all aliases and state live under the output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

try:
    import inseq
    from inseq.commands.attribute_context.attribute_context import (
        AttributeContextArgs,
        attribute_context_with_model,
    )
except Exception:  # pragma: no cover - handled at runtime on lean envs
    inseq = None
    AttributeContextArgs = None
    attribute_context_with_model = None

from utils import get_max_memory, load_model


MODEL_PATH = "/home/intern/models/DeepSeek-R1-Distill-Qwen-14B"
DEFAULT_DATA_ROOT = "/mnt/data2/zyc/mirage/data_1"
DEFAULT_OUTPUT_DIR = "runs/data1-deepseek-mirage"
SESSION_NAME = "mirage-data1-deepseek"


@dataclass
class Doc:
    doc_id: int
    number: int
    path: str
    title: str
    publish_time: str
    text: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def rewrite_jsonl_replace_alias(
    path: Path,
    alias: str,
    replacement_records: Sequence[Dict[str, Any]],
) -> None:
    kept = [r for r in read_jsonl(path) if r.get("workdir_alias") != alias]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in kept + list(replacement_records):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def finite_float(value: float) -> Optional[float]:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def update_heartbeat(output_dir: Path, **fields: Any) -> None:
    payload = {"updated_at": now_ts(), **fields}
    write_json(output_dir / ".heartbeat", payload)


def clean_question_from_workdir(name: str) -> str:
    question = re.sub(r"^workdir-", "", name)
    question = re.sub(r"-\d{14}$", "", question)
    return question


def search_timestamp(path: Path) -> str:
    match = re.search(r"search-results-(\d+)", path.name)
    return match.group(1) if match else path.name


def doc_number(path: Path) -> int:
    match = re.search(r"资料(\d+)\.txt$", path.name)
    return int(match.group(1)) if match else 10**9


def parse_doc(path: Path, doc_id: int) -> Doc:
    raw = read_text(path).strip()
    title = ""
    publish_time = ""
    text = raw
    title_match = re.search(r"^标题[:：]\s*(.*)$", raw, flags=re.MULTILINE)
    time_match = re.search(r"^发布时间[:：]\s*(.*)$", raw, flags=re.MULTILINE)
    body_match = re.search(r"文本内容[:：]\s*(.*)", raw, flags=re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
    if time_match:
        publish_time = time_match.group(1).strip()
    if body_match:
        text = body_match.group(1).strip()
    if not title:
        title = path.stem
    return Doc(
        doc_id=doc_id,
        number=doc_number(path),
        path=str(path),
        title=title,
        publish_time=publish_time,
        text=text,
    )


def build_manifest(data_root: Path, doc_limit: int) -> List[Dict[str, Any]]:
    workdirs = sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    manifest: List[Dict[str, Any]] = []
    for idx, workdir in enumerate(workdirs, start=1):
        alias = f"workdir-{idx}"
        history_path = workdir / "history_report" / "历史成果.txt"
        search_dirs = sorted(
            [p for p in workdir.iterdir() if p.is_dir() and p.name.startswith("search-results-")],
            key=search_timestamp,
        )
        entry: Dict[str, Any] = {
            "alias": alias,
            "original_name": workdir.name,
            "source_dir": str(workdir),
            "question": clean_question_from_workdir(workdir.name),
            "history_path": str(history_path) if history_path.exists() else "",
            "search_results_dirs": [str(p) for p in search_dirs],
            "search_results_dir": "",
            "doc_count": 0,
            "selected_doc_count": 0,
            "docs": [],
            "skipped": False,
            "skip_reason": "",
        }
        if not history_path.exists():
            entry["skipped"] = True
            entry["skip_reason"] = "missing history_report/历史成果.txt"
            manifest.append(entry)
            continue
        if not search_dirs:
            entry["skipped"] = True
            entry["skip_reason"] = "missing search-results-* directory"
            manifest.append(entry)
            continue
        search_dir = search_dirs[-1]
        doc_paths = sorted(search_dir.glob("资料*.txt"), key=doc_number)
        entry["search_results_dir"] = str(search_dir)
        entry["doc_count"] = len(doc_paths)
        if not doc_paths:
            entry["skipped"] = True
            entry["skip_reason"] = "missing 资料N.txt files"
            manifest.append(entry)
            continue
        selected = doc_paths[:doc_limit]
        entry["selected_doc_count"] = len(selected)
        docs_meta = []
        for doc_id, path in enumerate(selected, start=1):
            try:
                parsed = parse_doc(path, doc_id)
                docs_meta.append(
                    {
                        "doc_id": parsed.doc_id,
                        "number": parsed.number,
                        "path": parsed.path,
                        "title": parsed.title,
                        "publish_time": parsed.publish_time,
                    }
                )
            except Exception as exc:
                docs_meta.append(
                    {
                        "doc_id": doc_id,
                        "number": doc_number(path),
                        "path": str(path),
                        "title": path.stem,
                        "publish_time": "",
                        "parse_error": str(exc),
                    }
                )
        entry["docs"] = docs_meta
        manifest.append(entry)
    return manifest


def load_manifest_docs(entry: Dict[str, Any], doc_limit: int) -> List[Doc]:
    search_dir = Path(entry["search_results_dir"])
    doc_paths = sorted(search_dir.glob("资料*.txt"), key=doc_number)[:doc_limit]
    return [parse_doc(path, doc_id) for doc_id, path in enumerate(doc_paths, start=1)]


def split_sentences(text: str, min_chars: int) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"([。！？!?；;])", r"\1\n", text)
    candidates: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for part in re.split(r"\n+", line):
            sent = part.strip()
            sent = re.sub(r"\s+", " ", sent)
            if len(sent) >= min_chars:
                candidates.append(sent)
    return candidates


def trim_chars(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def format_doc_for_prompt(doc: Doc, max_chars: int) -> str:
    return (
        f"[资料{doc.doc_id}] 标题：{doc.title}\n"
        f"发布时间：{doc.publish_time}\n"
        f"正文：{trim_chars(doc.text, max_chars)}\n"
    )


def build_tfidf_ranker(docs: Sequence[Doc], max_chars: int):
    corpus = [trim_chars(f"{doc.title}\n{doc.text}", max_chars) for doc in docs]
    try:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=80000)
        doc_matrix = vectorizer.fit_transform(corpus)
        return vectorizer, doc_matrix
    except Exception:
        return None, None


def lexical_rank_docs(
    sentence: str,
    docs: Sequence[Doc],
    vectorizer: Any,
    doc_matrix: Any,
    top_k: int,
) -> List[Doc]:
    if not docs:
        return []
    if vectorizer is not None and doc_matrix is not None:
        try:
            query = vectorizer.transform([sentence])
            scores = cosine_similarity(query, doc_matrix)[0]
            order = np.argsort(scores)[::-1][:top_k]
            return [docs[int(i)] for i in order]
        except Exception:
            pass

    chars = set(sentence)
    scored = []
    for doc in docs:
        doc_chars = set(doc.title + doc.text[:4000])
        denom = max(1, len(chars))
        scored.append((len(chars & doc_chars) / denom, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def truncate_prefix_ids(
    prefix_ids: List[int],
    target_len: int,
    max_input_tokens: int,
) -> List[int]:
    keep = max_input_tokens - target_len - 1
    if keep <= 0:
        return prefix_ids[-1:]
    if len(prefix_ids) <= keep:
        return prefix_ids
    return prefix_ids[-keep:]


@torch.inference_mode()
def score_sentence(
    model: torch.nn.Module,
    tokenizer: Any,
    prefix: str,
    sentence: str,
    max_input_tokens: int,
) -> Tuple[float, List[float], List[str]]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    target_ids = tokenizer.encode(sentence, add_special_tokens=False)
    if not target_ids:
        return float("nan"), [], []
    prefix_ids = truncate_prefix_ids(prefix_ids, len(target_ids), max_input_tokens)
    input_ids_list = prefix_ids + target_ids
    labels = [-100] * len(prefix_ids) + target_ids
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=model_device(model))
    labels_t = torch.tensor([labels], dtype=torch.long, device=input_ids.device)
    outputs = model(input_ids=input_ids)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels_t[:, 1:]
    mask = shift_labels.ne(-100)
    if not bool(mask.any()):
        return float("nan"), [], []
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(
        -1,
        safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    selected = token_logprobs[mask].detach().float().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(target_ids)
    return float(np.mean(selected)), [float(x) for x in selected], tokens


def build_question_only_prefix(question: str) -> str:
    return f"问题：{question}\n答案："


def build_doc_prefix(question: str, doc: Doc, max_doc_chars: int) -> str:
    return f"问题：{question}\n\n参考资料：\n{format_doc_for_prompt(doc, max_doc_chars)}\n答案："


def build_compact_context_prefix(
    question: str,
    docs: Sequence[Doc],
    max_doc_chars: int,
) -> str:
    docs_text = "\n".join(format_doc_for_prompt(doc, max_doc_chars) for doc in docs)
    return f"问题：{question}\n\n参考资料：\n{docs_text}\n答案："


def stop_token_ids(tokenizer: Any, model: torch.nn.Module, model_path: str) -> List[int]:
    stop = ["\n", "Ċ", "ĊĊ", "<0x0A>"]
    ids = [tokenizer.convert_tokens_to_ids(token) for token in stop]
    ids.append(getattr(model.config, "eos_token_id", None))
    cleaned = [int(token_id) for token_id in ids if token_id is not None]
    model_lower = model_path.lower()
    if any(name in model_lower for name in ["llama", "zephyr", "mistral", "deepseek", "qwen"]):
        unk = getattr(tokenizer, "unk_token_id", None)
        cleaned = [token_id for token_id in cleaned if token_id != unk]
    return sorted(set(cleaned))


def inseq_decoder_separator(model_path: str) -> str:
    model_lower = model_path.lower()
    if "zephyr" in model_lower:
        return "\n "
    return " "


def cti_cache_path(output_dir: Path, alias: str, sentence_id: int) -> Path:
    return output_dir / "internal_cti" / alias / f"sentence-{sentence_id:04d}.json"


def load_cti_from_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "record" in payload:
            return payload["record"]
    except Exception:
        return None
    return None


def compute_cti_logprob_delta(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    sentence: str,
    docs: Sequence[Doc],
    max_doc_chars: int,
    max_input_tokens: int,
) -> Dict[str, Any]:
    base_prefix = build_question_only_prefix(question)
    context_prefix = build_compact_context_prefix(question, docs, max_doc_chars)
    base_avg, base_token_scores, base_tokens = score_sentence(
        model,
        tokenizer,
        base_prefix,
        sentence,
        max_input_tokens,
    )
    ctx_avg, ctx_token_scores, ctx_tokens = score_sentence(
        model,
        tokenizer,
        context_prefix,
        sentence,
        max_input_tokens,
    )
    length = min(len(base_token_scores), len(ctx_token_scores))
    token_cti = [abs(ctx_token_scores[i] - base_token_scores[i]) for i in range(length)]
    tokens = ctx_tokens[:length] if ctx_tokens else base_tokens[:length]
    sentence_cti = float(np.mean(token_cti)) if token_cti else 0.0
    return {
        "cti_method": "logprob_delta",
        "output_current_tokens": tokens,
        "cti_scores": [finite_float(x) for x in token_cti],
        "sentence_cti": finite_float(sentence_cti),
        "context_avg_logprob": finite_float(ctx_avg),
        "contextless_avg_logprob": finite_float(base_avg),
        "selected_context_docs": [doc.doc_id for doc in docs],
    }


def compute_cti_saliency(
    model: torch.nn.Module,
    tokenizer: Any,
    model_mirage: Any,
    model_path: str,
    question: str,
    sentence: str,
    docs: Sequence[Doc],
    max_doc_chars: int,
    max_new_tokens: int,
    cache_path: Path,
) -> Dict[str, Any]:
    if inseq is None or AttributeContextArgs is None or attribute_context_with_model is None:
        raise RuntimeError("inseq is not available")

    input_context_text = "\n".join(format_doc_for_prompt(doc, max_doc_chars) for doc in docs)
    input_template = "问题：{current}\n\n参考资料：\n{context}\n\n答案："
    contextless_input_current_text = input_template.replace("{context}\n\n", "")
    raw_save_path = str(cache_path.with_suffix(".inseq.json"))
    args = AttributeContextArgs(
        model_name_or_path=model_path,
        input_context_text=input_context_text,
        input_current_text=question,
        output_template="{current}",
        input_template=input_template,
        contextless_input_current_text=contextless_input_current_text,
        show_intermediate_outputs=False,
        attributed_fn="contrast_prob_diff",
        context_sensitivity_std_threshold=0,
        output_current_text=sentence,
        attribution_method="saliency",
        attribution_kwargs={"logprob": True},
        save_path=raw_save_path,
        tokenizer_kwargs={"use_fast": False},
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "max_memory": get_max_memory(),
            "load_in_8bit": False,
        },
        generation_kwargs={
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "eos_token_id": stop_token_ids(tokenizer, model, model_path),
        },
        decoder_input_output_separator=inseq_decoder_separator(model_path),
        special_tokens_to_keep=[],
        show_viz=False,
    )
    attribute_context_with_model(args, model_mirage)
    payload = json.loads(Path(raw_save_path).read_text(encoding="utf-8"))
    cti_scores = [float(x) for x in payload.get("cti_scores", [])]
    sentence_cti = float(np.mean(cti_scores)) if cti_scores else 0.0
    return {
        "cti_method": "inseq_saliency",
        "output_current_tokens": payload.get("output_current_tokens", []),
        "cti_scores": [finite_float(x) for x in cti_scores],
        "cci_scores": payload.get("cci_scores", []),
        "input_context_tokens": payload.get("input_context_tokens", []),
        "sentence_cti": finite_float(sentence_cti),
        "selected_context_docs": [doc.doc_id for doc in docs],
        "raw_inseq_path": raw_save_path,
    }


def compute_or_load_cti(
    args: argparse.Namespace,
    output_dir: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    model_mirage: Any,
    alias: str,
    question: str,
    sentence_id: int,
    sentence: str,
    selected_docs: Sequence[Doc],
) -> Dict[str, Any]:
    cache_path = cti_cache_path(output_dir, alias, sentence_id)
    cached = load_cti_from_cache(cache_path)
    if cached:
        return cached

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.cti_method == "saliency":
            cti = compute_cti_saliency(
                model,
                tokenizer,
                model_mirage,
                args.model,
                question,
                sentence,
                selected_docs,
                args.cti_doc_chars,
                args.max_new_tokens,
                cache_path,
            )
        else:
            cti = compute_cti_logprob_delta(
                model,
                tokenizer,
                question,
                sentence,
                selected_docs,
                args.cti_doc_chars,
                args.max_input_tokens,
            )
    except Exception as exc:
        if not args.allow_cti_fallback:
            raise
        cti = compute_cti_logprob_delta(
            model,
            tokenizer,
            question,
            sentence,
            selected_docs,
            args.cti_doc_chars,
            args.max_input_tokens,
        )
        cti["cti_method"] = "logprob_delta_fallback"
        cti["cti_error"] = f"{type(exc).__name__}: {exc}"
        cti["cti_traceback"] = traceback.format_exc(limit=3)

    record = {
        "sentence_id": sentence_id,
        "sentence": sentence,
        **cti,
    }
    write_json(cache_path, {"record": record})
    return record


def rank_sentence_records(records: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda r: (r.get("sentence_cti") is not None, r.get("sentence_cti") or -1.0),
        reverse=True,
    )
    top_ids = {r["sentence_id"] for r in ordered[:top_k]}
    ranked = []
    rank_lookup = {r["sentence_id"]: idx + 1 for idx, r in enumerate(ordered)}
    for record in records:
        copy = dict(record)
        copy["sensitivity_rank"] = rank_lookup[record["sentence_id"]]
        copy["is_top100"] = record["sentence_id"] in top_ids
        ranked.append(copy)
    ranked.sort(key=lambda r: r["sentence_id"])
    return ranked


def compute_sentence_cti_records(
    args: argparse.Namespace,
    output_dir: Path,
    entry: Dict[str, Any],
    docs: Sequence[Doc],
    sentences: Sequence[str],
    model: torch.nn.Module,
    tokenizer: Any,
    model_mirage: Any,
) -> List[Dict[str, Any]]:
    alias = entry["alias"]
    existing = [
        r for r in read_jsonl(output_dir / "sentence_cti.jsonl") if r.get("workdir_alias") == alias
    ]
    if len(existing) == len(sentences) and all("is_top100" in r for r in existing):
        return sorted(existing, key=lambda r: r["sentence_id"])

    vectorizer, doc_matrix = build_tfidf_ranker(docs, args.tfidf_doc_chars)
    records: List[Dict[str, Any]] = []
    for sentence_id, sentence in enumerate(tqdm(sentences, desc=f"{alias} sentence CTI")):
        update_heartbeat(
            output_dir,
            phase="sentence_cti",
            workdir_alias=alias,
            sentence_id=sentence_id,
            sentence_count=len(sentences),
        )
        selected_docs = lexical_rank_docs(
            sentence,
            docs,
            vectorizer,
            doc_matrix,
            args.cti_context_docs,
        )
        cti = compute_or_load_cti(
            args,
            output_dir,
            model,
            tokenizer,
            model_mirage,
            alias,
            entry["question"],
            sentence_id,
            sentence,
            selected_docs,
        )
        record = {
            "workdir_alias": alias,
            "workdir_name": entry["original_name"],
            "question": entry["question"],
            "sentence_id": sentence_id,
            "sentence": sentence,
            "selected_context_docs": cti.get("selected_context_docs", []),
            "cti_method": cti.get("cti_method"),
            "output_current_tokens": cti.get("output_current_tokens", []),
            "cti_scores": cti.get("cti_scores", []),
            "sentence_cti": cti.get("sentence_cti"),
            "context_avg_logprob": cti.get("context_avg_logprob"),
            "contextless_avg_logprob": cti.get("contextless_avg_logprob"),
        }
        for optional_key in ["cci_scores", "input_context_tokens", "raw_inseq_path", "cti_error"]:
            if optional_key in cti:
                record[optional_key] = cti[optional_key]
        records.append(record)

    ranked = rank_sentence_records(records, min(args.top_sensitive_sentences, len(records)))
    rewrite_jsonl_replace_alias(output_dir / "sentence_cti.jsonl", alias, ranked)
    return ranked


def done_doc_keys(path: Path) -> set:
    return {
        (r.get("workdir_alias"), int(r.get("sentence_id", -1)))
        for r in read_jsonl(path)
        if "sentence_id" in r
    }


def done_paragraph_keys(path: Path) -> set:
    return {
        (r.get("workdir_alias"), int(r.get("sentence_id", -1)), int(r.get("doc_id", -1)))
        for r in read_jsonl(path)
        if "sentence_id" in r and "doc_id" in r
    }


def compute_doc_perturbation(
    args: argparse.Namespace,
    output_dir: Path,
    entry: Dict[str, Any],
    docs: Sequence[Doc],
    sentence_records: Sequence[Dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
) -> None:
    alias = entry["alias"]
    out_path = output_dir / "doc_perturbation.jsonl"
    completed = done_doc_keys(out_path)
    targets = sorted(
        [r for r in sentence_records if r.get("is_top100")],
        key=lambda r: r.get("sensitivity_rank", 10**9),
    )
    for record in tqdm(targets, desc=f"{alias} doc perturbation"):
        sentence_id = int(record["sentence_id"])
        if (alias, sentence_id) in completed:
            continue
        sentence = record["sentence"]
        update_heartbeat(
            output_dir,
            phase="doc_perturbation",
            workdir_alias=alias,
            sentence_id=sentence_id,
            sentence_count=len(targets),
        )
        base_avg, _, _ = score_sentence(
            model,
            tokenizer,
            build_question_only_prefix(entry["question"]),
            sentence,
            args.max_input_tokens,
        )
        doc_scores = []
        for doc in docs:
            avg_logprob, _, _ = score_sentence(
                model,
                tokenizer,
                build_doc_prefix(entry["question"], doc, args.perturb_doc_chars),
                sentence,
                args.max_input_tokens,
            )
            delta = avg_logprob - base_avg
            doc_scores.append(
                {
                    "doc_id": doc.doc_id,
                    "doc_number": doc.number,
                    "title": doc.title,
                    "path": doc.path,
                    "avg_logprob": finite_float(avg_logprob),
                    "delta_vs_question_only": finite_float(delta),
                }
            )
        doc_scores.sort(
            key=lambda r: r["delta_vs_question_only"]
            if r["delta_vs_question_only"] is not None
            else -10**9,
            reverse=True,
        )
        for rank, doc_score in enumerate(doc_scores, start=1):
            doc_score["rank"] = rank
        append_jsonl(
            out_path,
            {
                "workdir_alias": alias,
                "workdir_name": entry["original_name"],
                "question": entry["question"],
                "sentence_id": sentence_id,
                "sensitivity_rank": record.get("sensitivity_rank"),
                "sentence_cti": record.get("sentence_cti"),
                "sentence": sentence,
                "base_avg_logprob": finite_float(base_avg),
                "doc_scores": doc_scores,
                "top_docs": doc_scores[: args.paragraph_doc_topk],
            },
        )
        completed.add((alias, sentence_id))


def split_paragraphs(text: str, min_chars: int) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    chunks = [p.strip() for p in re.split(r"\n\s*\n+", text) if len(p.strip()) >= min_chars]
    if chunks:
        return chunks
    line_chunks = [p.strip() for p in text.splitlines() if len(p.strip()) >= min_chars]
    if line_chunks:
        return line_chunks
    sentences = split_sentences(text, max(8, min_chars // 3))
    grouped = []
    for idx in range(0, len(sentences), 3):
        chunk = "".join(sentences[idx : idx + 3]).strip()
        if len(chunk) >= min_chars:
            grouped.append(chunk)
    return grouped if grouped else ([text] if text else [])


def remove_one_paragraph(paragraphs: Sequence[str], remove_idx: int) -> str:
    return "\n\n".join(p for idx, p in enumerate(paragraphs) if idx != remove_idx)


def compute_paragraph_perturbation(
    args: argparse.Namespace,
    output_dir: Path,
    entry: Dict[str, Any],
    doc_by_id: Dict[int, Doc],
    model: torch.nn.Module,
    tokenizer: Any,
) -> None:
    alias = entry["alias"]
    doc_records = [
        r for r in read_jsonl(output_dir / "doc_perturbation.jsonl") if r.get("workdir_alias") == alias
    ]
    out_path = output_dir / "paragraph_perturbation.jsonl"
    completed = done_paragraph_keys(out_path)
    for record in tqdm(doc_records, desc=f"{alias} paragraph perturbation"):
        sentence_id = int(record["sentence_id"])
        sentence = record["sentence"]
        top_docs = record.get("top_docs", [])[: args.paragraph_doc_topk]
        for doc_rank, doc_meta in enumerate(top_docs, start=1):
            doc_id = int(doc_meta["doc_id"])
            if (alias, sentence_id, doc_id) in completed:
                continue
            doc = doc_by_id.get(doc_id)
            if not doc:
                continue
            update_heartbeat(
                output_dir,
                phase="paragraph_perturbation",
                workdir_alias=alias,
                sentence_id=sentence_id,
                doc_id=doc_id,
            )
            paragraphs = split_paragraphs(doc.text, args.min_paragraph_chars)
            full_avg = doc_meta.get("avg_logprob")
            if full_avg is None:
                full_avg, _, _ = score_sentence(
                    model,
                    tokenizer,
                    build_doc_prefix(entry["question"], doc, args.perturb_doc_chars),
                    sentence,
                    args.max_input_tokens,
                )
            paragraph_scores = []
            for para_id, paragraph in enumerate(paragraphs, start=1):
                ablated_text = remove_one_paragraph(paragraphs, para_id - 1)
                ablated_doc = Doc(
                    doc_id=doc.doc_id,
                    number=doc.number,
                    path=doc.path,
                    title=doc.title,
                    publish_time=doc.publish_time,
                    text=ablated_text,
                )
                without_avg, _, _ = score_sentence(
                    model,
                    tokenizer,
                    build_doc_prefix(entry["question"], ablated_doc, args.perturb_doc_chars),
                    sentence,
                    args.max_input_tokens,
                )
                importance = float(full_avg) - without_avg
                paragraph_scores.append(
                    {
                        "paragraph_id": para_id,
                        "chars": len(paragraph),
                        "excerpt": trim_chars(paragraph, args.summary_excerpt_chars),
                        "without_paragraph_avg_logprob": finite_float(without_avg),
                        "importance_full_minus_without": finite_float(importance),
                    }
                )
            paragraph_scores.sort(
                key=lambda r: r["importance_full_minus_without"]
                if r["importance_full_minus_without"] is not None
                else -10**9,
                reverse=True,
            )
            for rank, para_score in enumerate(paragraph_scores, start=1):
                para_score["rank"] = rank
            append_jsonl(
                out_path,
                {
                    "workdir_alias": alias,
                    "workdir_name": entry["original_name"],
                    "question": entry["question"],
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "doc_rank": doc_rank,
                    "doc_id": doc.doc_id,
                    "doc_number": doc.number,
                    "doc_title": doc.title,
                    "doc_path": doc.path,
                    "full_doc_avg_logprob": finite_float(float(full_avg)),
                    "paragraph_scores": paragraph_scores,
                    "top_paragraphs": paragraph_scores[:3],
                },
            )
            completed.add((alias, sentence_id, doc_id))


def build_summary(output_dir: Path, manifest: Sequence[Dict[str, Any]]) -> None:
    sentence_records = read_jsonl(output_dir / "sentence_cti.jsonl")
    doc_records = read_jsonl(output_dir / "doc_perturbation.jsonl")
    para_records = read_jsonl(output_dir / "paragraph_perturbation.jsonl")
    docs_by_sentence = {
        (r.get("workdir_alias"), r.get("sentence_id")): r for r in doc_records
    }
    paras_by_sentence_doc = {
        (r.get("workdir_alias"), r.get("sentence_id"), r.get("doc_id")): r
        for r in para_records
    }
    selected_workdirs = sorted({r.get("workdir_alias") for r in sentence_records})
    lines = [
        "# data_1 DeepSeek MIRAGE 两级归因摘要",
        "",
        f"- 更新时间：{now_ts()}",
        f"- manifest workdir 数：{len(manifest)}",
        f"- 已处理 workdir：{', '.join(selected_workdirs) if selected_workdirs else '无'}",
        f"- sentence_cti 行数：{len(sentence_records)}",
        f"- doc_perturbation 行数：{len(doc_records)}",
        f"- paragraph_perturbation 行数：{len(para_records)}",
        "",
        "## Top 敏感句与引用依据",
        "",
    ]
    top_sentences = sorted(
        sentence_records,
        key=lambda r: r.get("sensitivity_rank", 10**9),
    )[:20]
    for record in top_sentences:
        alias = record.get("workdir_alias")
        sentence_id = record.get("sentence_id")
        lines.append(
            f"### {alias} / sentence-{int(sentence_id):04d} / rank {record.get('sensitivity_rank')}"
        )
        lines.append("")
        lines.append(f"- CTI：{record.get('sentence_cti')}")
        lines.append(f"- 句子：{record.get('sentence')}")
        doc_record = docs_by_sentence.get((alias, sentence_id))
        if doc_record:
            lines.append("- Top 引用资料：")
            for doc in doc_record.get("top_docs", [])[:3]:
                lines.append(
                    f"  - rank {doc.get('rank')} / 资料{doc.get('doc_number')} / "
                    f"delta {doc.get('delta_vs_question_only')}：{doc.get('title')}"
                )
                para_record = paras_by_sentence_doc.get((alias, sentence_id, doc.get("doc_id")))
                if para_record:
                    for para in para_record.get("top_paragraphs", [])[:1]:
                        lines.append(
                            f"    - 关键段落 rank {para.get('rank')} / importance "
                            f"{para.get('importance_full_minus_without')}：{para.get('excerpt')}"
                        )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_models(args: argparse.Namespace):
    model, tokenizer = load_model(args.model)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model_mirage = None
    if args.cti_method == "saliency" and inseq is not None:
        model_mirage = inseq.load_model(
            model,
            "saliency",
            model_kwargs={"device_map": "cuda:0", "torch_dtype": torch.float16},
            tokenizer_kwargs={"use_fast": False},
        )
    return model, tokenizer, model_mirage


def selected_manifest_entries(
    manifest: Sequence[Dict[str, Any]],
    workdir_limit: int,
    aliases: Sequence[str],
) -> List[Dict[str, Any]]:
    runnable = [entry for entry in manifest if not entry.get("skipped")]
    if aliases:
        alias_set = set(aliases)
        runnable = [entry for entry in runnable if entry["alias"] in alias_set]
    if workdir_limit > 0:
        runnable = runnable[:workdir_limit]
    return runnable


def validate_outputs(output_dir: Path) -> None:
    for path in [
        output_dir / "sentence_cti.jsonl",
        output_dir / "doc_perturbation.jsonl",
        output_dir / "paragraph_perturbation.jsonl",
    ]:
        for idx, record in enumerate(read_jsonl(path), start=1):
            text = json.dumps(record, ensure_ascii=False)
            if "NaN" in text or "Infinity" in text:
                raise RuntimeError(f"non-finite JSON value in {path}:{idx}")
            if '"path": ""' in text or '"doc_path": ""' in text:
                raise RuntimeError(f"empty path in {path}:{idx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--workdir-limit", type=int, default=1)
    parser.add_argument("--workdir-alias", action="append", default=[])
    parser.add_argument("--doc-limit", type=int, default=100)
    parser.add_argument("--cti-context-docs", type=int, default=5)
    parser.add_argument("--top-sensitive-sentences", type=int, default=100)
    parser.add_argument("--paragraph-doc-topk", type=int, default=3)
    parser.add_argument("--cti-method", choices=["saliency", "logprob_delta"], default="saliency")
    parser.add_argument("--allow-cti-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--tfidf-doc-chars", type=int, default=5000)
    parser.add_argument("--cti-doc-chars", type=int, default=0)
    parser.add_argument("--perturb-doc-chars", type=int, default=3500)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-sentence-chars", type=int, default=8)
    parser.add_argument("--min-paragraph-chars", type=int, default=30)
    parser.add_argument("--summary-excerpt-chars", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.data_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    update_heartbeat(output_dir, phase="manifest", session=SESSION_NAME)
    manifest = build_manifest(data_root, args.doc_limit)
    write_json(output_dir / "manifest.json", manifest)

    runnable_entries = selected_manifest_entries(manifest, args.workdir_limit, args.workdir_alias)
    if args.manifest_only:
        print(f"[data1] wrote manifest for {len(manifest)} workdirs; runnable={len(runnable_entries)}")
        return
    if not runnable_entries:
        raise RuntimeError("no runnable data_1 workdir found")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepSeek attribution")

    update_heartbeat(output_dir, phase="load_model", model=args.model)
    model, tokenizer, model_mirage = load_models(args)

    for entry in runnable_entries:
        alias = entry["alias"]
        update_heartbeat(output_dir, phase="load_workdir", workdir_alias=alias)
        docs = load_manifest_docs(entry, args.doc_limit)
        history_text = read_text(Path(entry["history_path"]))
        sentences = split_sentences(history_text, args.min_sentence_chars)
        if not sentences:
            raise RuntimeError(f"{alias} has no valid sentences")
        sentence_records = compute_sentence_cti_records(
            args,
            output_dir,
            entry,
            docs,
            sentences,
            model,
            tokenizer,
            model_mirage,
        )
        compute_doc_perturbation(args, output_dir, entry, docs, sentence_records, model, tokenizer)
        compute_paragraph_perturbation(
            args,
            output_dir,
            entry,
            {doc.doc_id: doc for doc in docs},
            model,
            tokenizer,
        )

    update_heartbeat(output_dir, phase="summary")
    build_summary(output_dir, manifest)
    validate_outputs(output_dir)
    update_heartbeat(output_dir, phase="done")
    print(f"[data1] complete; outputs written to {output_dir}")


if __name__ == "__main__":
    main()
