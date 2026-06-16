#!/usr/bin/env python
"""Generate corrected data_1 answers and run strict MIRAGE-style attribution.

V2 keeps the raw data_1 directories read-only. It writes generated answers and
all attribution artifacts to a separate run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import data1_deepseek_attribution as v1


MODEL_PATH = "/home/intern/models/DeepSeek-R1-Distill-Qwen-14B"
DEFAULT_DATA_ROOT = "/mnt/data2/zyc/mirage/data_1"
DEFAULT_OUTPUT_DIR = "runs/data1-deepseek-mirage-v2"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
SESSION_NAME = "mirage-data1-deepseek-v2"
OLD_HISTORY_TITLE = "生成式人工智能在美军指挥控制领域的发展现状"


@dataclass
class EvidenceChunk:
    chunk_id: int
    doc_id: int
    doc_number: int
    doc_title: str
    doc_path: str
    section_title: str
    start_sentence: int
    end_sentence: int
    text: str

    def prompt_title(self) -> str:
        title = f"资料{self.doc_number}｜{self.doc_title}"
        if self.section_title:
            title += f"｜{self.section_title}"
        return title


@dataclass
class SentenceItem:
    sentence_id: int
    raw_sentence_id: int
    sentence: str
    answer_prefix_before: str
    answer_prefix_chars: int
    answer_prefix_excerpt: str


class BgeEmbedder:
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().float().cpu().numpy())
        if not vectors:
            return np.zeros((0, 1), dtype=np.float32)
        return np.vstack(vectors)


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return re.sub(r"^#+\s*", "", line)
    return ""


def update_heartbeat(output_dir: Path, **fields: Any) -> None:
    v1.write_json(output_dir / ".heartbeat", {"updated_at": v1.now_ts(), **fields})


def normalize_for_validation(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def chunk_doc_sentences(
    doc: v1.Doc,
    chunk_size: int,
    min_chars: int,
) -> List[EvidenceChunk]:
    text = doc.text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_title = ""
    current_lines: List[str] = []
    heading_re = re.compile(r"^\s*(#{1,6}\s+.+|\*\*[^*]{2,}\*\*)\s*$")
    for line in lines:
        stripped = line.strip()
        if heading_re.match(stripped):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = re.sub(r"^\s*#{1,6}\s*", "", stripped).strip("* ")
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    if not sections:
        sections = [("", [text])]

    chunks: List[EvidenceChunk] = []
    for section_title, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        sentences = v1.split_sentences(section_text, min_chars)
        if not sentences and section_text:
            sentences = [section_text]
        for offset in range(0, len(sentences), chunk_size):
            grouped = sentences[offset : offset + chunk_size]
            chunk_text = " ".join(grouped).strip()
            if not chunk_text:
                continue
            chunks.append(
                EvidenceChunk(
                    chunk_id=0,
                    doc_id=doc.doc_id,
                    doc_number=doc.number,
                    doc_title=doc.title,
                    doc_path=doc.path,
                    section_title=section_title,
                    start_sentence=offset + 1,
                    end_sentence=offset + len(grouped),
                    text=chunk_text,
                )
            )
    return chunks


def build_evidence_chunks(
    docs: Sequence[v1.Doc],
    chunk_size: int,
    min_chars: int,
) -> List[EvidenceChunk]:
    chunks: List[EvidenceChunk] = []
    for doc in docs:
        chunks.extend(chunk_doc_sentences(doc, chunk_size, min_chars))
    for idx, chunk in enumerate(chunks, start=1):
        chunk.chunk_id = idx
    return chunks


def chunk_to_prompt_doc(chunk: EvidenceChunk, prompt_id: Optional[int] = None) -> v1.Doc:
    return v1.Doc(
        doc_id=prompt_id or chunk.chunk_id,
        number=chunk.doc_number,
        path=chunk.doc_path,
        title=chunk.prompt_title(),
        publish_time="",
        text=chunk.text,
    )


def rank_chunks(
    embedder: BgeEmbedder,
    chunk_embeddings: np.ndarray,
    chunks: Sequence[EvidenceChunk],
    query: str,
    top_k: int,
) -> List[Tuple[EvidenceChunk, float]]:
    if len(chunks) == 0:
        return []
    query_vec = embedder.encode([query])[0]
    scores = chunk_embeddings @ query_vec
    order = np.argsort(scores)[::-1][:top_k]
    return [(chunks[int(i)], float(scores[int(i)])) for i in order]


def is_answer_content_sentence(sentence: str) -> bool:
    text = sentence.strip()
    if not text:
        return False
    if text.startswith("#"):
        return False
    if re.match(r"^\d+[.、]\s*\*\*[^*]{1,80}\*\*\s*[:：]?$", text):
        return False
    if re.match(r"^\d+[.、]\s*[^。！？.!?]{1,40}\s*[:：]$", text):
        return False
    if text.endswith((":", "：")) and len(normalize_for_validation(text)) < 80:
        return False
    return True


def suffix_excerpt(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def split_answer_content_items(
    text: str,
    min_chars: int,
    excerpt_chars: int,
) -> Tuple[List[SentenceItem], List[Dict[str, Any]]]:
    kept: List[SentenceItem] = []
    skipped: List[Dict[str, Any]] = []
    cursor = 0
    for raw_id, sentence in enumerate(v1.split_sentences(text, min_chars)):
        start = text.find(sentence, cursor)
        if start < 0:
            start = text.find(sentence)
        if start < 0:
            start = cursor
        answer_prefix = text[:start]
        if is_answer_content_sentence(sentence):
            kept.append(
                SentenceItem(
                    sentence_id=len(kept),
                    raw_sentence_id=raw_id,
                    sentence=sentence,
                    answer_prefix_before=answer_prefix,
                    answer_prefix_chars=len(answer_prefix),
                    answer_prefix_excerpt=suffix_excerpt(answer_prefix, excerpt_chars),
                )
            )
        else:
            skipped.append({"raw_sentence_id": raw_id, "sentence": sentence})
        cursor = max(cursor, start + len(sentence))
    return kept, skipped


def format_chunk_for_generation(chunk: EvidenceChunk, max_chars: int) -> str:
    return (
        f"[资料{chunk.doc_number}/chunk{chunk.chunk_id}] {chunk.doc_title}\n"
        f"小节：{chunk.section_title or '正文'}\n"
        f"内容：{v1.trim_chars(chunk.text, max_chars)}"
    )


def build_generation_prompt(
    question: str,
    ranked_chunks: Sequence[Tuple[EvidenceChunk, float]],
    max_chars: int,
) -> str:
    evidence = "\n\n".join(format_chunk_for_generation(chunk, max_chars) for chunk, _ in ranked_chunks)
    return f"""你是一名军事与安全研究员。请只依据下面给定资料，围绕主题撰写一份中文历史成果报告。

主题：{question}

写作要求：
1. 标题必须是“## {question}”。
2. 内容必须围绕星链、俄乌冲突、军事作用、通信/侦察/无人机/指挥控制、暴露问题与启示。
3. 不要写“生成式人工智能在美军指挥控制领域”的内容。
4. 不要添加引用编号，不要输出思考过程。
5. 结构包含：发展背景、军事作用、暴露问题、影响启示、观点建议。
6. 语言精炼，约 900 到 1400 个汉字。

资料：
{evidence}

请直接输出报告正文："""


def strip_model_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = text.replace("<｜end▁of▁sentence｜>", "")
    return text.strip()


def ensure_history_title(text: str, question: str) -> str:
    text = text.strip()
    if not text.startswith("##"):
        return f"## {question}\n\n{text}".strip()
    if question not in first_nonempty_line(text):
        return f"## {question}\n\n" + "\n".join(text.splitlines()[1:]).strip()
    return text


def looks_like_final_history(text: str) -> bool:
    normalized = normalize_for_validation(text)
    if len(normalized) < 500:
        return False
    head = text[:500]
    thinking_markers = [
        "我现在需要",
        "用户",
        "资料中提到",
        "接下来",
        "在写作过程中",
        "首先，我",
        "现在，我需要",
    ]
    return not any(marker in head for marker in thinking_markers)


def generation_eos_token_ids(tokenizer: Any, model: torch.nn.Module) -> List[int]:
    ids = []
    eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, list):
        ids.extend(eos)
    elif eos is not None:
        ids.append(eos)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        ids.append(tokenizer_eos)
    return sorted({int(token_id) for token_id in ids if token_id is not None})


@torch.inference_mode()
def generate_history(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    ranked_chunks: Sequence[Tuple[EvidenceChunk, float]],
    args: argparse.Namespace,
) -> str:
    prompt = build_generation_prompt(question, ranked_chunks, args.generation_chunk_chars)
    if args.no_think_prefill:
        attempts = [True, False]
    else:
        attempts = [False, True]

    best_text = ""
    for force_final in attempts:
        if getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = prompt
        if force_final:
            prompt_text += "<think>\n</think>\n\n"
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=args.max_input_tokens)
        inputs = {k: v.to(v1.model_device(model)) for k, v in inputs.items()}
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.generated_max_new_tokens,
            eos_token_id=generation_eos_token_ids(tokenizer, model),
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        )
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        text = ensure_history_title(strip_model_output(text), question)
        best_text = text if len(text) > len(best_text) else best_text
        if looks_like_final_history(text):
            return text.strip() + "\n"
    raise RuntimeError(
        "DeepSeek generated reasoning or too-short history instead of final report. "
        f"Best candidate starts with: {best_text[:160]!r}"
    )


def validate_generated_history(
    entry: Dict[str, Any],
    generated_text: str,
    embedder: BgeEmbedder,
    threshold: float,
) -> Dict[str, Any]:
    title = first_nonempty_line(generated_text)
    similarity = float((embedder.encode([entry["question"]]) @ embedder.encode([title])[0])[0])
    normalized = normalize_for_validation(generated_text)
    errors = []
    if OLD_HISTORY_TITLE in generated_text:
        errors.append("generated history still contains old repeated report title")
    if "星链" not in generated_text and "Starlink" not in generated_text:
        errors.append("generated history does not mention 星链/Starlink")
    if similarity < threshold:
        errors.append(f"title-topic similarity {similarity:.4f} below threshold {threshold}")
    if len(normalized) < 500:
        errors.append("generated history is too short")
    return {
        "generated_title": title,
        "generated_history_md5": md5_text(generated_text),
        "title_topic_similarity": similarity,
        "validation_errors": errors,
    }


def annotate_manifest_histories(
    manifest: List[Dict[str, Any]],
    generated_entry: Optional[Dict[str, Any]],
) -> None:
    md5_counts: Dict[str, int] = {}
    for entry in manifest:
        path = entry.get("history_path")
        if path and Path(path).exists():
            text = v1.read_text(Path(path))
            digest = md5_text(text)
            entry["original_history_md5"] = digest
            entry["original_history_title"] = first_nonempty_line(text)
            md5_counts[digest] = md5_counts.get(digest, 0) + 1
        else:
            entry["original_history_md5"] = ""
            entry["original_history_title"] = ""
    for entry in manifest:
        digest = entry.get("original_history_md5", "")
        entry["original_history_duplicate_count"] = md5_counts.get(digest, 0) if digest else 0
    if generated_entry:
        for entry in manifest:
            if entry["alias"] == generated_entry["alias"]:
                entry.update(generated_entry)


def write_generation_debug(
    output_dir: Path,
    alias: str,
    question: str,
    ranked_chunks: Sequence[Tuple[EvidenceChunk, float]],
) -> None:
    payload = {
        "alias": alias,
        "question": question,
        "chunks": [
            {
                "rank": idx,
                "score": score,
                "chunk_id": chunk.chunk_id,
                "doc_number": chunk.doc_number,
                "doc_title": chunk.doc_title,
                "section_title": chunk.section_title,
                "excerpt": v1.trim_chars(chunk.text, 360),
            }
            for idx, (chunk, score) in enumerate(ranked_chunks, start=1)
        ],
    }
    v1.write_json(output_dir / "embedding_debug" / alias / "generation_chunks.json", payload)


def write_sentence_context_debug(
    output_dir: Path,
    alias: str,
    contexts: Dict[int, List[Tuple[EvidenceChunk, float]]],
) -> None:
    rows = []
    for sentence_id, ranked in contexts.items():
        rows.append(
            {
                "sentence_id": sentence_id,
                "chunks": [
                    {
                        "rank": idx,
                        "score": score,
                        "chunk_id": chunk.chunk_id,
                        "doc_number": chunk.doc_number,
                        "doc_title": chunk.doc_title,
                        "section_title": chunk.section_title,
                        "excerpt": v1.trim_chars(chunk.text, 220),
                    }
                    for idx, (chunk, score) in enumerate(ranked, start=1)
                ],
            }
        )
    v1.write_json(output_dir / "embedding_debug" / alias / "sentence_contexts.json", rows)


def escape_template_text(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def answer_prefix_md5(answer_prefix: str) -> str:
    return md5_text(answer_prefix)


def build_answer_only_prefix(question: str, answer_prefix: str) -> str:
    return f"问题：{question}\n答案：{answer_prefix}"


def build_doc_answer_prefix(question: str, doc: v1.Doc, max_doc_chars: int, answer_prefix: str) -> str:
    return f"问题：{question}\n\n参考资料：\n{v1.format_doc_for_prompt(doc, max_doc_chars)}\n答案：{answer_prefix}"


def build_compact_context_answer_prefix(
    question: str,
    docs: Sequence[v1.Doc],
    max_doc_chars: int,
    answer_prefix: str,
) -> str:
    docs_text = "\n".join(v1.format_doc_for_prompt(doc, max_doc_chars) for doc in docs)
    return f"问题：{question}\n\n参考资料：\n{docs_text}\n答案：{answer_prefix}"


def compute_cti_sentence_logprob(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    sentence: str,
    answer_prefix: str,
    prompt_docs: Sequence[v1.Doc],
    max_doc_chars: int,
    max_input_tokens: int,
) -> Dict[str, Any]:
    contextless_avg, _, _ = v1.score_sentence(
        model,
        tokenizer,
        build_answer_only_prefix(question, answer_prefix),
        sentence,
        max_input_tokens,
    )
    context_avg, _, _ = v1.score_sentence(
        model,
        tokenizer,
        build_compact_context_answer_prefix(question, prompt_docs, max_doc_chars, answer_prefix),
        sentence,
        max_input_tokens,
    )
    sentence_cti = context_avg - contextless_avg
    return {
        "cti_method": "sentence_logprob_delta_with_answer_prefix",
        "cti_mode": "sentence_logprob",
        "output_current_tokens": [],
        "cti_scores": [],
        "cci_scores": [],
        "input_context_tokens": [],
        "sentence_cti": v1.finite_float(sentence_cti),
        "context_avg_logprob": v1.finite_float(context_avg),
        "contextless_avg_logprob": v1.finite_float(contextless_avg),
        "selected_context_docs": [doc.doc_id for doc in prompt_docs],
    }


def compute_cti_saliency_with_answer_prefix(
    model: torch.nn.Module,
    tokenizer: Any,
    model_mirage: Any,
    model_path: str,
    question: str,
    sentence: str,
    answer_prefix: str,
    prompt_docs: Sequence[v1.Doc],
    max_doc_chars: int,
    max_new_tokens: int,
    cache_path: Path,
) -> Dict[str, Any]:
    if getattr(v1, "inseq", None) is None or getattr(v1, "AttributeContextArgs", None) is None:
        raise RuntimeError("inseq is not available")

    escaped_prefix = escape_template_text(answer_prefix)
    input_context_text = "\n".join(v1.format_doc_for_prompt(doc, max_doc_chars) for doc in prompt_docs)
    input_template = "问题：{current}\n\n参考资料：\n{context}\n\n答案：" + escaped_prefix
    contextless_input_current_text = "问题：{current}\n答案：" + escaped_prefix
    raw_save_path = str(cache_path.with_suffix(".inseq.json"))
    args = v1.AttributeContextArgs(
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
            "max_memory": v1.get_max_memory(),
            "load_in_8bit": False,
        },
        generation_kwargs={
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "eos_token_id": v1.stop_token_ids(tokenizer, model, model_path),
        },
        decoder_input_output_separator=v1.inseq_decoder_separator(model_path),
        special_tokens_to_keep=[],
        show_viz=False,
    )
    v1.attribute_context_with_model(args, model_mirage)
    payload = json.loads(Path(raw_save_path).read_text(encoding="utf-8"))
    cti_scores = [float(x) for x in payload.get("cti_scores", [])]
    sentence_cti = float(np.mean(cti_scores)) if cti_scores else 0.0
    return {
        "cti_method": "inseq_saliency",
        "cti_mode": "token_saliency",
        "output_current_tokens": payload.get("output_current_tokens", []),
        "cti_scores": [v1.finite_float(x) for x in cti_scores],
        "cci_scores": payload.get("cci_scores", []),
        "input_context_tokens": payload.get("input_context_tokens", []),
        "sentence_cti": v1.finite_float(sentence_cti),
        "selected_context_docs": [doc.doc_id for doc in prompt_docs],
        "raw_inseq_path": raw_save_path,
    }


def compute_or_record_strict_cti(
    args: argparse.Namespace,
    output_dir: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    model_mirage: Any,
    entry: Dict[str, Any],
    item: SentenceItem,
    ranked_chunks: Sequence[Tuple[EvidenceChunk, float]],
) -> Optional[Dict[str, Any]]:
    alias = entry["alias"]
    sentence_id = item.sentence_id
    sentence = item.sentence
    cache_path = v1.cti_cache_path(output_dir, alias, sentence_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = v1.load_cti_from_cache(cache_path)
    if (
        cached
        and cached.get("cti_mode") == args.cti_mode
        and cached.get("answer_prefix_md5") == answer_prefix_md5(item.answer_prefix_before)
    ):
        return cached

    prompt_docs = [chunk_to_prompt_doc(chunk, idx) for idx, (chunk, _) in enumerate(ranked_chunks, start=1)]
    try:
        if args.cti_mode == "token_saliency":
            cti = compute_cti_saliency_with_answer_prefix(
                model,
                tokenizer,
                model_mirage,
                args.model,
                entry["question"],
                sentence,
                item.answer_prefix_before,
                prompt_docs,
                args.cti_doc_chars,
                args.max_new_tokens,
                cache_path,
            )
        else:
            cti = compute_cti_sentence_logprob(
                model,
                tokenizer,
                entry["question"],
                sentence,
                item.answer_prefix_before,
                prompt_docs,
                args.cti_doc_chars,
                args.max_input_tokens,
            )
    except Exception as exc:
        failure = {
            "workdir_alias": alias,
            "workdir_name": entry["original_name"],
            "question": entry["question"],
            "sentence_id": sentence_id,
            "raw_sentence_id": item.raw_sentence_id,
            "sentence": sentence,
            "cti_mode": args.cti_mode,
            "answer_prefix_chars": item.answer_prefix_chars,
            "answer_prefix_excerpt": item.answer_prefix_excerpt,
            "selected_context_chunks": [
                {
                    "rank": idx,
                    "score": score,
                    "chunk_id": chunk.chunk_id,
                    "doc_number": chunk.doc_number,
                    "doc_title": chunk.doc_title,
                    "section_title": chunk.section_title,
                    "excerpt": v1.trim_chars(chunk.text, 220),
                }
                for idx, (chunk, score) in enumerate(ranked_chunks, start=1)
            ],
            "cti_error": f"{type(exc).__name__}: {exc}",
            "cti_traceback": traceback.format_exc(limit=5),
        }
        v1.append_jsonl(output_dir / "cti_failed.jsonl", failure)
        return None

    cti["selected_context_chunks"] = [
        {
            "rank": idx,
            "score": score,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "doc_number": chunk.doc_number,
            "doc_title": chunk.doc_title,
            "section_title": chunk.section_title,
            "excerpt": v1.trim_chars(chunk.text, 220),
        }
        for idx, (chunk, score) in enumerate(ranked_chunks, start=1)
    ]
    record = {
        "sentence_id": sentence_id,
        "raw_sentence_id": item.raw_sentence_id,
        "sentence": sentence,
        "answer_prefix_before": item.answer_prefix_before,
        "answer_prefix_chars": item.answer_prefix_chars,
        "answer_prefix_excerpt": item.answer_prefix_excerpt,
        "answer_prefix_md5": answer_prefix_md5(item.answer_prefix_before),
        **cti,
    }
    v1.write_json(cache_path, {"record": record})
    return record


def compute_sentence_cti_records_strict(
    args: argparse.Namespace,
    output_dir: Path,
    entry: Dict[str, Any],
    sentence_items: Sequence[SentenceItem],
    sentence_contexts: Dict[int, List[Tuple[EvidenceChunk, float]]],
    model: torch.nn.Module,
    tokenizer: Any,
    model_mirage: Any,
) -> List[Dict[str, Any]]:
    alias = entry["alias"]
    success_records: List[Dict[str, Any]] = []
    failed_path = output_dir / "cti_failed.jsonl"
    if failed_path.exists() and not args.retry_cti_failed:
        failed_ids = {
            int(r.get("sentence_id", -1))
            for r in v1.read_jsonl(failed_path)
            if r.get("workdir_alias") == alias
            and r.get("cti_mode") == args.cti_mode
        }
    else:
        failed_ids = set()
        if failed_path.exists() and args.retry_cti_failed:
            kept = [r for r in v1.read_jsonl(failed_path) if r.get("workdir_alias") != alias]
            with failed_path.open("w", encoding="utf-8") as f:
                for record in kept:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for item in tqdm(sentence_items, desc=f"{alias} {args.cti_mode} CTI"):
        sentence_id = item.sentence_id
        sentence = item.sentence
        if sentence_id in failed_ids:
            continue
        update_heartbeat(
            output_dir,
            phase="sentence_cti",
            workdir_alias=alias,
            sentence_id=sentence_id,
            sentence_count=len(sentence_items),
            cti_mode=args.cti_mode,
        )
        ranked_chunks = sentence_contexts[sentence_id]
        cti = compute_or_record_strict_cti(
            args,
            output_dir,
            model,
            tokenizer,
            model_mirage,
            entry,
            item,
            ranked_chunks,
        )
        if cti is None:
            continue
        record = {
            "workdir_alias": alias,
            "workdir_name": entry["original_name"],
            "question": entry["question"],
            "sentence_id": sentence_id,
            "raw_sentence_id": item.raw_sentence_id,
            "sentence": sentence,
            "answer_prefix_before": item.answer_prefix_before,
            "answer_prefix_chars": item.answer_prefix_chars,
            "answer_prefix_excerpt": item.answer_prefix_excerpt,
            "answer_prefix_md5": answer_prefix_md5(item.answer_prefix_before),
            "selected_context_docs": cti.get("selected_context_docs", []),
            "selected_context_chunks": cti.get("selected_context_chunks", []),
            "cti_method": cti.get("cti_method"),
            "cti_mode": cti.get("cti_mode", args.cti_mode),
            "output_current_tokens": cti.get("output_current_tokens", []),
            "cti_scores": cti.get("cti_scores", []),
            "sentence_cti": cti.get("sentence_cti"),
            "context_avg_logprob": cti.get("context_avg_logprob"),
            "contextless_avg_logprob": cti.get("contextless_avg_logprob"),
            "cci_scores": cti.get("cci_scores", []),
            "input_context_tokens": cti.get("input_context_tokens", []),
        }
        if cti.get("raw_inseq_path"):
            record["raw_inseq_path"] = cti["raw_inseq_path"]
        success_records.append(record)

    if not success_records:
        raise RuntimeError(f"{alias} has no successful {args.cti_mode} CTI records")
    ranked = v1.rank_sentence_records(success_records, min(args.top_sensitive_sentences, len(success_records)))
    v1.rewrite_jsonl_replace_alias(output_dir / "sentence_cti.jsonl", alias, ranked)
    return ranked


def split_doc_chunks_for_perturbation(
    doc: v1.Doc,
    chunk_size: int,
    min_chars: int,
) -> List[EvidenceChunk]:
    chunks = chunk_doc_sentences(doc, chunk_size, min_chars)
    for idx, chunk in enumerate(chunks, start=1):
        chunk.chunk_id = idx
    return chunks


def done_doc_keys(path: Path) -> set:
    return {
        (r.get("workdir_alias"), int(r.get("sentence_id", -1)))
        for r in v1.read_jsonl(path)
        if "sentence_id" in r
    }


def compute_doc_perturbation_with_answer_prefix(
    args: argparse.Namespace,
    output_dir: Path,
    entry: Dict[str, Any],
    docs: Sequence[v1.Doc],
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
        answer_prefix = record.get("answer_prefix_before", "")
        update_heartbeat(
            output_dir,
            phase="doc_perturbation",
            workdir_alias=alias,
            sentence_id=sentence_id,
            sentence_count=len(targets),
        )
        base_avg, _, _ = v1.score_sentence(
            model,
            tokenizer,
            build_answer_only_prefix(entry["question"], answer_prefix),
            sentence,
            args.max_input_tokens,
        )
        doc_scores = []
        for doc in docs:
            avg_logprob, _, _ = v1.score_sentence(
                model,
                tokenizer,
                build_doc_answer_prefix(entry["question"], doc, args.perturb_doc_chars, answer_prefix),
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
                    "avg_logprob": v1.finite_float(avg_logprob),
                    "delta_vs_contextless_with_answer_prefix": v1.finite_float(delta),
                    "delta_vs_question_only": v1.finite_float(delta),
                }
            )
        doc_scores.sort(
            key=lambda r: r["delta_vs_contextless_with_answer_prefix"]
            if r["delta_vs_contextless_with_answer_prefix"] is not None
            else -10**9,
            reverse=True,
        )
        for rank, doc_score in enumerate(doc_scores, start=1):
            doc_score["rank"] = rank
        v1.append_jsonl(
            out_path,
            {
                "workdir_alias": alias,
                "workdir_name": entry["original_name"],
                "question": entry["question"],
                "sentence_id": sentence_id,
                "raw_sentence_id": record.get("raw_sentence_id"),
                "sensitivity_rank": record.get("sensitivity_rank"),
                "sentence_cti": record.get("sentence_cti"),
                "cti_mode": record.get("cti_mode"),
                "cti_method": record.get("cti_method"),
                "sentence": sentence,
                "answer_prefix_before": answer_prefix,
                "answer_prefix_chars": record.get("answer_prefix_chars", len(answer_prefix)),
                "answer_prefix_excerpt": record.get("answer_prefix_excerpt", suffix_excerpt(answer_prefix, args.summary_excerpt_chars)),
                "base_avg_logprob": v1.finite_float(base_avg),
                "doc_scores": doc_scores,
                "top_docs": doc_scores[: args.paragraph_doc_topk],
            },
        )
        completed.add((alias, sentence_id))


def remove_one_chunk(chunks: Sequence[EvidenceChunk], remove_idx: int) -> str:
    return "\n\n".join(chunk.text for idx, chunk in enumerate(chunks) if idx != remove_idx)


def done_paragraph_keys(path: Path) -> set:
    return {
        (r.get("workdir_alias"), int(r.get("sentence_id", -1)), int(r.get("doc_id", -1)))
        for r in v1.read_jsonl(path)
        if "sentence_id" in r and "doc_id" in r
    }


def compute_paragraph_perturbation_chunks(
    args: argparse.Namespace,
    output_dir: Path,
    entry: Dict[str, Any],
    doc_by_id: Dict[int, v1.Doc],
    model: torch.nn.Module,
    tokenizer: Any,
) -> None:
    alias = entry["alias"]
    doc_records = [
        r for r in v1.read_jsonl(output_dir / "doc_perturbation.jsonl") if r.get("workdir_alias") == alias
    ]
    out_path = output_dir / "paragraph_perturbation.jsonl"
    completed = done_paragraph_keys(out_path)
    for record in tqdm(doc_records, desc=f"{alias} chunk perturbation"):
        sentence_id = int(record["sentence_id"])
        sentence = record["sentence"]
        answer_prefix = record.get("answer_prefix_before", "")
        for doc_rank, doc_meta in enumerate(record.get("top_docs", [])[: args.paragraph_doc_topk], start=1):
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
            chunks = split_doc_chunks_for_perturbation(doc, args.paragraph_chunk_sentences, args.min_paragraph_chars)
            full_avg = doc_meta.get("avg_logprob")
            if full_avg is None:
                full_avg, _, _ = v1.score_sentence(
                    model,
                    tokenizer,
                    build_doc_answer_prefix(entry["question"], doc, args.perturb_doc_chars, answer_prefix),
                    sentence,
                    args.max_input_tokens,
                )
            chunk_scores = []
            for idx, chunk in enumerate(chunks):
                ablated_doc = v1.Doc(
                    doc_id=doc.doc_id,
                    number=doc.number,
                    path=doc.path,
                    title=doc.title,
                    publish_time=doc.publish_time,
                    text=remove_one_chunk(chunks, idx),
                )
                without_avg, _, _ = v1.score_sentence(
                    model,
                    tokenizer,
                    build_doc_answer_prefix(entry["question"], ablated_doc, args.perturb_doc_chars, answer_prefix),
                    sentence,
                    args.max_input_tokens,
                )
                importance = float(full_avg) - without_avg
                chunk_scores.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "section_title": chunk.section_title,
                        "start_sentence": chunk.start_sentence,
                        "end_sentence": chunk.end_sentence,
                        "chars": len(chunk.text),
                        "excerpt": v1.trim_chars(chunk.text, args.summary_excerpt_chars),
                        "without_chunk_avg_logprob": v1.finite_float(without_avg),
                        "importance_full_minus_without": v1.finite_float(importance),
                    }
                )
            chunk_scores.sort(
                key=lambda r: r["importance_full_minus_without"]
                if r["importance_full_minus_without"] is not None
                else -10**9,
                reverse=True,
            )
            for rank, chunk_score in enumerate(chunk_scores, start=1):
                chunk_score["rank"] = rank
            v1.append_jsonl(
                out_path,
                {
                    "workdir_alias": alias,
                    "workdir_name": entry["original_name"],
                    "question": entry["question"],
                    "sentence_id": sentence_id,
                    "raw_sentence_id": record.get("raw_sentence_id"),
                    "sentence": sentence,
                    "answer_prefix_before": answer_prefix,
                    "answer_prefix_chars": record.get("answer_prefix_chars", len(answer_prefix)),
                    "answer_prefix_excerpt": record.get("answer_prefix_excerpt", suffix_excerpt(answer_prefix, args.summary_excerpt_chars)),
                    "cti_mode": record.get("cti_mode"),
                    "cti_method": record.get("cti_method"),
                    "doc_rank": doc_rank,
                    "doc_id": doc.doc_id,
                    "doc_number": doc.number,
                    "doc_title": doc.title,
                    "doc_path": doc.path,
                    "full_doc_avg_logprob": v1.finite_float(float(full_avg)),
                    "paragraph_scores": chunk_scores,
                    "top_paragraphs": chunk_scores[:3],
                },
            )
            completed.add((alias, sentence_id, doc_id))


def build_summary(output_dir: Path, manifest: Sequence[Dict[str, Any]]) -> None:
    sentence_records = v1.read_jsonl(output_dir / "sentence_cti.jsonl")
    failed_records = v1.read_jsonl(output_dir / "cti_failed.jsonl")
    doc_records = v1.read_jsonl(output_dir / "doc_perturbation.jsonl")
    para_records = v1.read_jsonl(output_dir / "paragraph_perturbation.jsonl")
    skipped_count = 0
    for skipped_path in (output_dir / "embedding_debug").glob("*/skipped_answer_sentences.json"):
        try:
            skipped_count += int(json.loads(skipped_path.read_text(encoding="utf-8")).get("skipped_count", 0))
        except Exception:
            continue
    docs_by_sentence = {(r.get("workdir_alias"), r.get("sentence_id")): r for r in doc_records}
    paras_by_sentence_doc = {
        (r.get("workdir_alias"), r.get("sentence_id"), r.get("doc_id")): r for r in para_records
    }
    selected_workdirs = sorted({r.get("workdir_alias") for r in sentence_records})
    cti_modes = sorted({str(r.get("cti_mode") or r.get("cti_method")) for r in sentence_records})
    lines = [
        "# data_1 DeepSeek MIRAGE v2 严格归因摘要",
        "",
        f"- 更新时间：{v1.now_ts()}",
        f"- manifest workdir 数：{len(manifest)}",
        f"- 已处理 workdir：{', '.join(selected_workdirs) if selected_workdirs else '无'}",
        f"- CTI 模式：{', '.join(cti_modes) if cti_modes else '无'}",
        f"- sentence_cti 成功行数：{len(sentence_records)}",
        f"- cti_failed 行数：{len(failed_records)}",
        f"- 跳过结构句数：{skipped_count}",
        f"- doc_perturbation 行数：{len(doc_records)}",
        f"- paragraph_perturbation 行数：{len(para_records)}",
        "",
        "## Top 敏感句与引用依据",
        "",
    ]
    for record in sorted(sentence_records, key=lambda r: r.get("sensitivity_rank", 10**9))[:20]:
        alias = record.get("workdir_alias")
        sentence_id = record.get("sentence_id")
        lines.append(f"### {alias} / sentence-{int(sentence_id):04d} / rank {record.get('sensitivity_rank')}")
        lines.append("")
        lines.append(f"- CTI：{record.get('sentence_cti')}")
        lines.append(f"- 答案前文长度：{record.get('answer_prefix_chars')}")
        lines.append(f"- 句子：{record.get('sentence')}")
        lines.append("- CTI 语义上下文：")
        for chunk in record.get("selected_context_chunks", [])[:5]:
            lines.append(
                f"  - chunk {chunk.get('chunk_id')} / 资料{chunk.get('doc_number')} / "
                f"score {chunk.get('score')}：{chunk.get('doc_title')}"
            )
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
                    for para in para_record.get("top_paragraphs", [])[:2]:
                        lines.append(
                            f"    - chunk {para.get('chunk_id')} / rank {para.get('rank')} / "
                            f"importance {para.get('importance_full_minus_without')}：{para.get('excerpt')}"
                        )
        lines.append("")
    if failed_records:
        lines.extend(["## CTI 失败句", ""])
        for record in failed_records[:20]:
            lines.append(f"- sentence-{int(record.get('sentence_id')):04d}: {record.get('cti_error')}｜{record.get('sentence')}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(output_dir: Path) -> None:
    v1.validate_outputs(output_dir)
    valid_methods = {"inseq_saliency", "sentence_logprob_delta_with_answer_prefix"}
    for record in v1.read_jsonl(output_dir / "sentence_cti.jsonl"):
        if record.get("cti_method") not in valid_methods:
            raise RuntimeError("unknown CTI record found in sentence_cti.jsonl")
        if "answer_prefix_chars" not in record:
            raise RuntimeError("answer_prefix_chars missing from sentence_cti.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--workdir-limit", type=int, default=1)
    parser.add_argument("--workdir-alias", action="append", default=[])
    parser.add_argument("--doc-limit", type=int, default=100)
    parser.add_argument("--generation-context-chunks", type=int, default=16)
    parser.add_argument("--generation-chunk-chars", type=int, default=900)
    parser.add_argument("--generated-max-new-tokens", type=int, default=3072)
    parser.add_argument("--no-think-prefill", action="store_true")
    parser.add_argument("--force-generate-history", action="store_true")
    parser.add_argument("--title-similarity-threshold", type=float, default=0.35)
    parser.add_argument("--cti-context-docs", type=int, default=5)
    parser.add_argument("--top-sensitive-sentences", type=int, default=100)
    parser.add_argument("--paragraph-doc-topk", type=int, default=3)
    parser.add_argument("--paragraph-chunk-sentences", type=int, default=5)
    parser.add_argument("--cti-mode", choices=["token_saliency", "sentence_logprob"], default="token_saliency")
    parser.add_argument("--sentence-limit", type=int, default=0, help="Smoke-test limit; 0 keeps all content sentences.")
    parser.add_argument("--cti-method", choices=["saliency"], default="saliency")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--retry-cti-failed", action="store_true")
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
    manifest = v1.build_manifest(data_root, args.doc_limit)
    annotate_manifest_histories(manifest, None)

    runnable_entries = v1.selected_manifest_entries(manifest, args.workdir_limit, args.workdir_alias)
    if args.manifest_only:
        v1.write_json(output_dir / "manifest.json", manifest)
        print(f"[data1-v2] wrote manifest for {len(manifest)} workdirs; runnable={len(runnable_entries)}")
        return
    if not runnable_entries:
        raise RuntimeError("no runnable data_1 workdir found")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepSeek attribution")
    if args.cti_mode == "sentence_logprob":
        args.cti_method = "sentence_logprob"

    update_heartbeat(output_dir, phase="load_embedding", model=args.embedding_model)
    embedder = BgeEmbedder(
        args.embedding_model,
        args.embedding_device,
        args.embedding_batch_size,
        args.embedding_max_length,
    )

    update_heartbeat(output_dir, phase="load_model", model=args.model)
    model, tokenizer, model_mirage = v1.load_models(args)

    generated_manifest_entry: Optional[Dict[str, Any]] = None
    for entry in runnable_entries:
        alias = entry["alias"]
        update_heartbeat(output_dir, phase="load_workdir", workdir_alias=alias)
        docs = v1.load_manifest_docs(entry, args.doc_limit)
        update_heartbeat(output_dir, phase="build_evidence_chunks", workdir_alias=alias, doc_count=len(docs))
        chunks = build_evidence_chunks(docs, args.paragraph_chunk_sentences, args.min_paragraph_chars)
        chunk_texts = [f"{chunk.prompt_title()}\n{chunk.text}" for chunk in chunks]
        update_heartbeat(output_dir, phase="embed_evidence_chunks", workdir_alias=alias, chunk_count=len(chunks))
        chunk_embeddings = embedder.encode(chunk_texts)

        update_heartbeat(output_dir, phase="rank_generation_chunks", workdir_alias=alias)
        generation_chunks = rank_chunks(
            embedder,
            chunk_embeddings,
            chunks,
            entry["question"],
            args.generation_context_chunks,
        )
        write_sentence_context_debug(output_dir, alias, {0: generation_chunks})
        write_generation_debug(output_dir, alias, entry["question"], generation_chunks)

        generated_path = output_dir / "generated_history" / alias / "历史成果.txt"
        if args.force_generate_history or not generated_path.exists():
            update_heartbeat(output_dir, phase="generate_history", workdir_alias=alias)
            generated_text = generate_history(model, tokenizer, entry["question"], generation_chunks, args)
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            generated_path.write_text(generated_text, encoding="utf-8")
        else:
            generated_text = v1.read_text(generated_path)

        validation = validate_generated_history(
            entry,
            generated_text,
            embedder,
            args.title_similarity_threshold,
        )
        generated_manifest_entry = {
            "alias": alias,
            "generated_history_path": str(generated_path),
            **validation,
        }
        annotate_manifest_histories(manifest, generated_manifest_entry)
        v1.write_json(output_dir / "manifest.json", manifest)
        if validation["validation_errors"]:
            raise RuntimeError(f"{alias} generated history validation failed: {validation['validation_errors']}")
        if args.generate_only:
            continue

        sentence_items, skipped_sentences = split_answer_content_items(
            generated_text,
            args.min_sentence_chars,
            args.summary_excerpt_chars,
        )
        if args.sentence_limit > 0:
            sentence_items = sentence_items[: args.sentence_limit]
        v1.write_json(
            output_dir / "embedding_debug" / alias / "skipped_answer_sentences.json",
            {
                "workdir_alias": alias,
                "skipped_count": len(skipped_sentences),
                "skipped": skipped_sentences,
            },
        )
        if not sentence_items:
            raise RuntimeError(f"{alias} generated history has no valid sentences")

        sentence_contexts = {
            item.sentence_id: rank_chunks(
                embedder,
                chunk_embeddings,
                chunks,
                item.sentence,
                args.cti_context_docs,
            )
            for item in sentence_items
        }
        write_sentence_context_debug(output_dir, alias, sentence_contexts)

        sentence_records = compute_sentence_cti_records_strict(
            args,
            output_dir,
            entry,
            sentence_items,
            sentence_contexts,
            model,
            tokenizer,
            model_mirage,
        )
        compute_doc_perturbation_with_answer_prefix(args, output_dir, entry, docs, sentence_records, model, tokenizer)
        compute_paragraph_perturbation_chunks(
            args,
            output_dir,
            entry,
            {doc.doc_id: doc for doc in docs},
            model,
            tokenizer,
        )

    if args.generate_only:
        update_heartbeat(output_dir, phase="done")
        print(f"[data1-v2] generated histories written to {output_dir / 'generated_history'}")
        return

    update_heartbeat(output_dir, phase="summary")
    build_summary(output_dir, manifest)
    validate_outputs(output_dir)
    update_heartbeat(output_dir, phase="done")
    print(f"[data1-v2] complete; outputs written to {output_dir}")


if __name__ == "__main__":
    main()
