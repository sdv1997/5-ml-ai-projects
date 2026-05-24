"""
Day 3 — What's Up, Docs?
Summarization de papers de ciencias sociales (SocArXiv) con LLM local.
Métrica: ROUGE-2 (macro).

Pipeline:
1. Extraer intro + conclusión del paper (secciones más próximas al abstract)
2. Truncar a max_chars para caber en contexto
3. Generar abstract con Qwen2.5-3B-Instruct via vLLM
4. Evaluar ROUGE-2 en train, generar submission para test
"""

import os
# Redirigir caché de HF a /workspace para no agotar los 20 GB del overlay raíz
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer

DATA_DIR = Path("data/day03")
OUT_DIR = Path("day_03")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# ─── Sección extraction ──────────────────────────────────────────────────────

SECTION_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)

INTRO_KEYWORDS = {"introduction", "background", "overview", "context", "motivation"}
CONCLUSION_KEYWORDS = {
    "conclusion", "conclusions", "discussion", "summary", "implications",
    "findings", "results", "concluding", "final remarks",
}


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (header, body) pairs. First section may have empty header."""
    headers = list(SECTION_RE.finditer(text))
    if not headers:
        return [("", text)]
    sections = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections.append((m.group(1).strip(), text[m.end():end].strip()))
    # text before first header
    preamble = text[: headers[0].start()].strip()
    if preamble:
        sections.insert(0, ("", preamble))
    return sections


def extract_key_sections(text: str, max_chars: int = 6000) -> str:
    """
    Extract intro + conclusion sections and concatenate up to max_chars.
    Falls back to truncated full text if no sections found.
    """
    sections = split_sections(text)

    intro_parts, conclusion_parts, other_parts = [], [], []
    for header, body in sections:
        h = header.lower()
        if any(k in h for k in INTRO_KEYWORDS) or header == "":
            intro_parts.append(body)
        elif any(k in h for k in CONCLUSION_KEYWORDS):
            conclusion_parts.append(body)
        else:
            other_parts.append(body)

    # Priority: intro → conclusion → rest
    combined = "\n\n".join(intro_parts + conclusion_parts)
    if len(combined) < max_chars // 2 and other_parts:
        combined += "\n\n" + "\n\n".join(other_parts)

    return combined[:max_chars]


# ─── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert academic editor. Your task is to write a concise abstract "
    "for a social science research paper. The abstract should be 150-250 words. "
    "Use precise academic language. Focus on: research question, methodology, key findings, "
    "and implications. Use the exact terminology from the paper — do not paraphrase unnecessarily. "
    "Output only the abstract text, no preamble, no labels."
)


def build_prompt(paper_text: str) -> str:
    return (
        f"Write an abstract for the following research paper excerpt:\n\n"
        f"{paper_text}\n\n"
        f"Abstract:"
    )


# ─── vLLM inference ──────────────────────────────────────────────────────────

def run_vllm(
    texts: list[str],
    model_id: str = MODEL_ID,
    max_new_tokens: int = 350,
    temperature: float = 0.1,
) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        quantization="awq",
        dtype="float16",
        max_model_len=12000,
        gpu_memory_utilization=0.85,
    )
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop=["###", "\n\n\n"],
    )

    messages_batch = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(t)},
        ]
        for t in texts
    ]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text.strip() for o in outputs]


# ─── ROUGE-2 evaluation ───────────────────────────────────────────────────────

def evaluate_rouge2(preds: list[str], refs: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
    scores = [scorer.score(r, p)["rouge2"].fmeasure for r, p in zip(refs, preds)]
    return sum(scores) / len(scores)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Run on first 100 train rows to get ROUGE-2 estimate")
    parser.add_argument("--max-chars", type=int, default=24000,
                        help="Max chars of paper text fed to LLM")
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test_features.csv")
    sub_fmt = pd.read_csv(DATA_DIR / "submission_format.csv")

    if args.eval_only:
        print(f"Eval mode: using first 100 train rows")
        subset = train.head(100)
        excerpts = [extract_key_sections(t, args.max_chars) for t in subset["text"]]
        preds = run_vllm(excerpts, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        score = evaluate_rouge2(preds, subset["summary"].tolist())
        print(f"\nROUGE-2 (n=100 train): {score:.4f}")

        # Save sample for inspection
        sample = subset[["paper_id", "summary"]].copy()
        sample["predicted"] = preds
        sample_path = OUT_DIR / "eval_sample.csv"
        sample.to_csv(sample_path, index=False)
        print(f"Sample saved to {sample_path}")
        return

    # Full test set
    print(f"Generating summaries for {len(test)} test papers...")
    excerpts = [extract_key_sections(t, args.max_chars) for t in test["text"]]
    preds = run_vllm(excerpts, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    submission = sub_fmt.copy()
    id_to_pred = dict(zip(test["paper_id"], preds))
    submission["summary"] = submission["paper_id"].map(id_to_pred)
    out_path = OUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path} ({len(submission)} rows)")

    # Also score on train for reference
    print("\nScoring on full train set for reference...")
    train_excerpts = [extract_key_sections(t, args.max_chars) for t in train["text"]]
    train_preds = run_vllm(train_excerpts, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    train_score = evaluate_rouge2(train_preds, train["summary"].tolist())
    print(f"ROUGE-2 (train): {train_score:.4f}")

    results = {
        "model": MODEL_ID,
        "max_chars": args.max_chars,
        "temperature": args.temperature,
        "rouge2_train": train_score,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
