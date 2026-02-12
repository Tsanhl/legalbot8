#!/usr/bin/env python3
"""
Batch retrieval quality audit for legal prompts.

Purpose:
- Quickly regression-check many topics without manually running full answers.
- Report inferred retrieval topic, source-mix quality, and retry pressure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemini_service import (  # type: ignore
    _infer_retrieval_profile,
    _rag_quality_audit,
    get_retrieved_content,
)


DEFAULT_PROMPTS: List[Tuple[str, str]] = [
    (
        "employment_eqa",
        "Equality Act 2010 direct vs indirect discrimination, section 13 and section 19, justify critically.",
    ),
    (
        "medical_end_of_life",
        "Medical law end of life: assisted suicide, withdrawal of CANH, Mental Capacity Act 2005 best interests, Article 8.",
    ),
    (
        "immigration_article8",
        "Advise on deportation under NIAA 2002 section 117C with Article 8 and KO (Nigeria).",
    ),
    (
        "competition_102",
        "Problem question on Article 102 abuse of dominance, self-preferencing and predatory pricing.",
    ),
    (
        "consumer_cra",
        "Consumer Rights Act 2015 unfair terms analysis under section 62 and section 64.",
    ),
    (
        "defamation",
        "Defamation Act 2013 serious harm and defences (truth, honest opinion, public interest) in online publication.",
    ),
    (
        "company_law",
        "Company law problem on directors duties under CA 2006 and minority shareholder unfair prejudice under section 994.",
    ),
    (
        "insolvency",
        "Insolvency Act 1986 wrongful trading, undervalue and preference claims with Sequana creditor duty.",
    ),
    (
        "tax_gaar",
        "Tax law essay on Ramsay principle, GAAR under Finance Act 2013, and Duke of Westminster.",
    ),
    (
        "pil_force",
        "Public international law: legality of force under Article 2(4) and self-defence under Article 51.",
    ),
]


@dataclass
class AuditRow:
    label: str
    prompt: str
    inferred_topic: str
    score: float
    needs_retry: bool
    expected_hits: int
    query_hits: int
    excluded_hits: int
    missing_must_cover: List[str]
    source_mix: dict
    rag_chars: int
    rag_error: str | None


def _load_prompt_file(path: Path) -> List[Tuple[str, str]]:
    prompts: List[Tuple[str, str]] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            label, prompt = line.split("\t", 1)
            prompts.append((label.strip() or f"prompt_{i}", prompt.strip()))
        else:
            prompts.append((f"prompt_{i}", line))
    return prompts


def run_audit(prompts: List[Tuple[str, str]], max_chunks: int, dry_run: bool = False) -> List[AuditRow]:
    rows: List[AuditRow] = []
    for label, prompt in prompts:
        profile = _infer_retrieval_profile(prompt)
        if dry_run:
            rows.append(
                AuditRow(
                    label=label,
                    prompt=prompt,
                    inferred_topic=profile.get("topic", "general_legal"),
                    score=0.0,
                    needs_retry=False,
                    expected_hits=0,
                    query_hits=0,
                    excluded_hits=0,
                    missing_must_cover=list(profile.get("must_cover") or []),
                    source_mix={"statutes": 0, "cases": 0, "secondary": 0},
                    rag_chars=0,
                    rag_error="dry-run",
                )
            )
            continue

        print(f"[audit] {label} -> topic={profile.get('topic', 'general_legal')}", flush=True)
        retrieved = get_retrieved_content(prompt, max_chunks=max_chunks)
        rag_text = (retrieved.get("content") or "") if isinstance(retrieved, dict) else ""

        if rag_text and not rag_text.startswith("[RAG]") and not rag_text.startswith("[RAG ERROR]"):
            audit = _rag_quality_audit(rag_text, profile)
            rag_chars = len(rag_text)
        else:
            audit = {
                "score": 0.0,
                "needs_retry": True,
                "expected_hits": 0,
                "query_hits": 0,
                "excluded_hits": 0,
                "missing_must_cover": profile.get("must_cover") or [],
                "mix": {"statutes": 0, "cases": 0, "secondary": 0},
            }
            rag_chars = 0

        rows.append(
            AuditRow(
                label=label,
                prompt=prompt,
                inferred_topic=profile.get("topic", "general_legal"),
                score=float(audit.get("score", 0.0)),
                needs_retry=bool(audit.get("needs_retry", False)),
                expected_hits=int(audit.get("expected_hits", 0)),
                query_hits=int(audit.get("query_hits", 0)),
                excluded_hits=int(audit.get("excluded_hits", 0)),
                missing_must_cover=list(audit.get("missing_must_cover") or []),
                source_mix=dict(audit.get("mix") or {}),
                rag_chars=rag_chars,
                rag_error=(retrieved.get("error") if isinstance(retrieved, dict) else "unknown"),
            )
        )
    return rows


def _print_summary(rows: List[AuditRow]) -> None:
    if not rows:
        print("No prompts audited.")
        return

    header = f"{'label':22} {'topic':36} {'score':>6} {'retry':>5} {'mix(s/c/sec)':>14} {'miss':>4}"
    print(header)
    print("-" * len(header))
    for r in rows:
        mix = r.source_mix or {}
        mix_txt = f"{mix.get('statutes',0)}/{mix.get('cases',0)}/{mix.get('secondary',0)}"
        print(
            f"{r.label[:22]:22} {r.inferred_topic[:36]:36} "
            f"{r.score:6.2f} {str(r.needs_retry):>5} {mix_txt:>14} {len(r.missing_must_cover):4d}"
        )

    retries = sum(1 for r in rows if r.needs_retry)
    avg_score = sum(r.score for r in rows) / len(rows)
    print("")
    print(f"Prompts: {len(rows)} | Needs-retry: {retries} | Avg score: {avg_score:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch audit retrieval quality across legal prompts.")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Optional TSV/txt file. Format: label<TAB>prompt (or one prompt per line).",
    )
    parser.add_argument("--max-chunks", type=int, default=28, help="Chunk count per prompt.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/retrieval_quality_audit.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only infer topic/profile without running retrieval.",
    )
    args = parser.parse_args()

    prompts = _load_prompt_file(args.prompts_file) if args.prompts_file else DEFAULT_PROMPTS
    rows = run_audit(prompts, max_chunks=max(1, args.max_chunks), dry_run=bool(args.dry_run))
    _print_summary(rows)

    payload = [asdict(r) for r in rows]
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
