"""The hand-written held-out evaluation set, and the drift report for it.

Rule 4: *the held-out evaluation set is hand-written, never template-generated
or back-translated, and is reported separately from the in-distribution test
split.* This module loads it and measures how far it actually sits from the
generator's output, because the rule is only worth anything if the separation
is real.

**The circularity, stated up front.** The same author wrote the back-translation
phrase banks in ``dataset_gen.py`` and the instructions in
``evals/heldout.yaml``. "Independent of the templates" is therefore a weaker
claim here than it sounds: knowing the phrase banks makes it easy to echo them
without meaning to. That is why ``drift_report`` exists -- it turns the claim
into numbers (shared vocabulary, reused n-grams, length and register spread)
that a reader can check instead of taking on trust. The numbers are in
ARCHITECTURE section 10, and the residual circularity stays in section 9 until
an outside author replaces it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from planguard.dsl.schema import MissionPlan

#: Registers the set deliberately spans. Tagged per item so the drift report
#: can show the spread rather than assert it.
REGISTERS = (
    "radio",  # clipped, abbreviated, no pleasantries
    "formal",  # written tasking order
    "chatty",  # conversational, hedged, digressive
    "bulleted",  # list-shaped
    "terse",  # minimal but not radio jargon
)


@dataclass(frozen=True)
class HeldoutItem:
    item_id: str
    instruction: str
    plan: MissionPlan
    register: str
    notes: str | None = None


def load_heldout(path: str | Path) -> list[HeldoutItem]:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    items: list[HeldoutItem] = []
    for entry in raw["items"]:
        items.append(
            HeldoutItem(
                item_id=entry["id"],
                instruction=entry["instruction"].strip(),
                plan=MissionPlan.model_validate(entry["plan"]),
                register=entry["register"],
                notes=entry.get("notes"),
            )
        )
    return items


def _words(text: str) -> list[str]:
    return [w.strip(".,;:()-—\"'").lower() for w in text.split() if w.strip(".,;:()-—\"'")]


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def drift_report(
    heldout: list[str], generated: list[str], n: int = 4, seed: int = 0
) -> dict[str, float]:
    """How far the hand-written set sits from the generator's output.

    The number that matters most is ``ngram_echo``: the fraction of the
    held-out set's n-grams that also occur somewhere in the generated corpus.
    A high value means the "hand-written" set is quietly reproducing the
    phrase banks, and the separate reporting rule 4 demands would be
    theatre. ``vocab_shared`` is the softer version of the same question --
    some overlap is unavoidable, since both talk about the same map.

    Type-token ratio is reported on a **size-matched** sample of the generated
    corpus as well as on the whole of it. TTR falls as a corpus grows, so
    comparing 150 hand-written instructions against 600 generated ones and
    calling the difference lexical richness would be an artefact of the sample
    sizes, not a finding.
    """
    import random

    h_words = [_words(t) for t in heldout]
    g_words = [_words(t) for t in generated]
    h_vocab = set().union(*h_words) if h_words else set()
    g_vocab = set().union(*g_words) if g_words else set()

    h_ngrams: set[tuple[str, ...]] = set()
    for ws in h_words:
        h_ngrams |= _ngrams(ws, n)
    g_ngrams: set[tuple[str, ...]] = set()
    for ws in g_words:
        g_ngrams |= _ngrams(ws, n)

    matched = g_words
    if len(g_words) > len(h_words) and h_words:
        matched = random.Random(seed).sample(g_words, len(h_words))
    m_vocab = set().union(*matched) if matched else set()
    m_tokens = sum(len(ws) for ws in matched)

    h_lens = [len(ws) for ws in h_words]
    g_lens = [len(ws) for ws in g_words]
    h_tokens = sum(h_lens)
    g_tokens = sum(g_lens)

    return {
        "n_heldout": float(len(heldout)),
        "n_generated": float(len(generated)),
        "heldout_vocab": float(len(h_vocab)),
        "generated_vocab": float(len(g_vocab)),
        # share of held-out vocabulary the generator can also produce
        "vocab_shared": len(h_vocab & g_vocab) / len(h_vocab) if h_vocab else 0.0,
        # share of held-out vocabulary the generator never uses
        "vocab_unique_to_heldout": len(h_vocab - g_vocab) / len(h_vocab) if h_vocab else 0.0,
        f"ngram_echo_n{n}": len(h_ngrams & g_ngrams) / len(h_ngrams) if h_ngrams else 0.0,
        "heldout_mean_words": h_tokens / len(h_lens) if h_lens else 0.0,
        "generated_mean_words": g_tokens / len(g_lens) if g_lens else 0.0,
        "heldout_min_words": float(min(h_lens)) if h_lens else 0.0,
        "heldout_max_words": float(max(h_lens)) if h_lens else 0.0,
        "heldout_type_token_ratio": len(h_vocab) / h_tokens if h_tokens else 0.0,
        "generated_type_token_ratio": len(g_vocab) / g_tokens if g_tokens else 0.0,
        # the comparable one: same number of instructions on both sides
        "generated_type_token_ratio_matched": len(m_vocab) / m_tokens if m_tokens else 0.0,
    }


def plan_shape_report(items: list[HeldoutItem]) -> dict[str, float]:
    """Does the held-out set cover the shapes the generator produces?

    Distribution drift cuts both ways. A held-out set written in beautifully
    varied prose that only ever asks for one-task plans would be easy and
    would not test anything.
    """
    kinds: Counter[str] = Counter()
    for item in items:
        for task in item.plan.tasks:
            kinds[task.kind] += 1
    n_tasks = [len(i.plan.tasks) for i in items]
    report = {
        "n_items": float(len(items)),
        "mean_tasks": sum(n_tasks) / len(n_tasks) if n_tasks else 0.0,
        "max_tasks": float(max(n_tasks)) if n_tasks else 0.0,
        "frac_multi_task": sum(1 for n in n_tasks if n > 1) / len(n_tasks) if n_tasks else 0.0,
        "frac_with_dependency": (
            sum(1 for i in items if any(t.depends_on for t in i.plan.tasks)) / len(items)
            if items
            else 0.0
        ),
        "frac_with_link": (
            sum(1 for i in items if any(t.requires_link for t in i.plan.tasks)) / len(items)
            if items
            else 0.0
        ),
        "frac_multi_aircraft": (
            sum(1 for i in items if any(t.aircraft_count > 1 for t in i.plan.tasks)) / len(items)
            if items
            else 0.0
        ),
        "frac_with_start_after": (
            sum(1 for i in items if any(t.start_after_s for t in i.plan.tasks)) / len(items)
            if items
            else 0.0
        ),
    }
    for kind in sorted(kinds):
        report[f"tasks_{kind}"] = float(kinds[kind])
    for register in REGISTERS:
        report[f"register_{register}"] = float(sum(1 for i in items if i.register == register))
    return report
