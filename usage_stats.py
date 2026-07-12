"""Cumulative token stats and the stdout "savings ticker".

Detailed per-model breakdown goes to the log file (stderr). A one-line tier
summary overwrites the current terminal line on stdout while Claude Code runs.
"""
from __future__ import annotations

import logging
import sys

from route_config import CONFIG

log = logging.getLogger("model-router")

_STDOUT_IS_TTY = sys.stdout.isatty()

# Reference price for savings comparison — read from the opus route in config, fallback to $5/MTok.
_OPUS_PRICE_PER_MTOK: float = next(
    (float(r.get("price_per_mtok", 5.0)) for r in CONFIG.get("routes", []) if r.get("tier") == "opus"),
    5.0,
)

# Per-model cumulative token counters {upstream_model: {"input": int, "output": int, "price_per_mtok": float}}
_token_stats: dict[str, dict] = {}

# Maps tier names to short display labels used in the status line (e.g. "haiku" → "low").
_TIER_LABEL = {"haiku": "low", "sonnet": "med", "opus": "high", "fable": "top"}
_TIER_ORDER = ["low", "med", "high", "top"]


def _record_tokens(upstream_model: str, in_tok: int, out_tok: int, price_per_mtok: float, backend_label: str, tier: str = "") -> None:
    """Update cumulative stats and emit a compact status line."""
    log.info(
        "[model-router] tokens  in=%-6d out=%-6d backend=%-12s model=%s",
        in_tok, out_tok, backend_label, upstream_model,
    )
    entry = _token_stats.setdefault(upstream_model, {"input": 0, "output": 0, "price_per_mtok": price_per_mtok, "tier": tier})
    entry["input"] += in_tok
    entry["output"] += out_tok
    entry["price_per_mtok"] = price_per_mtok

    # Detailed per-model totals → log file only
    grand_total = sum(s["input"] + s["output"] for s in _token_stats.values())
    grand_cost = sum((s["input"] + s["output"]) / 1_000_000 * s["price_per_mtok"] for s in _token_stats.values())
    grand_opus_cost = grand_total / 1_000_000 * _OPUS_PRICE_PER_MTOK
    log.info("[model-router] ── running totals ──────────────────────────────────────────────────")
    for mdl, s in _token_stats.items():
        mtok = s["input"] + s["output"]
        pct = 100.0 * mtok / grand_total if grand_total else 0.0
        cost = mtok / 1_000_000 * s["price_per_mtok"]
        cost_str = "FREE    " if s["price_per_mtok"] == 0 else f"${cost:.4f}"
        saved = mtok / 1_000_000 * (_OPUS_PRICE_PER_MTOK - s["price_per_mtok"])
        saved_str = f"  saved ${saved:.4f}" if saved > 0 else ""
        tier_label = _TIER_LABEL.get(s.get("tier", ""), s.get("tier", ""))
        log.info("[model-router]   %-4s %-46s %8d tok  %5.1f%%  %s%s", tier_label, mdl[:46], mtok, pct, cost_str, saved_str)
    total_saved = grand_opus_cost - grand_cost
    log.info(
        "[model-router]   %-4s %-46s %8d tok  100.0%%  $%.4f total  (saved $%.4f)",
        "", "ALL MODELS", grand_total, grand_cost, total_saved,
    )

    # Compact tier summary → stdout, single updating line
    by_tier: dict[str, dict] = {}
    for s in _token_stats.values():
        label = _TIER_LABEL.get(s.get("tier", ""), s.get("tier", "") or "?")
        e = by_tier.setdefault(label, {"tokens": 0, "cost": 0.0, "saved": 0.0})
        mtok2 = s["input"] + s["output"]
        e["tokens"] += mtok2
        e["cost"] += mtok2 / 1_000_000 * s["price_per_mtok"]
        savings = mtok2 / 1_000_000 * (_OPUS_PRICE_PER_MTOK - s["price_per_mtok"])
        if savings > 0:
            e["saved"] += savings

    parts = []
    for label in _TIER_ORDER:
        if label not in by_tier:
            continue
        e = by_tier[label]
        cost_str = "FREE" if e["cost"] == 0 else f"${e['cost']:.4f}"
        parts.append(f"{label}: {e['tokens']:,} tok {cost_str}")
    total_saved2 = sum(e["saved"] for e in by_tier.values())
    if total_saved2 > 0:
        parts.append(f"saved: ${total_saved2:.4f}")
    line = "  |  ".join(parts)
    if _STDOUT_IS_TTY:
        print(f"\033[2K\r{line}", end="", flush=True)
    else:
        print(line, flush=True)
