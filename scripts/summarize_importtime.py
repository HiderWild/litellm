# ruff: noqa: T201
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

_LINE_RE = re.compile(r"^import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s(\s*)(.*)$")


@dataclass(frozen=True, slots=True)
class ImportRow:
    name: str
    self_ms: float
    cumulative_ms: float
    depth: int


def parse_importtime(text: str) -> list[ImportRow]:
    rows: list[ImportRow] = []
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match is None:
            continue
        self_us, cum_us, indent, name = match.groups()
        rows.append(
            ImportRow(
                name=name,
                self_ms=int(self_us) / 1000.0,
                cumulative_ms=int(cum_us) / 1000.0,
                depth=len(indent),
            )
        )
    return rows


def _format_table(title: str, rows: Sequence[ImportRow]) -> str:
    lines = [title, f"{'self ms':>12} | {'cum ms':>12} | module"]
    for row in rows:
        lines.append(f"{row.self_ms:>12.1f} | {row.cumulative_ms:>12.1f} | {row.name}")
    return "\n".join(lines)


def summarize_importtime(text: str, top: int = 40) -> str:
    rows = parse_importtime(text)
    if not rows:
        return "No import-time entries found."
    by_self = sorted(rows, key=lambda r: r.self_ms, reverse=True)[:top]
    litellm_rows = [r for r in rows if r.name.startswith("litellm")]
    by_cum = sorted(litellm_rows, key=lambda r: r.cumulative_ms, reverse=True)[:top]
    total = max((r.cumulative_ms for r in rows), default=0.0)
    sections = [
        f"Total measured import wall time (max cumulative): {total:.1f} ms",
        _format_table(
            f"\nTop {len(by_self)} modules by SELF time (all packages):", by_self
        ),
        _format_table(
            f"\nTop {len(by_cum)} litellm.* modules by CUMULATIVE time:", by_cum
        ),
    ]
    return "\n".join(sections)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Python -X importtime output."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to importtime stderr file. Reads stdin when omitted.",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args(argv)
    if args.path:
        with open(args.path, encoding="utf-8") as handle:
            text = handle.read()
    else:
        text = sys.stdin.read()
    print(summarize_importtime(text, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
