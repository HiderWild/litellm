from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from scripts.summarize_importtime import (
    ImportRow,
    parse_importtime,
    summarize_importtime,
)

SAMPLE = """import time: self [us] | cumulative | imported package
import time:       100 |        500 | litellm
import time:        50 |        150 |   litellm.proxy
import time:       200 |        200 | tiktoken
import time:        10 |         10 | _io
"""


def test_parse_importtime_extracts_self_cumulative_and_depth() -> None:
    rows = parse_importtime(SAMPLE)

    assert [r.name for r in rows] == ["litellm", "litellm.proxy", "tiktoken", "_io"]
    assert rows[0] == ImportRow(name="litellm", self_ms=0.1, cumulative_ms=0.5, depth=0)
    assert rows[1].depth > rows[0].depth


def test_summarize_importtime_orders_self_descending_and_flags_litellm() -> None:
    output = summarize_importtime(SAMPLE, top=5)

    assert "Total measured import wall time (max cumulative): 0.5 ms" in output
    assert "tiktoken" in output
    assert "litellm" in output
    tiktoken_index = output.index("tiktoken")
    litellm_proxy_index = output.index("litellm.proxy")
    assert tiktoken_index < litellm_proxy_index
