"""Spreadsheet analysis for Nap Bot — .xlsx and .csv files.

Uses pandas (openpyxl as its .xlsx engine) so any tabular file — orders,
inventory, ad-performance exports, whatever — can be read and reasoned over,
not just plain text.

Honest limit: a sheet with tens of thousands of rows cannot ALL fit in an
LLM's context window — no amount of engineering changes that. So this reads
and computes summary statistics over the FULL file (exact, not sampled), but
only shows a bounded sample of raw rows. The model gets real numbers over
everything, plus a representative look at the actual data.
"""

from __future__ import annotations

from pathlib import Path

from core.deps import ensure_package
from utils.logger import get_logger

log = get_logger(__name__)

MAX_ROWS_SHOWN = 200


def _to_markdown_table(df) -> str:
    """Render a DataFrame as a pipe-delimited markdown table.

    Deliberately NOT pandas' own `to_string()` — that pads columns with
    spaces assuming a monospace terminal, which looks fine in a shell but
    turns into a garbled mess once rendered as HTML in a proportional-width
    chat bubble. A markdown table is standard for the model to read/write,
    and the chat UI renders it as a real <table> (see renderMarkdown in
    frontend/index.html).
    """
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if pd_isna(v):
                cells.append("")
            elif isinstance(v, float):
                # Fixed 2dp so e.g. describe()'s stats show "300.00", not a
                # trailing-zero-stripped "300.0" — str() on a float is too
                # inconsistent for a table meant to look uniform.
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pd_isna(value) -> bool:
    """Tiny standalone NaN check so this module doesn't need to import
    pandas at module load time just for one helper."""
    return value != value  # NaN is the only value that isn't equal to itself


def read_spreadsheet(path: str) -> str:
    """Read a .xlsx/.xls/.csv file. Returns shape, column names, full summary
    statistics for numeric columns, and a sample of rows."""
    pandas = ensure_package("pandas")
    if pandas is None:
        return "Error: pandas is not installed."
    ensure_package("openpyxl")  # the engine pandas needs for .xlsx/.xls

    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"File not found: {p}"

    try:
        if p.suffix.lower() == ".csv":
            df = pandas.read_csv(p)
        else:
            df = pandas.read_excel(p)
    except Exception as exc:  # noqa: BLE001 - surface to the model, don't crash
        return f"Couldn't read {p.name}: {exc}"

    lines = [
        f"File: {p.name}",
        f"Rows: {len(df)}  Columns: {len(df.columns)}",
        f"Column names: {', '.join(str(c) for c in df.columns)}",
    ]

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        lines.append("\nNumeric column summary (computed over ALL rows, not a sample):")
        stats = numeric.describe().round(2).reset_index().rename(columns={"index": "stat"})
        lines.append(_to_markdown_table(stats))

    shown = df.head(MAX_ROWS_SHOWN)
    lines.append(f"\nFirst {len(shown)} of {len(df)} rows:")
    lines.append(_to_markdown_table(shown))
    if len(df) > MAX_ROWS_SHOWN:
        lines.append(
            f"\n(({len(df) - MAX_ROWS_SHOWN} more rows exist but aren't shown here — "
            "the summary statistics above still cover the full file.))"
        )

    log.info("read_spreadsheet %s -> %d rows, %d cols", p, len(df), len(df.columns))
    return "\n".join(lines)
