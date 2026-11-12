from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv_and_md(df: pd.DataFrame, base_path: str | Path, float_fmt: str = ".4f") -> None:
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)

    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = [
            "NaN" if (isinstance(row[c], float) and pd.isna(row[c]))
            else (format(row[c], float_fmt) if isinstance(row[c], float) else str(row[c]))
            for c in cols
        ]
        lines.append("| " + " | ".join(cells) + " |")
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_run_id(config: Dict[str, Any]) -> str:
    bidir = "bi" if config.get("bidirectional") else "uni"
    return (
        f"h{config['hidden_size']}_l{config['num_layers']}_{bidir}_"
        f"do{config['dropout']}_{config['optimizer']}_lr{config['lr']}_bs{config['batch_size']}"
    )


def setup_logger(name: str = "assignment6", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger
