"""Parse SHAC training logs (msk-warp + DiffRL formats — they share the same iter line) and produce a comparison summary.

Usage:
    python scripts/phase4_plot_training.py \
        --runs phase3a-softref=tests/data/phase4_p3a_softref_train.log \
               phase3a-baseline=tests/data/phase4_p3a_baseline_train.log \
               diffrl=tests/data/diffrl_smoke50.log \
        --output tests/data/phase4_training_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ITER_RE = re.compile(
    r"^iter (\d+): ep loss ([-\d.]+|inf|nan), ep discounted loss ([-\d.]+|inf|nan), "
    r"ep len ([\d.]+|inf|nan), fps total ([\d.]+|inf|nan)"
)


def parse_log(path: Path):
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = ITER_RE.match(line.strip())
            if m:
                it = int(m.group(1))
                ep_loss = float(m.group(2)) if m.group(2) not in ("inf", "nan") else None
                ep_disc = float(m.group(3)) if m.group(3) not in ("inf", "nan") else None
                ep_len = float(m.group(4)) if m.group(4) not in ("inf", "nan") else 0.0
                fps = float(m.group(5)) if m.group(5) not in ("inf", "nan") else None
                rows.append({"iter": it, "ep_loss": ep_loss, "ep_disc": ep_disc, "ep_len": ep_len, "fps": fps})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="name=path pairs, e.g. 'phase3a=logs/x.log'")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    parsed = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        rows = parse_log(Path(path))
        parsed[name] = rows

    # Console summary
    print(f"\n{'iter':>5}", *[f"{n:>32}" for n in parsed.keys()])
    iters = sorted(set(r["iter"] for rows in parsed.values() for r in rows))
    rep_iters = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for it in rep_iters:
        if it not in iters:
            continue
        line = f"{it:>5}"
        for name, rows in parsed.items():
            row = next((r for r in rows if r["iter"] == it), None)
            if row is None:
                line += f"  {'-':>30}"
            else:
                el = row["ep_len"]; ep = row["ep_loss"]
                r_step = (-ep / el) if (ep is not None and el > 0) else None
                rs = f"{r_step:.2f}" if r_step is not None else "inf"
                line += f"  len={el:6.1f} loss={'inf' if ep is None else f'{ep:.1f}'} r/s={rs}"
        print(line)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(parsed, open(out_path, "w"), indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
