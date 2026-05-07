"""Quick parser for the cfd_baseline.log / cfd_cfd.log SHAC training output.

Extracts per-iter (ep_loss, ep_len, value_loss, grad_norm_before_clip, fps)
and prints a side-by-side comparison.
"""

import re
import sys


_PAT = re.compile(
    r"iter (\d+): ep loss ([\-\d\.inf]+), ep discounted loss ([\-\d\.inf]+), "
    r"ep len ([\d\.]+), fps total ([\d\.]+), value loss ([\d\.]+), "
    r"grad norm before clip ([\d\.]+), grad norm after clip ([\d\.]+)"
)


def parse(path):
    rows = []
    with open(path, "r") as f:
        text = f.read()
    for m in _PAT.finditer(text):
        rows.append(dict(
            iter=int(m.group(1)),
            ep_loss=float(m.group(2)) if m.group(2) != "inf" else float("inf"),
            ep_disc=float(m.group(3)) if m.group(3) != "inf" else float("inf"),
            ep_len=float(m.group(4)),
            fps=float(m.group(5)),
            value_loss=float(m.group(6)),
            grad_norm_before=float(m.group(7)),
            grad_norm_after=float(m.group(8)),
        ))
    return rows


def main(baseline_path, cfd_path):
    base = parse(baseline_path)
    cfd = parse(cfd_path)
    n = max(len(base), len(cfd))
    print(f"{'iter':>4}  {'b_eploss':>9}  {'c_eploss':>9}  {'b_eplen':>7}  {'c_eplen':>7}  "
          f"{'b_vloss':>7}  {'c_vloss':>7}  {'b_gnb':>7}  {'c_gnb':>7}  {'b_fps':>7}  {'c_fps':>7}")
    for i in range(n):
        b = base[i] if i < len(base) else {}
        c = cfd[i] if i < len(cfd) else {}
        bl = b.get("ep_loss", float("nan"))
        cl = c.get("ep_loss", float("nan"))
        bl_s = "inf" if bl == float("inf") else f"{bl:.3f}"
        cl_s = "inf" if cl == float("inf") else f"{cl:.3f}"
        print(f"{b.get('iter', i+1):>4}  {bl_s:>9}  {cl_s:>9}  "
              f"{b.get('ep_len', float('nan')):>7.1f}  {c.get('ep_len', float('nan')):>7.1f}  "
              f"{b.get('value_loss', float('nan')):>7.2f}  {c.get('value_loss', float('nan')):>7.2f}  "
              f"{b.get('grad_norm_before', float('nan')):>7.0f}  "
              f"{c.get('grad_norm_before', float('nan')):>7.0f}  "
              f"{b.get('fps', float('nan')):>7.0f}  {c.get('fps', float('nan')):>7.0f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/cfd_baseline.log",
         sys.argv[2] if len(sys.argv) > 2 else "logs/cfd_cfd.log")
