# prof_analyze.py
import os
import pstats
from pathlib import Path

PROFILE = "prof.out"
TOP_N = 80

ROOT = str(Path.cwd())

def short_path(filename: str) -> str:
    if filename.startswith(ROOT):
        return os.path.relpath(filename, ROOT)
    home = str(Path.home())
    if filename.startswith(home):
        return "~/" + os.path.relpath(filename, home)
    return filename

def classify(filename: str) -> str:
    if filename.startswith(ROOT):
        return "user"
    if filename.startswith("<") and filename.endswith(">"):
        return "builtin"
    if "python3." in filename or "/lib/python" in filename:
        return "stdlib"
    return "other"

def load_rows(profile_path: str):
    stats = pstats.Stats(profile_path)
    rows = []
    for (filename, lineno, funcname), stat in stats.stats.items():
        cc, nc, tt, ct, callers = stat
        rows.append({
            "filename": filename,
            "lineno": lineno,
            "funcname": funcname,
            "ncalls": nc,
            "pcalls": cc,
            "tottime": tt,
            "cumtime": ct,
            "percall_tottime": tt / nc if nc else 0.0,
            "percall_cumtime": ct / nc if nc else 0.0,
            "kind": classify(filename),
            "label": f"{short_path(filename)}:{lineno}({funcname})",
        })
    return rows

def fmt_num(x):
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:,.3f}"

def print_table(rows, sort_key: str, title: str, limit: int = TOP_N, only_kind: str | None = None):
    if only_kind is not None:
        rows = [r for r in rows if r["kind"] == only_kind]
    rows = sorted(rows, key=lambda r: r[sort_key], reverse=True)[:limit]

    print(f"\n{title}")
    print("=" * len(title))

    headers = ["rank", "kind", "ncalls", "tottime", "cumtime", "tottime/call", "cumtime/call", "function"]
    widths = [5, 8, 12, 10, 10, 14, 14, 0]

    print(
        f"{headers[0]:>{widths[0]}}  "
        f"{headers[1]:<{widths[1]}}  "
        f"{headers[2]:>{widths[2]}}  "
        f"{headers[3]:>{widths[3]}}  "
        f"{headers[4]:>{widths[4]}}  "
        f"{headers[5]:>{widths[5]}}  "
        f"{headers[6]:>{widths[6]}}  "
        f"{headers[7]}"
    )
    print("-" * 140)

    for i, r in enumerate(rows, 1):
        print(
            f"{i:>{widths[0]}}  "
            f"{r['kind']:<{widths[1]}}  "
            f"{fmt_num(r['ncalls']):>{widths[2]}}  "
            f"{fmt_num(r['tottime']):>{widths[3]}}  "
            f"{fmt_num(r['cumtime']):>{widths[4]}}  "
            f"{fmt_num(r['percall_tottime']):>{widths[5]}}  "
            f"{fmt_num(r['percall_cumtime']):>{widths[6]}}  "
            f"{r['label']}"
        )

def print_summary(rows):
    total_tt = sum(r["tottime"] for r in rows)
    total_ct = max((r["cumtime"] for r in rows), default=0.0)
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], 0.0)
        by_kind[r["kind"]] += r["tottime"]

    print("Profile summary")
    print("===============")
    print(f"Total tottime across all functions: {total_tt:,.3f}s")
    print(f"Max cumtime observed:              {total_ct:,.3f}s")
    print("Exclusive time by kind:")
    for kind, val in sorted(by_kind.items(), key=lambda kv: kv[1], reverse=True):
        pct = (100.0 * val / total_tt) if total_tt else 0.0
        print(f"  {kind:<8} {val:>10.3f}s   {pct:>6.2f}%")

if __name__ == "__main__":
    rows = load_rows(PROFILE)

    print_summary(rows)
    print_table(rows, "tottime", "Top functions by exclusive time (tottime)", limit=60)
    print_table(rows, "cumtime", "Top functions by inclusive time (cumtime)", limit=60)
    print_table(rows, "tottime", "Top USER functions by exclusive time (tottime)", limit=60, only_kind="user")
    print_table(rows, "cumtime", "Top USER functions by inclusive time (cumtime)", limit=60, only_kind="user")