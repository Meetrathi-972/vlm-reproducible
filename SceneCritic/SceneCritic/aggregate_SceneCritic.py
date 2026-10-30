"""
Aggregate SceneCritic TSV summaries into a LaTeX table.

"""

import os
import sys
import csv
import re
from collections import defaultdict
from typing import Dict, List, Optional, Iterable


REQUIRED_COLS = {
    "group",
    "method",
    "sp_weighted",
    "orientation_pct",
    "overlap_pct",
    "status",
}


def collect_tsv_files(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(".tsv"):
            out.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(".tsv"):
                        out.append(os.path.join(root, f))
    return sorted(set(out))


def sstrip(x) -> str:
    if x is None:
        return ""
    return x.strip() if isinstance(x, str) else str(x)


def to_float(x) -> Optional[float]:
    s = sstrip(x)
    if s == "" or s.lower() in {"nan", "none", "null", "na"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_rows_tab(tsv_path: str) -> Optional[Iterable[Dict[str, str]]]:
    f = open(tsv_path, "r", newline="")
    reader = csv.DictReader(f, delimiter="\t")

    if reader.fieldnames is None:
        f.close()
        raise ValueError("No header found")

    if len(reader.fieldnames) <= 1:
        f.close()
        return None

    fieldnames = [sstrip(h) for h in reader.fieldnames]
    missing = sorted(REQUIRED_COLS - set(fieldnames))
    if missing:
        f.close()
        raise ValueError(f"Missing required columns (tab mode): {missing}. Found={fieldnames}")

    def gen():
        try:
            for row in reader:
                yield {sstrip(k): sstrip(v) for k, v in row.items()}
        finally:
            f.close()

    return gen()


def _read_rows_whitespace(tsv_path: str) -> Iterable[Dict[str, str]]:
    with open(tsv_path, "r", newline="") as f:
        header = f.readline().strip()
        if not header:
            raise ValueError("Empty file / missing header")

        cols = re.split(r"\s+", header)
        missing = sorted(REQUIRED_COLS - set(cols))
        if missing:
            raise ValueError(f"Missing required columns (whitespace mode): {missing}. Header={cols}")

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < len(cols):
                continue
            row = dict(zip(cols, parts[:len(cols)]))
            yield {k: sstrip(v) for k, v in row.items()}


def read_rows(tsv_path: str) -> Iterable[Dict[str, str]]:
    tab_rows = _read_rows_tab(tsv_path)
    if tab_rows is not None:
        return tab_rows
    return _read_rows_whitespace(tsv_path)


def aggregate(tsv_files: List[str]) -> Dict[str, Dict[str, List[float]]]:
    agg = defaultdict(lambda: defaultdict(list))

    for tsv in tsv_files:
        try:
            for row in read_rows(tsv):
                status = sstrip(row.get("status")).lower()
                if status != "ok":
                    continue

                group = sstrip(row.get("group"))
                method = sstrip(row.get("method"))
                if not group or not method:
                    continue

                method_name = f"{group}_{method}"

                sem = to_float(row.get("sp_weighted"))
                ori = to_float(row.get("orientation_pct"))
                ovp = to_float(row.get("overlap_pct"))
                ovp_obb = to_float(row.get("overlap_obb_pct"))

                if sem is not None:
                    agg[method_name]["sem"].append(sem)
                if ori is not None:
                    agg[method_name]["ori"].append(ori)
                if ovp is not None:
                    agg[method_name]["overlap"].append(ovp)
                if ovp_obb is not None:
                    agg[method_name]["overlap_obb"].append(ovp_obb)

                agg[method_name]["n"].append(1)

        except Exception as e:
            print(f"Warning: skipping {tsv}: {e}", file=sys.stderr)

    return agg


def safe_mean(vals: List[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def compute_means(agg: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for m, d in agg.items():
        n = len(d.get("n", []))

        sem         = safe_mean(d.get("sem", []))
        ori         = safe_mean(d.get("ori", []))
        overlap     = safe_mean(d.get("overlap", []))
        overlap_obb = safe_mean(d.get("overlap_obb", []))

        avg3 = (sem * 100 + ori * 100 + overlap * 100) / 3.0
        
        avg3_obb = (sem * 100 + ori * 100 + overlap_obb * 100) / 3.0

        results[m] = {
            "sem": sem,
            "ori": ori,
            "overlap": overlap,
            "overlap_obb": overlap_obb,
            "avg3": avg3,
            "avg3_obb": avg3_obb,
            "n": float(n),
        }
    return results


def make_latex_table(results: Dict[str, Dict[str, float]]) -> str:
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Semantic verifier scores averaged over all rooms.}")
    lines.append(r"\label{tab:semantic_verifier_scores}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & Sem. (weighted) & Orientation (\%) "
        r"& Overlap\textsubscript{AABB} (\%) & Overlap\textsubscript{OBB} (\%) "
        r"& Avg.\textsubscript{AABB} (\%) & Avg.\textsubscript{OBB} (\%) \\"
    )
    lines.append(r"\midrule")

    for method in sorted(results.keys()):
        r = results[method]
        name = method.replace("_", r"\_")
        lines.append(
            f"{name} "
            f"& {r['sem']:.3f} "
            f"& {r['ori'] * 100:.1f} "
            f"& {r['overlap'] * 100:.1f} "
            f"& {r['overlap_obb'] * 100:.1f} "
            f"& {r['avg3']:.1f} "
            f"& {r['avg3_obb']:.1f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python aggregate_solver.py <tsv_or_dir1> [tsv_or_dir2] ...", file=sys.stderr)
        return 2

    tsv_files = collect_tsv_files(argv[1:])
    if not tsv_files:
        print("Error: no .tsv files found in given paths.", file=sys.stderr)
        return 2

    print(f"Found {len(tsv_files)} TSV files.", file=sys.stderr)

    agg = aggregate(tsv_files)
    results = compute_means(agg)

    if not results:
        print("Warning: no OK rows parsed (results empty). Check delimiter / status values.", file=sys.stderr)

    for m in sorted(results.keys()):
        r = results[m]
        print(
            f"  {m}: sem={r['sem']:.3f}, ori={r['ori'] * 100:.1f}%, "
            f"overlap_aabb={r['overlap'] * 100:.1f}%, "
            f"overlap_obb={r['overlap_obb'] * 100:.1f}%, "
            f"avg_aabb={r['avg3']:.1f}%, avg_obb={r['avg3_obb']:.1f}% "
            f"(n={int(r['n'])})",
            file=sys.stderr,
        )

    print(make_latex_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))