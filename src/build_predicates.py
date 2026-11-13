"""Run the parser over a batch of reports -> predicate table (Parquet) + a
human-readable dump for hand-validation.

Prototype usage (Week-2 plan: prototype on ~200, hand-check ~10 before full run):
    python src/build_predicates.py --limit 200 --dump 10
Full run later:
    python src/build_predicates.py            # all 48,976
"""
from __future__ import annotations

import argparse
import collections
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import DATA_DIR, LABELS_CSV, REPORTS_DIR, REPORTS_ZIP
from parser import extract_predicates, iter_reports

CHUNK = 2000  # reports per Parquet flush (keeps memory flat on the full run)
SCHEMA = pa.schema([
    ("sha", pa.string()), ("family", pa.dictionary(pa.int16(), pa.string())),
    ("field", pa.dictionary(pa.int8(), pa.string())),
    ("klass", pa.dictionary(pa.int8(), pa.string())),
    ("ptype", pa.dictionary(pa.int16(), pa.string())),
    ("value", pa.string()),
])


def load_fam_map() -> dict:
    df = pd.read_csv(LABELS_CSV)
    return dict(zip(df["sha256"], df["classification_family"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="report count (default: all)")
    ap.add_argument("--dump", type=int, default=10, help="reports to dump for hand-check")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    fam = load_fam_map()

    out = DATA_DIR / ("predicates.parquet" if args.limit is None
                      else f"predicates_sample{args.limit}.parquet")
    writer = pq.ParquetWriter(out, SCHEMA, compression="zstd")

    buf: list[dict] = []
    dump_lines: list[str] = []
    field_ctr: collections.Counter = collections.Counter()
    klass_ctr: collections.Counter = collections.Counter()
    n_pred_min, n_pred_max, n_pred_tot = 10**9, 0, 0
    total = 0
    n = 0
    t0 = time.time()

    def flush():
        if not buf:
            return
        writer.write_table(pa.Table.from_pylist(buf, schema=SCHEMA))
        buf.clear()

    for sha, report in iter_reports(REPORTS_ZIP, limit=args.limit):
        preds = extract_predicates(report)
        family = fam.get(sha, "?")
        n_pred_min = min(n_pred_min, len(preds))
        n_pred_max = max(n_pred_max, len(preds))
        n_pred_tot += len(preds)
        for p in preds:
            buf.append({**p.as_row(sha), "family": family})
            field_ctr[p.field] += 1
            klass_ctr[p.klass] += 1
        if n < args.dump:
            dump_lines.append(_dump_one(sha, family, preds))
        n += 1
        total += len(preds)
        if n % CHUNK == 0:
            flush()
            rate = n / (time.time() - t0)
            print(f"  parsed {n:,} reports  ({total:,} preds, {rate:.0f} rep/s)")
    flush()
    writer.close()

    dump_path = REPORTS_DIR / "predicate_dump.txt"
    dump_path.write_text("\n".join(dump_lines))

    # --- summary ---
    print(f"\nreports parsed : {n:,}   in {time.time()-t0:.0f}s")
    print(f"predicates     : {total:,}   ({n_pred_tot/max(1,n):.0f} avg/report, "
          f"min {n_pred_min}, max {n_pred_max})")
    print("\nby field:")
    for k, v in field_ctr.most_common():
        print(f"  {k:10} {v:>10,}")
    print("\nby klass:")
    for k, v in klass_ctr.most_common():
        print(f"  {k:10} {v:>10,}  ({100*v/max(1,total):.1f}%)")
    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.0f} MB)")
    print(f"wrote {dump_path}  (hand-check the first {args.dump} reports)")


def _dump_one(sha: str, family: str, preds) -> str:
    lines = [f"{'='*72}", f"{family:12} {sha[:20]}   ({len(preds)} predicates)"]
    by_field: dict = collections.defaultdict(list)
    for p in preds:
        by_field[p.field].append(p)
    for field in ["mutex", "registry", "file", "network", "command", "process",
                  "api_set", "api_seq"]:
        items = by_field.get(field, [])
        if not items:
            continue
        # api_seq is huge — show a sample
        show = items if field != "api_seq" else items[:8]
        lines.append(f"  [{field}]  n={len(items)}")
        for p in show:
            lines.append(f"      {p.klass:9} {p.ptype:16} {p.value[:90]}")
        if field == "api_seq" and len(items) > 8:
            lines.append(f"      ... (+{len(items)-8} more api_{'3'}grams)")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
