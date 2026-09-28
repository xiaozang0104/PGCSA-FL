# -*- coding: utf-8 -*-
"""Build the Excel workbook that holds the data of the four complexity panels, laid out so
that each sheet can be charted in Excel directly (rows = x values, columns = methods).

    python make_fig_excel.py [bench_final.csv] [out.xlsx]

Sheets
    Fig3-computation-N   rows = number of clients, columns = methods, values in ms
    Fig3-computation-d   rows = update dimension,  columns = methods, values in ms
    Fig4-communication-N rows = number of clients, columns = methods, values in bytes
    Fig4-communication-d rows = update dimension,  columns = methods, values in bytes
    raw                  the long table, including the standard deviation (error bars)
    notes                provenance of every number
"""
import csv, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bench_final.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "fig_data_for_excel.xlsx")

ORDER = ["Ours", "FLTrust", "Dnc", "PFLF", "LSFL", "ESFL", "Split", "FoundationFL", "Qin"]
FLOAT4 = 4.0                      # float32 payload, bytes per value


def load(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("status") != "ok" or not r.get("comp_median_ms"):
            continue
        rows.append(dict(sweep=r["sweep"], method=r["method"], N=int(r["N"]), d=int(r["d"]),
                         ms=float(r["comp_median_ms"]), sd=float(r["comp_std_ms"] or 0.0),
                         impl=r.get("impl", "")))
    return rows


def pivot(rows, sweep, xkey, value):
    pts = sorted({r[xkey] for r in rows if r["sweep"] == sweep})
    methods = [m for m in ORDER if any(r["method"] == m and r["sweep"] == sweep for r in rows)]
    data = {}
    for m in methods:
        col = []
        for x in pts:
            hit = [r for r in rows if r["sweep"] == sweep and r["method"] == m and r[xkey] == x]
            col.append(round(value(hit[0]), 4) if hit else None)
        data[m] = col
    label = {"N": "Number of clients N", "d": "Update dimension d"}[xkey]
    return pd.DataFrame(data, index=pd.Index(pts, name=label))


def main():
    rows = load(SRC)
    ms = lambda r: r["ms"]
    byt = lambda r: r["N"] * (256 if r["method"] in ("Ours", "ESFL") else r["d"]) * FLOAT4
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        pivot(rows, "N", "N", ms).to_excel(xw, sheet_name="Fig3-computation-N")
        pivot(rows, "d", "d", ms).to_excel(xw, sheet_name="Fig3-computation-d")
        pivot(rows, "N", "N", byt).to_excel(xw, sheet_name="Fig4-communication-N")
        pivot(rows, "d", "d", byt).to_excel(xw, sheet_name="Fig4-communication-d")
        pd.DataFrame(rows).to_excel(xw, sheet_name="raw", index=False)
        pd.DataFrame({"notes": [
            "Source file: bench_final.csv (single run of bench_table_methods.py on the "
            "AutoDL cloud container, serial, one thread, median of 5 repetitions).",
            "Fig3 sheets: per-round SERVER-SIDE aggregation cost in ms. Scope: the "
            "aggregation rule itself (subspace estimation, anomaly scoring and "
            "aggregation for ours; the filtering and combination steps for each "
            "baseline). Local training, the global model update, the evaluation and the "
            "network are identical for all methods and are excluded.",
            "Ours = detect on the compressed updates, average the accepted ones, "
            "reconstruct that single average once (the pipeline of Step 5 / Table IV "
            "after the compressed-domain averaging).",
            "Fig4 sheets: uplink volume per round in BYTES, computed from the protocol "
            "(float32): N x 256 x 4 for ours and ESFL, N x d x 4 for the rest. The "
            "downlink broadcast is identical for all methods and is not counted. The "
            "full-dimensional methods therefore coincide exactly, as do ours and ESFL.",
            "Sweep 1: N = 20, 50, 100, 200 at d = 222410. Sweep 2: d = 100000, 222410, "
            "455000, 800000, 1000000 at N = 20.",
            "The 'raw' sheet keeps the standard deviation used for the error bars.",
        ]}).to_excel(xw, sheet_name="notes", index=False)
    print("written:", OUT)
    print("\nFig3-computation (ms) -- vs N\n", pivot(rows, "N", "N", ms).to_string())
    print("\nFig3-computation (ms) -- vs d\n", pivot(rows, "d", "d", ms).to_string())


if __name__ == "__main__":
    main()
