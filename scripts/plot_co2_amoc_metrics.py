"""Land maps of whether AMOC slowdown exacerbates or ameliorates the 4xCO2 response.

Cases (``co2_amoc_metrics``): c1a20 = 1xCO2, c4a20 = 4xCO2_m03Sv (AMOC held near
20 Sv), c4a04 = 4xCO2 (AMOC slowed). From 2101-2150 means (the last 50 years common
to the runs), per land cell: C = c4a20 - c1a20 (CO2 effect), A = c4a04 - c4a20
(AMOC-slowdown effect), T = A + C, and sigma = interannual SD of the c1a20 annual
means over the whole control (1850-2150, linearly detrended). Ocean is missing data.

One page per variable (``output.plot_co2_amoc_metrics``), in rows: C, A, T;
C / sigma, A / sigma, T / sigma; and
E = (|T| - |C|) / sigma (> 0 exacerbates, < 0 ameliorates), the ratio R = A / C
(-1: AMOC slowdown cancels the CO2 effect, < -1: overcompensates, +1: doubles it;
only where the CO2 effect is substantive, and marked where it is not but the total
T is), and the three-part map of |A|, |C| and sigma as shares of sigma + |A| + |C|
with its key triangle and land-wide numbers (``co2_amoc_metrics.land_summary``:
RMS(A) / RMS(C), pattern congruence and correlation of A with C). An effect is
substantive where it is at least ``--threshold-sd`` sigma and significant (Welch
t-test on 10-year means, p <= 0.05).
The categories (not substantive / exacerbates / partly offsets / overcompensates /
AMOC-only) are tabulated in the CSV, not mapped.

Outputs in ``data/output/co2_amoc_metrics/``: ``<var>_2101-2150_<k>sd.pdf``, merged into ``book_2101-2150_<k>sd.pdf``
after a guide page explaining the metrics (``guide_2101-2150_<k>sd.pdf``) and
before a summary table page (``land_summary_2101-2150_<k>sd.pdf``); the same table
as ``land_summary_2101-2150_<k>sd.csv`` and ``.xlsx``, one row per variable
(``co2_amoc_metrics.SUMMARY_COLUMNS``): each category's share of the area-weighted
land area, the shares of land with R < -1 (overcompensation) and where only T is
substantive (a change only with AMOC slowdown), the median R, the land-mean E,
and the land-wide numbers.
``--variables`` takes set names from ``data_loader.VARIABLE_SETS`` and/or variable
names (default ``all``); prescribed variables (``data_loader.PRESCRIBED_VARIABLES``,
e.g. SOLIN), identical in every run and so with no CO2 or AMOC effect, are skipped.

    python scripts/plot_co2_amoc_metrics.py [--variables minimal] [--threshold-sd 1]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from pypdf import PdfWriter

import co2_amoc_metrics as cam
import data_loader as dl
from output import (plot_co2_amoc_metrics, plot_co2_amoc_metrics_guide,
                    plot_co2_amoc_summary_table,
                    write_co2_amoc_summary_xlsx)

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "co2_amoc_metrics")
FIRST_YEAR, LAST_YEAR = 2101, 2150


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    dl.add_variables_argument(parser)
    parser.add_argument("--threshold-sd", type=float, default=cam.DEFAULT_THRESHOLD_SD,
                        help="an effect is substantive where |effect| >= this many "
                             "interannual SDs of c1a20 (and significant); default "
                             f"{cam.DEFAULT_THRESHOLD_SD:g}")
    args = parser.parse_args()
    variables = [var for var in dl.resolve_variables(args.variables)
                 if var not in dl.PRESCRIBED_VARIABLES]
    tag = f"{FIRST_YEAR}-{LAST_YEAR}_{args.threshold_sd:g}sd"

    os.makedirs(OUT_DIR, exist_ok=True)
    page_paths = []
    rows = {}
    for var in variables:
        fields = dl.case_annual_fields(var, dl.CO2_AMOC_CASES, FIRST_YEAR, LAST_YEAR)
        control_annual = dl.load_annual_field(cam.BASELINE_CASE, var).load()
        metrics = cam.land_metrics(fields, control_annual, args.threshold_sd)
        summary = cam.land_summary(metrics["amoc_effect"], metrics["co2_effect"])
        out_path = os.path.join(OUT_DIR, f"{var}_{tag}.pdf")
        plot_co2_amoc_metrics(metrics, summary, out_path)
        page_paths.append(out_path)
        print(f"wrote {out_path}")
        rows[var] = cam.variable_summary(metrics, summary)

    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("variable")
    time_mean = metrics.attrs["time_mean"]
    table_path = os.path.join(OUT_DIR, f"land_summary_{tag}.csv")
    table.to_csv(table_path, float_format="%.4f")
    xlsx_path = os.path.join(OUT_DIR, f"land_summary_{tag}.xlsx")
    write_co2_amoc_summary_xlsx(table, time_mean, args.threshold_sd, xlsx_path)
    summary_page_path = os.path.join(OUT_DIR, f"land_summary_{tag}.pdf")
    plot_co2_amoc_summary_table(table, time_mean, args.threshold_sd, summary_page_path)
    for path in (table_path, xlsx_path, summary_page_path):
        print(f"wrote {path}")

    guide_path = os.path.join(OUT_DIR, f"guide_{tag}.pdf")
    plot_co2_amoc_metrics_guide(time_mean, metrics.attrs["sd_years"],
                                args.threshold_sd, variables, guide_path)
    print(f"wrote {guide_path}")

    book_path = os.path.join(OUT_DIR, f"book_{tag}.pdf")
    writer = PdfWriter()
    book_pages = [guide_path, *page_paths, summary_page_path]
    for path in book_pages:
        writer.append(path)
    writer.write(book_path)
    print(f"wrote {book_path}  ({len(book_pages)} pages)")


if __name__ == "__main__":
    main()
