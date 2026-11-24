"""Land maps of whether AMOC slowdown exacerbates or ameliorates the 4xCO2 response.

Cases (``co2_amoc_metrics``): c1a20 = 1xCO2, c4a20 = 4xCO2_m03Sv (AMOC held near
20 Sv), c4a04 = 4xCO2 (AMOC slowed). From 2101-2150 means (the last 50 years common
to the runs), per land cell: C = c4a20 - c1a20 (CO2 effect), A = c4a04 - c4a20
(AMOC-slowdown effect), T = A + C, and sigma = interannual SD of the c1a20 annual
means over the whole control (1850-2150, linearly detrended). Ocean is missing data.

One page per variable (``output.plot_co2_amoc_metrics``): C / sigma, A / sigma,
sigma; the three-part map of |A|, |C| and sigma as shares of sigma + |A| + |C|;
E = (|T| - |C|) / sigma (> 0 exacerbates, < 0 ameliorates); the bounded share
f = A sgn(C) / (|A| + |C|); categories (not substantive / exacerbates / partly
offsets / reverses / AMOC-only), an effect counting as substantive where it is at
least ``--threshold-sd`` sigma and significant (Welch t-test on 10-year means,
p <= 0.05); and the three-part map's key with land-wide numbers
(``co2_amoc_metrics.land_summary``: RMS(A) / RMS(C), pattern congruence and
correlation of A with C). Water-flux variables (mm/day) get a second page
(``output.plot_co2_amoc_scaled_metrics``) with the c1a20 gross water flux
S = P0 + E0 (``data_loader.gross_water_flux``, same years) in place of sigma.

Outputs in ``data/output/co2_amoc_metrics/``: ``<var>_2101-2150_<k>sd.pdf`` (and
``<var>_gross_flux_2101-2150_<k>sd.pdf``), merged into ``book_2101-2150_<k>sd.pdf``,
and ``land_summary_2101-2150_<k>sd.csv``, one row per variable: each category's
share of the area-weighted land area, the land-mean E, and the land-wide numbers. ``--variables`` takes set names from ``data_loader.VARIABLE_SETS``
and/or variable names (default ``all``).

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
from output import plot_co2_amoc_metrics, plot_co2_amoc_scaled_metrics

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
    variables = dl.resolve_variables(args.variables)
    tag = f"{FIRST_YEAR}-{LAST_YEAR}_{args.threshold_sd:g}sd"

    os.makedirs(OUT_DIR, exist_ok=True)
    gross_flux = dl.gross_water_flux(cam.BASELINE_CASE, FIRST_YEAR, LAST_YEAR).load()
    gross_flux_label = (f"P₀ + E₀, gross water flux of {cam.CASE_LABELS[cam.BASELINE_CASE]} "
                        f"({FIRST_YEAR}-{LAST_YEAR} mean)")
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
        rows[var] = {**cam.land_area_fractions(metrics["category"], metrics["land"]),
                     "land_mean_E_in_sd": float(dl.global_mean(metrics["departure_change"])),
                     **summary}
        if fields.attrs["units"] == dl.WATER_FLUX_UNITS:
            scaled = cam.land_scaled_metrics(metrics, gross_flux, gross_flux_label)
            pvalues = {"co2": metrics["co2_effect_pvalue"], "amoc": metrics["amoc_effect_pvalue"]}
            out_path = os.path.join(OUT_DIR, f"{var}_gross_flux_{tag}.pdf")
            plot_co2_amoc_scaled_metrics(scaled, pvalues, summary, out_path)
            page_paths.append(out_path)
            print(f"wrote {out_path}")

    table_path = os.path.join(OUT_DIR, f"land_summary_{tag}.csv")
    pd.DataFrame.from_dict(rows, orient="index").rename_axis("variable").to_csv(
        table_path, float_format="%.4f")
    print(f"wrote {table_path}")

    book_path = os.path.join(OUT_DIR, f"book_{tag}.pdf")
    writer = PdfWriter()
    for path in page_paths:
        writer.append(path)
    writer.write(book_path)
    print(f"wrote {book_path}  ({len(page_paths)} pages)")


if __name__ == "__main__":
    main()
