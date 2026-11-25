"""Metrics of whether AMOC slowdown exacerbates or ameliorates the 4xCO2 response.

Cases (``data_loader.CO2_AMOC_CASES``), with the short labels used in figures:

    c1a20 = 1xCO2        control: 1xCO2, AMOC ~20 Sv
    c4a20 = 4xCO2_m03Sv  4xCO2 with -0.3 Sv hosing holding AMOC near 20 Sv
    c4a04 = 4xCO2        4xCO2, AMOC slowed

From time means over a common window, per grid cell:

    C = c4a20 - c1a20     CO2 effect without AMOC slowdown
    A = c4a04 - c4a20     AMOC-slowdown effect
    T = A + C             total response, c4a04 - c1a20

and sigma, the interannual standard deviation of the c1a20 annual means
(``interannual_sd``: whole control, linearly detrended), as the scale of natural
variability. The metrics avoid the ratio A / C, which blows up where C ~ 0:

- ``effect_ratio``  R = A / C = T / C - 1, kept only where the CO2 effect is
  ``substantive`` (so its blow-up where C ~ 0 never shows). -1: AMOC slowdown
  cancels the CO2 effect, -0.5 halves it, 0 no effect, +1 doubles it, and R < -1:
  it overcompensates, leaving a change of the opposite sign to the CO2 change.
  Where C is not substantive but the total T is (``change_only_with_amoc``), R is
  undefined but AMOC slowdown makes a substantive change that CO2 alone does not.
- ``departure_change``  E = (|T| - |C|) / sigma: how far AMOC slowdown moves the
  4xCO2 climate farther from (E > 0, exacerbates) or closer to (E < 0,
  ameliorates) the c1a20 climate, in units of interannual SD. |E| <= |A| / sigma.
- ``component_fractions``  |A|, |C| and sigma as shares of sigma + |A| + |C|
  (summing to 1): which of AMOC slowdown, CO2 or natural variability dominates.
- ``classify``  categories with a substantiveness threshold of ``k`` sigma (see
  ``CLASS_NAMES``).
- ``land_summary``  land-wide numbers: the RMS ratio of A to C and the pattern
  congruence / correlation of A with C.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal

import data_loader as dl
import regression as reg

BASELINE_CASE, CO2_CASE, AMOC_CASE = dl.CO2_AMOC_CASES
CASE_LABELS = {BASELINE_CASE: "c1a20", CO2_CASE: "c4a20", AMOC_CASE: "c4a04"}

# Effect name -> (minuend, subtrahend) case.
EFFECTS = {
    "co2_effect": (CO2_CASE, BASELINE_CASE),
    "amoc_effect": (AMOC_CASE, CO2_CASE),
    "total": (AMOC_CASE, BASELINE_CASE),
}

SIGNIFICANCE_P = 0.05
DEFAULT_THRESHOLD_SD = 1.0

# Category codes (index into CLASS_NAMES). An effect is "substantive" where its
# magnitude is at least k sigma and it is significant (p <= SIGNIFICANCE_P).
#   0  AMOC effect not substantive
#   1  exacerbates: AMOC effect has the sign of a substantive CO2 effect
#   2  partly offsets: opposite sign, |A| < |C|
#   3  overcompensates: opposite sign, |A| >= |C| (the total has the opposite sign
#      to C, or is zero); these are the cells with R <= -1
#   4  AMOC-only: substantive AMOC effect where the CO2 effect is not substantive
CLASS_NAMES = ["AMOC effect not substantive", "exacerbates", "partly offsets", "overcompensates",
               "AMOC-only (CO₂ effect not substantive)"]


def interannual_sd(annual):
    """Standard deviation over ``year`` of a ``(year, lat, lon)`` field after
    removing each cell's linear trend (``scipy.signal.detrend``), so slow
    control drift does not count as variability."""
    detrended = xr.apply_ufunc(signal.detrend, annual, input_core_dims=[["year"]],
                               output_core_dims=[["year"]])
    return detrended.std("year", ddof=1)


def resolvable_variability(sd, control_annual, land):
    """True where ``sd`` exceeds the float32 resolution of the variable's typical
    land magnitude: float32 machine epsilon times the median over ``land`` of the
    absolute c1a20 time mean. Below it, sigma is zero up to rounding (a constant
    series, e.g. snow depth held at its cap, detrends to noise of ~1e-23) or traces
    of a field that is physically zero there (e.g. ~1e-21 m of snow), and no
    natural-variability scale can be measured."""
    typical = float(abs(control_annual.mean("year")).where(land).median())
    return sd > np.finfo(np.float32).eps * typical


def effects(means):
    """``EFFECTS`` differences of a ``(case, lat, lon)`` field of time means, as a
    Dataset with ``co2_effect``, ``amoc_effect`` and ``total``."""
    return xr.Dataset({name: means.sel(case=minuend, drop=True) - means.sel(case=subtrahend, drop=True)
                       for name, (minuend, subtrahend) in EFFECTS.items()})


def effect_ratio(amoc_effect, co2_effect):
    """R = A / C: the AMOC effect as a fraction of the CO2 effect (-1 = cancels it,
    < -1 = overcompensates, +1 = doubles it)."""
    return amoc_effect / co2_effect


def substantive(effect, sd, pvalue, threshold_sd):
    """True where |``effect``| >= ``threshold_sd`` * ``sd`` and ``pvalue`` <= ``SIGNIFICANCE_P``."""
    return (abs(effect) >= threshold_sd * sd) & (pvalue <= SIGNIFICANCE_P)


def departure_change(amoc_effect, co2_effect, sd):
    """E = (|A + C| - |C|) / sd: change in distance from the baseline climate caused
    by AMOC slowdown, in units of ``sd``. Positive exacerbates, negative ameliorates."""
    return (abs(amoc_effect + co2_effect) - abs(co2_effect)) / sd


def component_fractions(amoc_effect, co2_effect, scale):
    """|A|, |C| and ``scale`` as fractions of ``scale`` + |A| + |C| (they sum to 1)."""
    total = scale + abs(amoc_effect) + abs(co2_effect)
    return xr.Dataset({"amoc_fraction": abs(amoc_effect) / total,
                       "co2_fraction": abs(co2_effect) / total,
                       "scale_fraction": scale / total})


# Fraction variables of ``component_fractions``, in drawing (stacking) order.
FRACTION_NAMES = ["amoc_fraction", "co2_fraction", "scale_fraction"]


def scaled_metrics(amoc_effect, co2_effect, scale):
    """The metrics that depend on the scale S (``scale``): C / S, A / S, T / S,
    E = ``departure_change`` and the ``component_fractions``."""
    return xr.Dataset({
        "co2_scaled": co2_effect / scale,
        "amoc_scaled": amoc_effect / scale,
        "total_scaled": (amoc_effect + co2_effect) / scale,
        "departure_change": departure_change(amoc_effect, co2_effect, scale),
        **component_fractions(amoc_effect, co2_effect, scale),
    })


def classify(amoc_effect, co2_effect, sd, amoc_pvalue, co2_pvalue, threshold_sd):
    """Integer category (index into ``CLASS_NAMES``) per cell; see ``CLASS_NAMES``.
    Effects are tested with ``substantive``."""
    amoc_substantive = substantive(amoc_effect, sd, amoc_pvalue, threshold_sd)
    co2_substantive = substantive(co2_effect, sd, co2_pvalue, threshold_sd)
    same_sign = np.sign(amoc_effect) == np.sign(co2_effect)
    codes = np.select(
        [~amoc_substantive.values, ~co2_substantive.values, same_sign.values,
         (abs(amoc_effect) < abs(co2_effect)).values],
        [0, 4, 1, 2], default=3)
    return xr.DataArray(codes, coords=amoc_effect.coords, dims=amoc_effect.dims, name="category")


def land_fraction(condition, land):
    """Fraction of the (area-weighted) land area where the boolean ``(lat, lon)``
    ``condition`` holds."""
    weights = dl.latitude_band_weights(land["lat"]) * land
    return float(weights.where(condition).sum() / weights.sum())


def land_area_fractions(categories, land):
    """Fraction of the (area-weighted) land area in each category, as a Series
    indexed by ``CLASS_NAMES``."""
    return pd.Series([land_fraction(categories == code, land)
                      for code in range(len(CLASS_NAMES))], index=CLASS_NAMES)


def land_summary(amoc_effect, co2_effect):
    """Land-wide (area-weighted, over the non-NaN cells) comparison of the AMOC
    effect A with the CO2 effect C, as a dict:

    - ``rms_ratio``: RMS(A) / RMS(C), the relative size of the two effects without
      averaging local ratios (and so without their blow-up where C ~ 0);
    - ``pattern_congruence``: sum(w A C) / sqrt(sum(w A^2) sum(w C^2)), the uncentered
      pattern correlation: -1 where A everywhere opposes C in proportion (AMOC
      slowdown counters the CO2 pattern), +1 where it reinforces it;
    - ``pattern_correlation``: the centered (Pearson) pattern correlation, comparing
      only the spatial variations about each field's land mean.
    """
    weights = dl.latitude_band_weights(amoc_effect["lat"])

    def mean_square(field):
        return float((field ** 2).weighted(weights).mean())

    return {
        "rms_ratio": np.sqrt(mean_square(amoc_effect) / mean_square(co2_effect)),
        "pattern_congruence": float((amoc_effect * co2_effect).weighted(weights).mean())
                              / np.sqrt(mean_square(amoc_effect) * mean_square(co2_effect)),
        "pattern_correlation": float(xr.corr(amoc_effect, co2_effect, weights=weights)),
    }


# Columns of the per-variable summary (``variable_summary``), in order, with their
# definitions ("substantive": |effect| >= k sigma and p <= SIGNIFICANCE_P).
SUMMARY_COLUMNS = {
    "description": "the variable's long name and units",
    CLASS_NAMES[0]: "share of the land area where A is not substantive",
    CLASS_NAMES[1]: "share of the land area where A and C are substantive and A has the sign of C",
    CLASS_NAMES[2]: "share of the land area where A and C are substantive, of opposite sign, "
                    "and |A| < |C|",
    CLASS_NAMES[3]: "share of the land area where A and C are substantive, of opposite sign, "
                    "and |A| ≥ |C|",
    CLASS_NAMES[4]: "share of the land area where A is substantive but C is not",
    "land_fraction_R_below_minus1": "share of the land area where C is substantive and "
                                    "R = A / C < −1 (gold on the R map)",
    "land_fraction_change_only_with_amoc": "share of the land area where C is not substantive "
                                           "but T is (orange on the R map)",
    "land_median_R": "area-weighted median of R = A / C over land where C is substantive",
    "land_mean_E_in_sd": "area-weighted land mean of E = (|T| − |C|) / σ, in σ",
    "rms_ratio": "RMS(A) / RMS(C), area-weighted over land",
    "pattern_congruence": "uncentered pattern correlation of A with C over land, "
                          "Σ w·A·C / √(Σ w·A² · Σ w·C²)",
    "pattern_correlation": "centered (Pearson) area-weighted pattern correlation of A with C "
                           "over land",
}


def variable_summary(metrics, summary):
    """One row of the summary table (``SUMMARY_COLUMNS``) for a ``land_metrics``
    result and its ``land_summary``."""
    land = metrics["land"]
    ratio = metrics["effect_ratio"]
    return {
        "description": f"{metrics.attrs['long_name']} ({metrics.attrs['units']})",
        **land_area_fractions(metrics["category"], land),
        "land_fraction_R_below_minus1": land_fraction(ratio < -1, land),
        "land_fraction_change_only_with_amoc": land_fraction(metrics["change_only_with_amoc"] == 1,
                                                             land),
        "land_median_R": float(dl.area_weighted_median(ratio)),
        "land_mean_E_in_sd": float(dl.global_mean(metrics["departure_change"])),
        **summary,
    }


def land_metrics(fields, control_annual, threshold_sd):
    """All metrics for one variable on land, as a ``(lat, lon)`` Dataset with ocean
    cells set to NaN (``data_loader.land_mask``). Land cells where the variable has no
    resolvable variability in c1a20 (``resolvable_variability``: e.g. low cloud over
    the high Antarctic and Tibetan plateaus, which lie above the low-cloud levels, or
    snow depth held at its cap on ice sheets) are missing data too, as nothing can be
    measured against a zero scale. The mask of the remaining cells is
    kept as ``land``; land-area shares are of these cells.

    ``effect_ratio`` is NaN where C is not substantive; ``change_only_with_amoc``
    (1 / 0, NaN on ocean) marks the cells where C is not substantive but T is.

    ``fields`` is a ``data_loader.case_annual_fields`` result for ``CO2_AMOC_CASES``
    (the averaging window); ``control_annual`` the c1a20 ``(year, lat, lon)`` annual
    means from which sigma is estimated (``interannual_sd``). Effect p-values are
    Welch t-tests on 10-year means (``regression.difference_pvalue``).
    """
    runs = {case: fields.sel(case=case, drop=True) for case in dl.CO2_AMOC_CASES}
    differences = effects(fields.mean("year"))
    pvalues = {f"{name}_pvalue": reg.difference_pvalue(runs[minuend], runs[subtrahend],
                                                        reg.DECADAL_BLOCK)
               for name, (minuend, subtrahend) in EFFECTS.items()}
    sd = interannual_sd(control_annual)
    land_mask = dl.land_mask(fields["lat"], fields["lon"])
    land = land_mask & resolvable_variability(sd, control_annual, land_mask)
    amoc_effect, co2_effect = differences["amoc_effect"], differences["co2_effect"]
    co2_substantive = substantive(co2_effect, sd, pvalues["co2_effect_pvalue"], threshold_sd)
    total_substantive = substantive(differences["total"], sd, pvalues["total_pvalue"], threshold_sd)
    metrics = xr.Dataset({
        **differences, **pvalues, "sd": sd,
        "effect_ratio": effect_ratio(amoc_effect, co2_effect).where(co2_substantive),
        "change_only_with_amoc": ~co2_substantive & total_substantive,
        **scaled_metrics(amoc_effect, co2_effect, sd),
        "category": classify(amoc_effect, co2_effect, sd, pvalues["amoc_effect_pvalue"],
                             pvalues["co2_effect_pvalue"], threshold_sd),
    }).where(land).assign(land=land)
    years = control_annual["year"].values
    metrics.attrs = {**fields.attrs, "name": fields.name, "threshold_sd": threshold_sd,
                     "sd_years": f"{years.min()}-{years.max()}"}
    return metrics
