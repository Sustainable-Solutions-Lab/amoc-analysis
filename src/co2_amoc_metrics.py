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

- ``bounded_share``  f = A sgn(C) / (|A| + |C|) = R / (1 + |R|), R = A / C. In
  [-1, 1]: 0 no AMOC effect, +0.5 AMOC doubles the CO2 change, -0.5 cancels it,
  < -0.5 reverses its sign, +-1 AMOC dominates where C ~ 0.
- ``departure_change``  E = (|T| - |C|) / sigma: how far AMOC slowdown moves the
  4xCO2 climate farther from (E > 0, exacerbates) or closer to (E < 0,
  ameliorates) the c1a20 climate, in units of interannual SD. |E| <= |A| / sigma.
- ``component_fractions``  |A|, |C| and the scale S as shares of S + |A| + |C|
  (summing to 1): which of AMOC slowdown, CO2 or natural variability dominates.
- ``classify``  categories with a substantiveness threshold of ``k`` sigma (see
  ``CLASS_NAMES``).
- ``land_summary``  land-wide numbers: the RMS ratio of A to C and the pattern
  congruence / correlation of A with C.

E and the fractions depend on the scale S: sigma for every variable
(``land_metrics``), and for water fluxes also the gross water flux of the
c1a20 base state, P0 + E0 (``land_scaled_metrics``).
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
#   3  reverses: opposite sign, |A| >= |C| (the total has the opposite sign to C)
#   4  AMOC-only: substantive AMOC effect where the CO2 effect is not substantive
CLASS_NAMES = ["AMOC effect not substantive", "exacerbates", "partly offsets", "reverses",
               "AMOC-only (CO₂ effect not substantive)"]


def interannual_sd(annual):
    """Standard deviation over ``year`` of a ``(year, lat, lon)`` field after
    removing each cell's linear trend (``scipy.signal.detrend``), so slow
    control drift does not count as variability."""
    detrended = xr.apply_ufunc(signal.detrend, annual, input_core_dims=[["year"]],
                               output_core_dims=[["year"]])
    return detrended.std("year", ddof=1)


def effects(means):
    """``EFFECTS`` differences of a ``(case, lat, lon)`` field of time means, as a
    Dataset with ``co2_effect``, ``amoc_effect`` and ``total``."""
    return xr.Dataset({name: means.sel(case=minuend, drop=True) - means.sel(case=subtrahend, drop=True)
                       for name, (minuend, subtrahend) in EFFECTS.items()})


def bounded_share(amoc_effect, co2_effect):
    """f = A sgn(C) / (|A| + |C|), the ratio R = A / C mapped to [-1, 1] as R / (1 + |R|)."""
    return amoc_effect * np.sign(co2_effect) / (abs(amoc_effect) + abs(co2_effect))


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
    """The metrics that depend on the scale S (``scale``): C / S, A / S,
    E = ``departure_change`` and the ``component_fractions``."""
    return xr.Dataset({
        "co2_scaled": co2_effect / scale,
        "amoc_scaled": amoc_effect / scale,
        "departure_change": departure_change(amoc_effect, co2_effect, scale),
        **component_fractions(amoc_effect, co2_effect, scale),
    })


def classify(amoc_effect, co2_effect, sd, amoc_pvalue, co2_pvalue, threshold_sd):
    """Integer category (index into ``CLASS_NAMES``) per cell; see ``CLASS_NAMES``.
    An effect is substantive where |effect| >= ``threshold_sd`` * ``sd`` and its
    p-value is <= ``SIGNIFICANCE_P``."""
    amoc_substantive = (abs(amoc_effect) >= threshold_sd * sd) & (amoc_pvalue <= SIGNIFICANCE_P)
    co2_substantive = (abs(co2_effect) >= threshold_sd * sd) & (co2_pvalue <= SIGNIFICANCE_P)
    same_sign = np.sign(amoc_effect) == np.sign(co2_effect)
    codes = np.select(
        [~amoc_substantive.values, ~co2_substantive.values, same_sign.values,
         (abs(amoc_effect) < abs(co2_effect)).values],
        [0, 4, 1, 2], default=3)
    return xr.DataArray(codes, coords=amoc_effect.coords, dims=amoc_effect.dims, name="category")


def land_area_fractions(categories, land):
    """Fraction of the (area-weighted) land area in each category, as a Series
    indexed by ``CLASS_NAMES``."""
    weights = dl.latitude_band_weights(categories["lat"]) * land
    return pd.Series([float(weights.where(categories == code).sum() / weights.sum())
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


def zonal_category_fractions(categories, land):
    """Fraction of the land cells at each latitude in each category, as a
    ``(category, lat)`` DataArray (NaN at latitudes with no land)."""
    land_cells = land.sum("lon")
    return xr.concat([(categories == code).where(land).sum("lon") / land_cells.where(land_cells > 0)
                      for code in range(len(CLASS_NAMES))],
                     dim=pd.Index(range(len(CLASS_NAMES)), name="category"))


def land_metrics(fields, control_annual, threshold_sd):
    """All metrics for one variable on land, as a ``(lat, lon)`` Dataset with ocean
    cells set to NaN (``data_loader.land_mask``; the mask itself is kept as ``land``).

    ``fields`` is a ``data_loader.case_annual_fields`` result for ``CO2_AMOC_CASES``
    (the averaging window); ``control_annual`` the c1a20 ``(year, lat, lon)`` annual
    means from which sigma is estimated (``interannual_sd``). Effect p-values are
    Welch t-tests on 10-year means (``regression.difference_pvalue``).
    """
    land = dl.land_mask(fields["lat"], fields["lon"])
    runs = {case: fields.sel(case=case, drop=True) for case in dl.CO2_AMOC_CASES}
    differences = effects(fields.mean("year"))
    pvalues = {f"{name}_pvalue": reg.difference_pvalue(runs[minuend], runs[subtrahend],
                                                        reg.DECADAL_BLOCK)
               for name, (minuend, subtrahend) in EFFECTS.items()}
    sd = interannual_sd(control_annual)
    amoc_effect, co2_effect = differences["amoc_effect"], differences["co2_effect"]
    metrics = xr.Dataset({
        **differences, **pvalues, "sd": sd,
        "share": bounded_share(amoc_effect, co2_effect),
        **scaled_metrics(amoc_effect, co2_effect, sd),
        "category": classify(amoc_effect, co2_effect, sd, pvalues["amoc_effect_pvalue"],
                             pvalues["co2_effect_pvalue"], threshold_sd),
    }).where(land).assign(land=land)
    years = control_annual["year"].values
    metrics.attrs = {**fields.attrs, "name": fields.name, "threshold_sd": threshold_sd,
                     "sd_years": f"{years.min()}-{years.max()}", "scale_label": "σ"}
    return metrics


def land_scaled_metrics(metrics, scale, scale_label):
    """``scaled_metrics`` of a ``land_metrics`` result's effects with another scale
    S (``scale``, a ``(lat, lon)`` field in the variable's units, described by
    ``scale_label``), on land, with S kept as ``scale``."""
    scaled = scaled_metrics(metrics["amoc_effect"], metrics["co2_effect"], scale)
    scaled = scaled.assign(scale=scale).where(metrics["land"]).assign(land=metrics["land"])
    scaled.attrs = {**metrics.attrs, "scale_label": scale_label}
    return scaled
