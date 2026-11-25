"""Plot regression-coefficient maps with significance contours.

Coefficient maps use a diverging colormap with symmetric bounds (white = 0, per
the project plotting conventions) and simplified Natural Earth coastlines
(``draw_coastlines``). Fields with p-values are also
contoured at their colorbar ticks (``contour_significant``): black where p <= 0.05,
light grey where not.
"""

import matplotlib

matplotlib.use("Agg")
import functools
import textwrap
from collections import namedtuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contourpy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import openpyxl.styles
import pandas as pd
import shapely
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy import stats

import co2_amoc_metrics as cam
import data_loader as dl

SIGNIFICANCE_P = 0.05
PROJECTION = ccrs.EqualEarth()  # default for all maps (UN guidance)
DATA_CRS = ccrs.PlateCarree()

# Coarse coastline: Natural Earth 110m coastline lines, Douglas-Peucker simplified
# with a tolerance of COASTLINE_TOLERANCE degrees (lon/lat), dropping lines shorter
# than COASTLINE_MIN_LENGTH degrees (small islands). Still finer than the
# 1.9 x 2.5 deg model grid, with ~1/5 the vertices of the 110m set, so vector
# coastlines stay small in many-panel PDFs.
COASTLINE_TOLERANCE = 1.0
COASTLINE_MIN_LENGTH = 5.0

# CF-convention units string for a dimensionless quantity (e.g. planetary albedo).
# It stays "1" in the data attributes but is left off plot labels.
DIMENSIONLESS_UNITS = "1"


def label_with_units(label, units):
    """``"label (units)"``, or bare ``label`` for a dimensionless quantity."""
    return label if units == DIMENSIONLESS_UNITS else f"{label} ({units})"


def value_with_units(value_text, units):
    """``"value_text units"``, or bare ``value_text`` for a dimensionless quantity."""
    return value_text if units == DIMENSIONLESS_UNITS else f"{value_text} {units}"


@functools.cache
def coarse_coastline():
    """Simplified global coastline as an array of Shapely LineStrings (lon/lat)."""
    lines = shapely.get_parts(np.array(list(
        cfeature.NaturalEarthFeature("physical", "coastline", "110m").geometries())))
    return shapely.simplify(lines[shapely.length(lines) >= COASTLINE_MIN_LENGTH],
                            COASTLINE_TOLERANCE)


def draw_coastlines(ax):
    """Draw ``coarse_coastline`` on a Cartopy map ``ax``."""
    ax.add_geometries(coarse_coastline(), crs=DATA_CRS, facecolor="none",
                      edgecolor="black", linewidth=0.5)


def centered_lon(da):
    """``da`` with longitude relabeled to [-180, 180) and sorted, for mapping.

    The CAM grid runs 0..357.5 E, so drawn as is its wrap point lands on the
    central meridian of the Greenwich-centered map projection, leaving a thin
    unfilled stripe at 0 deg. Relabeled, the wrap point moves to +/-180 deg, the
    map's outer edge. ``draw_field`` applies it; contour calls on a
    map need it too.
    """
    return da.assign_coords(lon=(da["lon"] + 180) % 360 - 180).sortby("lon")


def draw_field(ax, da, **kwargs):
    """``pcolormesh`` of a ``(lat, lon)`` field on map ``ax``, returning the mesh.

    The cell corners are projected here, once, and drawn in the map's own
    coordinates. Cartopy's ``pcolormesh(transform=...)`` does the same job but
    spends ~1.3 s per call checking for cells that wrap around the map edge,
    which made a 9-panel page take ~12 s. So that no cell straddles the edge,
    longitudes are relabeled to [-180, 180), the -180 column is repeated at +180,
    and the outer cell edges are clipped to +/-180 deg lon and +/-90 deg lat.
    ``kwargs`` go to ``pcolormesh`` (``cmap``, ``vmin``, ``vmax``, ``rasterized``).
    """
    da = centered_lon(da)
    lon, lat = da["lon"].values, da["lat"].values
    values = np.concatenate([da.values, da.values[:, :1]], axis=1)
    lon_edges = np.concatenate([[-180.0], (lon[1:] + lon[:-1]) / 2,
                                [lon[-1] + (lon[-1] - lon[-2]) / 2, 180.0]])
    lat_edges = np.concatenate([[-90.0], (lat[1:] + lat[:-1]) / 2, [90.0]])
    corners = ax.projection.transform_points(DATA_CRS, *np.meshgrid(lon_edges, lat_edges))
    return ax.pcolormesh(corners[..., 0], corners[..., 1], values, shading="flat", **kwargs)

# Line-plot convention for cases: CO2 level sets the line style (1x solid,
# 2x dotted, 4x dashed) and hosing sets the color (-0.3 Sv red, 0 black,
# +0.3 Sv blue).
CO2_LINESTYLE = {1: "-", 2: ":", 4: "--"}
HOSING_COLOR = {-0.3: "red", 0.0: "black", 0.3: "blue"}


def case_line_style(case):
    """Matplotlib ``color``/``linestyle`` kwargs for a case's line."""
    spec = dl.EXPERIMENTS[case]
    return {"color": HOSING_COLOR[spec["hosing_sv"]],
            "linestyle": CO2_LINESTYLE[spec["co2_multiple"]]}


# Scatter-marker convention: CO2 level sets the marker shape -- 1x circle,
# 2x triangle, 4x square; hosing sets the color, as for lines. The triangle has the
# least ink at a given size, so the least-emphasized 2xCO2 level draws least attention.
# Markers are filled when each case contributes at most FILLED_MARKER_MAX_POINTS
# points (e.g. 10 decadal means) and open when it contributes more (e.g. ~100
# annual values), so dense clouds of overlapping points stay readable.
CO2_MARKER = {1: "o", 2: "^", 4: "s"}
FILLED_MARKER_MAX_POINTS = 10


def case_marker_style(case, points_per_case):
    """``ax.scatter`` kwargs for a case's ``points_per_case`` points: marker shape by
    CO2, color by hosing, filled or open per ``FILLED_MARKER_MAX_POINTS``."""
    spec = dl.EXPERIMENTS[case]
    color = HOSING_COLOR[spec["hosing_sv"]]
    filled = points_per_case <= FILLED_MARKER_MAX_POINTS
    return {"marker": CO2_MARKER[spec["co2_multiple"]],
            "facecolors": color if filled else "none",
            "edgecolors": "none" if filled else color,
            "linewidths": 0.7}

# Axis labels for the scalar predictors (used by the scatter plot).
SCALAR_AXIS_LABELS = {
    "tas_global_mean": "global-mean tas (K)",
    "amoc_strength": "AMOC strength (Sv)",
    "tas_interhemispheric_diff": "interhemispheric tas diff, NH−SH (K)",
}


def symmetric_bound(values):
    """Robust symmetric color bound: the 99th percentile of |values| (NaN-safe)."""
    return float(np.nanpercentile(np.abs(values), 99))


# Zonal-statistics plots: each map page is followed by a page of line plots laid
# out panel for panel like it, showing the statistics over longitude at each
# latitude (``data_loader.zonal_statistics``) on a sine-of-latitude axis, so equal
# widths are equal areas. The statistics are told apart by line weight and fill,
# never by dashes (dots and dashes mean 2x and 4xCO2 in the case line convention).
# Case panels take their case's hosing color; other panels are ZONAL_NEUTRAL_COLOR.
# Latitude ticks nearly evenly spaced in sin(latitude) (0, 0.26, 0.5, 0.77, 1), so
# the pole labels have as much room as the rest (90° and 60° would crowd).
SINE_LATITUDE_TICKS = [-90, -50, -30, -15, 0, 15, 30, 50, 90]
ZONAL_STYLE = {
    "mean": {"lw": 1.8},
    "median": {"lw": 0.8},
    "band": {"alpha": 0.2, "lw": 0},
    "extremes": {"lw": 0.4, "alpha": 0.5},
}
ZONAL_NEUTRAL_COLOR = "black"
ZONAL_LEGEND_COLOR = "0.3"
ZONAL_PAD_FRACTION = 0.02  # y-range padding, as a fraction of the data span
ZONAL_LEGEND_HANDLES = [
    Line2D([], [], color=ZONAL_LEGEND_COLOR, label="mean", **ZONAL_STYLE["mean"]),
    Line2D([], [], color=ZONAL_LEGEND_COLOR, label="median", **ZONAL_STYLE["median"]),
    Patch(facecolor=ZONAL_LEGEND_COLOR, label="{:.0f}–{:.0f}% of cells".format(
        *(100 * q for q in dl.ZONAL_BAND_QUANTILES)), **ZONAL_STYLE["band"]),
    Line2D([], [], color=ZONAL_LEGEND_COLOR, label="min / max", **ZONAL_STYLE["extremes"]),
]

# One panel of a zonal-statistics page: ``field`` is a (lat, lon) DataArray and
# ``ylim`` the (shared) y range.
ZonalPanel = namedtuple("ZonalPanel", "title field color ylim ylabel")


def latitude_label(lat):
    """Tick label for latitude ``lat`` (deg): ``"60°S"``, ``"EQ"``, ``"60°N"``."""
    return "EQ" if lat == 0 else f"{abs(lat)}°{'N' if lat > 0 else 'S'}"


def set_sine_latitude_axis(ax):
    """Scale the x axis of ``ax`` by sin(latitude), ticked and labeled in degrees."""
    ax.set_xscale("function", functions=(
        lambda lat: np.sin(np.deg2rad(lat)),
        lambda sine: np.rad2deg(np.arcsin(np.clip(sine, -1, 1)))))
    ax.set_xlim(-90, 90)
    ax.set_xticks(SINE_LATITUDE_TICKS, [latitude_label(t) for t in SINE_LATITUDE_TICKS],
                  fontsize=7)
    ax.set_xlabel("latitude (sine scale)", fontsize=8)


def draw_zonal_statistics(ax, field, color):
    """Draw the zonal statistics of a ``(lat, lon)`` field on ``ax`` in ``color``:
    the 5-95% band, min/max hairlines, the median and the mean, plus a zero line
    (off-axis once a raw field's y range is set) on a sine-latitude axis."""
    zonal = dl.zonal_statistics(field)
    lat = zonal["lat"].values
    ax.fill_between(lat, zonal["low"].values, zonal["high"].values, color=color,
                    **ZONAL_STYLE["band"])
    for extreme in ("min", "max"):
        ax.plot(lat, zonal[extreme].values, color=color, **ZONAL_STYLE["extremes"])
    ax.plot(lat, zonal["median"].values, color=color, **ZONAL_STYLE["median"])
    ax.plot(lat, zonal["mean"].values, color=color, **ZONAL_STYLE["mean"])
    ax.axhline(0, color="0.5", lw=0.5)
    set_sine_latitude_axis(ax)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.3)


def zonal_range(fields):
    """Shared y range spanning the min and max of every field (NaN-safe), padded
    by ``ZONAL_PAD_FRACTION`` of the span."""
    low = min(float(field.min()) for field in fields)
    high = max(float(field.max()) for field in fields)
    pad = ZONAL_PAD_FRACTION * (high - low)
    return low - pad, high + pad


def plot_zonal_grid(panels, shape, title, pdf, panel_size=(5.0, 3.2)):
    """One page of ``pdf``: zonal-statistics panels on a ``shape`` (rows, cols) grid,
    each ``panel_size`` (width, height) inches.

    ``panels`` maps ``(row, col)`` to a ``ZonalPanel``; grid cells not in
    ``panels`` are left blank, mirroring the map page it follows.
    """
    fig, axes = plt.subplots(*shape, figsize=(panel_size[0] * shape[1], panel_size[1] * shape[0]),
                             squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.set_visible(False)
    for (i, j), panel in panels.items():
        ax = axes[i, j]
        ax.set_visible(True)
        draw_zonal_statistics(ax, panel.field, panel.color)
        ax.set_ylim(panel.ylim)
        ax.set_ylabel(panel.ylabel, fontsize=8)
        ax.set_title(panel.title, fontsize=9)
    fig.legend(handles=ZONAL_LEGEND_HANDLES, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle(title, fontsize=12)
    _save_figure(fig, pdf=pdf)


def _save_figure(fig, out_path=None, pdf=None, dpi=300):
    """Write ``fig`` as a standalone PDF (``out_path``) or one page of ``pdf``.

    ``pdf`` is a ``matplotlib.backends.backend_pdf.PdfPages`` handle; when given,
    the figure is appended as a page (so many figures collect into one file).
    """
    if pdf is not None:
        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# Significance on maps: the field is contoured at its colorbar ticks
# (``colorbar_levels``) in black where p <= SIGNIFICANCE_P and in light grey where it
# is not, so small significant areas stand out. Lines are solid (dashes mean CO2 level in this project's figures).
SIGNIFICANT_CONTOUR_COLOR = "black"
INSIGNIFICANT_CONTOUR_COLOR = "0.75"
SIGNIFICANCE_CONTOUR_WIDTH = 0.4
SIGNIFICANCE_CONTOUR_KEY = (
    f"black contours: p ≤ {SIGNIFICANCE_P}; light grey contours: not significant"
)
COLORBAR_MAX_TICKS = 8


def colorbar_levels(vmin, vmax):
    """Round tick values within [``vmin``, ``vmax``]: a map's colorbar ticks and the
    levels of its significance contours."""
    ticks = MaxNLocator(nbins=COLORBAR_MAX_TICKS).tick_values(vmin, vmax)
    return ticks[(ticks >= vmin) & (ticks <= vmax)]


def contour_significant(ax, field, pvalue, levels):
    """Contour the ``(lat, lon)`` ``field`` on map ``ax`` at ``levels``: black where
    ``pvalue`` <= ``SIGNIFICANCE_P``, light grey where it is not.

    The full field is contoured once (``contourpy``) and each line segment is
    colored by the p-value of the grid cell nearest its midpoint, so a contour runs
    unbroken and changes color at the edge of the significant region. Segments over
    cells with NaN p-values are left out. As in ``draw_field``, the -180 column is
    repeated at +180 (no gap at the map edge) and the vertices are projected here."""
    field, pvalue = centered_lon(field), centered_lon(pvalue)
    values = np.concatenate([field.values, field.values[:, :1]], axis=1)
    generator = contourpy.contour_generator(np.append(field["lon"].values, 180.0),
                                            field["lat"].values, np.ma.masked_invalid(values))
    segments = np.concatenate(
        [np.stack([line[:-1], line[1:]], axis=1)
         for level in levels for line in generator.lines(level)] + [np.empty((0, 2, 2))])
    midpoints = segments.mean(axis=1)
    segment_pvalue = pvalue.sel(lon=xr.DataArray(midpoints[:, 0]),
                                lat=xr.DataArray(midpoints[:, 1]), method="nearest").values
    keep = ~np.isnan(segment_pvalue)
    significant = segment_pvalue[keep] <= SIGNIFICANCE_P
    order = np.argsort(significant, kind="stable")  # grey first, black drawn on top
    projected = ax.projection.transform_points(DATA_CRS, segments[keep, :, 0],
                                               segments[keep, :, 1])[order, :, :2]
    ax.add_collection(LineCollection(
        projected,
        colors=np.where(significant[order], SIGNIFICANT_CONTOUR_COLOR,
                        INSIGNIFICANT_CONTOUR_COLOR),
        linewidths=SIGNIFICANCE_CONTOUR_WIDTH, linestyles="solid"), autolim=False)


def plot_coefficient_map(coef, pvalue, title, units, ax, cmap="RdBu_r", bound=None,
                         rasterized=False):
    """Draw one coefficient map on a Cartopy ``ax``: filled field, contoured at its
    colorbar ticks, black where p <= 0.05 and light grey elsewhere (``contour_significant``).

    ``coef`` and ``pvalue`` are 2-D (lat, lon) DataArrays. ``cmap`` is a diverging
    colormap (white = 0); use ``RdBu_r`` for temperature (warm = red) and ``RdBu``
    for precipitation (wet = blue). The symmetric color scale is fixed to ±``bound``
    if given, else the 99th percentile of |coef| (values beyond saturate). With
    ``rasterized`` the filled field (the heavy artist) is embedded as raster while
    axes/text stay vector -- keeps many-panel PDFs small. Returns the mappable.
    """
    coef = centered_lon(coef)
    b = bound if bound is not None else symmetric_bound(coef.values)
    mesh = draw_field(ax, coef, cmap=cmap, vmin=-b, vmax=b, rasterized=rasterized)
    levels = colorbar_levels(-b, b)
    contour_significant(ax, coef, centered_lon(pvalue), levels)
    draw_coastlines(ax)
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax.set_title(title, fontsize=10)
    cbar = ax.figure.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.03,
                              ticks=levels)
    cbar.set_label(units)
    return mesh


def plot_set(fit, set_def, run_label, pdf, predictand, centering=None):
    """Render all predictor coefficient maps for one regression set as one page of
    ``pdf`` (an open ``PdfPages`` book), followed by a page of their zonal
    statistics in the same layout.

    One panel per predictor (the intercept is omitted). Contours (at the colorbar
    ticks) mark cells where p <= 0.05.
    Panels whose coefficients carry the same units share one symmetric color scale
    and, on the zonal page, one y range.
    ``predictand`` is a ``regression.PREDICTANDS`` entry (label + units), used for
    titles and to form coefficient units ([predictand units] / [predictor units]).
    ``centering`` is the ``tag -> (mean, units)`` dict from
    ``regression.centering_means_for_set`` (empty/None for raw-predictor sets); when
    present its means are annotated, since they are the offsets a user must subtract
    before applying the centered cross-product terms.
    """
    from regression import PREDICTORS  # local import to avoid a cycle at import time

    plabel, punits = predictand["label"], predictand["units"]
    cmap = predictand.get("cmap", "RdBu_r")
    predictors = set_def["predictors"]
    n = len(predictors)
    # Stack few panels in a single column; lay many (e.g. the 9-term set) on a grid.
    ncols = 3 if n > 4 else 1
    nrows = -(-n // ncols)
    panel_size = (9.0, 4.0) if ncols == 1 else (6.0, 3.4)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
        subplot_kw={"projection": PROJECTION},
    )
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    units_of = {name: f"({punits}) / {PREDICTORS[name]['units']}" for name in predictors}
    bound_of = {units: symmetric_bound(np.concatenate(
                    [fit["coef"].sel(param=name).values.ravel()
                     for name in predictors if units_of[name] == units]))
                for units in set(units_of.values())}
    title_of = {name: f"∂{plabel}/∂{PREDICTORS[name]['label']}  "
                      f"({PREDICTORS[name]['label']} coefficient)" for name in predictors}
    for ax, name in zip(flat, predictors):
        plot_coefficient_map(
            fit["coef"].sel(param=name),
            fit["pvalue"].sel(param=name),
            title=title_of[name],
            units=units_of[name],
            ax=ax,
            cmap=cmap,
            bound=bound_of[units_of[name]],
        )
    preds = ", ".join(PREDICTORS[p]["label"] for p in predictors)
    centering_line = ""
    if centering:
        parts = ", ".join(f"{tag} = {mean:.4g} {units}" for tag, (mean, units) in centering.items())
        centering_line = f"\ncentering means (subtract before applying centered terms): {parts}"
    set_title = f"Set {set_def['number']}: {plabel} ~ {preds}  |  {run_label}"
    fig.suptitle(
        f"{set_title}\n"
        f"{SIGNIFICANCE_CONTOUR_KEY} "
        f"(nominal OLS; autocorrelation not corrected)"
        f"{centering_line}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, pdf=pdf)

    ylim_of = {units: zonal_range([fit["coef"].sel(param=name)
                                   for name in predictors if units_of[name] == units])
               for units in set(units_of.values())}
    plot_zonal_grid(
        {divmod(k, ncols): ZonalPanel(title_of[name], fit["coef"].sel(param=name),
                                      ZONAL_NEUTRAL_COLOR, ylim_of[units_of[name]],
                                      units_of[name])
         for k, name in enumerate(predictors)},
        (nrows, ncols),
        f"{set_title}\nzonal statistics over longitude of the coefficients "
        f"(all cells, significant or not){centering_line}",
        pdf, panel_size,
    )


def plot_eof_patterns(eof_ds, title, pdf, cmap="RdBu_r", max_patterns=9):
    """Map the leading EOF spatial patterns plus a scree panel of variance explained.

    ``eof_ds`` is the Dataset from ``eof.compute_eofs``, whose EOFs are dimensionless
    patterns of area-weighted RMS 1 (the PCs carry the physical amplitude). All
    mapped patterns share one symmetric diverging color scale. Only the leading ``max_patterns`` modes are mapped (fields
    such as precipitation are not low-rank and can retain hundreds of modes at the
    95% threshold — mapping them all is unreadable and the regression uses every
    retained mode regardless). The final panel is a scree: a per-mode bar when the
    modes are few, otherwise a cumulative-variance curve marking the retained count.
    Appended as a page of ``pdf``, followed by a page of the mapped patterns' zonal
    statistics in the same layout (sharing one y range; no scree panel).
    """
    eofs = eof_ds["eofs"]
    n = eofs.sizes["mode"]
    var = eof_ds["variance_fraction"].values
    n_plot = min(n, max_patterns)
    ncols = 2
    nrows = -(-(n_plot + 1) // ncols)  # +1 for the scree panel
    fig = plt.figure(figsize=(6.0 * ncols, 3.4 * nrows))
    bound = symmetric_bound(eofs.isel(mode=slice(0, n_plot)).values)
    for i in range(n_plot):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=PROJECTION)
        mesh = draw_field(ax, eofs.isel(mode=i), cmap=cmap, vmin=-bound, vmax=bound)
        draw_coastlines(ax)
        ax.set_global()
        ax.set_title(f"EOF {i + 1}  ({var[i] * 100:.1f}% var)", fontsize=10)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("dimensionless (RMS = 1)")
    ax = fig.add_subplot(nrows, ncols, n_plot + 1)
    if n <= 20:
        ax.bar(np.arange(1, n + 1), var * 100)
        ax.set_ylabel("variance explained (%)")
    else:
        ax.plot(np.arange(1, n + 1), np.cumsum(var) * 100, lw=1.2)
        ax.set_ylabel("cumulative variance (%)")
    ax.set_xlabel("EOF mode")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"scree: {n} modes retained (Σ = {var.sum() * 100:.1f}%); "
                 f"mapped leading {n_plot}", fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, pdf=pdf)

    patterns = [eofs.isel(mode=i) for i in range(n_plot)]
    ylim = zonal_range(patterns)
    plot_zonal_grid(
        {divmod(i, ncols): ZonalPanel(f"EOF {i + 1}  ({var[i] * 100:.1f}% var)", pattern,
                                      ZONAL_NEUTRAL_COLOR, ylim, "dimensionless (RMS = 1)")
         for i, pattern in enumerate(patterns)},
        (-(-n_plot // ncols), ncols), f"{title}\nzonal statistics over longitude", pdf,
    )


def plot_pc_regression(pc_fit, predictors, pcs, title, out_path=None,
                       variance_fraction=None, pdf=None):
    """Per-mode bar charts of the PC-on-predictor regression (standardized).

    The EOF analog of the 2D coefficient maps: one panel per retained EOF mode,
    with one bar per predictor. Bar height is the **standardized** coefficient
    β·σ(xⱼ)/σ(PCₘ) (z-scoring predictors and the PC) so bars are comparable across
    modes — raw coefficients scale with each PC's amplitude. The whisker is the
    matching standardized SE. Bars significant at p < ``SIGNIFICANCE_P`` are drawn
    solid, non-significant ones faded; significance/p come straight from ``pc_fit``
    (scale-invariant). Panel titles report R² and (if given) the mode's variance
    fraction. ``predictors`` must contain the regressed columns; ``pcs`` is the
    (sample, mode) PC array used in the fit.
    """
    from regression import PREDICTORS  # local import to avoid an import cycle

    names = [p for p in pc_fit["param"].values if p != "intercept"]
    labels = [PREDICTORS[p]["label"] for p in names]
    sigma_x = np.array([predictors[p].values.std() for p in names])  # (k,)
    sigma_y = pcs.std("sample").values  # (mode,)
    modes = pc_fit["mode"].values
    n = modes.size

    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False,
                             sharey=True)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    x = np.arange(len(names))
    for i, mode in enumerate(modes):
        ax = flat[i]
        sel = pc_fit.sel(mode=mode)
        scale = sigma_x / sigma_y[i]  # standardize each predictor's coefficient
        beta = sel["coef"].sel(param=names).values * scale
        err = sel["se"].sel(param=names).values * scale
        sig = sel["pvalue"].sel(param=names).values < SIGNIFICANCE_P
        base = np.where(beta >= 0, "#c0392b", "#2c5fa8")  # warm +, cool -
        rgba = [mcolors.to_rgba(c, 1.0 if s else 0.35) for c, s in zip(base, sig)]
        ax.bar(x, beta, yerr=err, color=rgba, edgecolor="k", linewidth=0.5, capsize=3)
        ax.axhline(0, color="k", lw=0.6)
        vtxt = f", {variance_fraction[i] * 100:.0f}% var" if variance_fraction is not None else ""
        ax.set_title(f"EOF {int(mode)}  (R²={float(sel['r2'].values):.2f}{vtxt})", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("standardized coef (β·σx/σy)")
    fig.suptitle(
        f"{title}\nbars = standardized coefficients; faded = not significant "
        f"(p > {SIGNIFICANCE_P})", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, out_path, pdf)


def plot_pc_prediction(eof_ds, pc_fit, predictors, title, units, out_path=None,
                       pdf=None, max_modes=3):
    """Overlay the fitted (X·β) PC against the actual PC over time, per simulation.

    A direct view of how well the scalar predictors reproduce each EOF weighting:
    the fitted PC is ``intercept + Σⱼ βⱼ xⱼ`` in raw PC units. The leading
    ``max_modes`` modes are drawn (solid = actual, dashed = fitted) on one panel
    per run; lines break across genuine year gaps but stay connected across
    regular decadal steps. ``predictors`` must contain the regressed columns.
    ``units`` are the field's units, which the PCs carry (``eof.compute_eofs``).
    """
    pcs = eof_ds["pcs"]
    years = eof_ds["sample"].values
    run_of = eof_ds["run"].values
    runs = list(dict.fromkeys(run_of))
    params = list(pc_fit["param"].values)
    names = [p for p in params if p != "intercept"]

    X = np.column_stack([np.ones(pcs.sizes["sample"])] + [predictors[p].values for p in names])
    coef = pc_fit["coef"].sel(param=params).values  # (k, mode), intercept first
    fitted = X @ coef  # (sample, mode), aligned with pcs' sample axis

    n_modes = min(pcs.sizes["mode"], max_modes)
    colors = plt.cm.tab10(np.arange(n_modes))
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.6 * len(runs)), squeeze=False,
                             sharex=True, sharey=True)
    for ax, run in zip(axes[:, 0], runs):
        m = run_of == run
        order = np.argsort(years[m])
        yr = years[m][order].astype(float)
        d = np.diff(yr)
        thresh = 1.5 * np.median(d) if d.size else np.inf
        gaps = np.where(d > thresh)[0] + 1
        yr_b = np.insert(yr, gaps, np.nan)
        for k in range(n_modes):
            act = pcs.isel(mode=k).values[m][order]
            fit_run = fitted[:, k][m][order]
            ax.plot(yr_b, np.insert(act, gaps, np.nan), lw=1.3, color=colors[k],
                    label=f"PC{k + 1} actual")
            ax.plot(yr_b, np.insert(fit_run, gaps, np.nan), lw=1.0, color=colors[k],
                    ls="--", label=f"PC{k + 1} fitted")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(run, fontsize=10)
        ax.set_ylabel(label_with_units("PC", units))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=n_modes, loc="best")
    axes[-1, 0].set_xlabel("year")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, out_path, pdf)


def plot_scalar_timeseries(annual, decadal, title, out_path):
    """Per-simulation time series of the scalar predictors, annual + decadal overlay.

    One panel per simulation. Tglob and ΔT_NS share the left axis as **anomalies
    from their pooled means** (they cannot share a raw axis — Tglob is ~287 K,
    ΔT_NS only a few K); AMOC strength is on a right axis in **absolute Sv** (its
    own axis, so real magnitudes and any collapse stay visible). Annual values are
    thin lines; decadal block means are overlaid as marked lines (centered on the
    same annual baseline, so they sit on the annual curves). Lines break across
    genuine year gaps (a discontinuity left by dropping NaN years). ``annual`` and
    ``decadal`` are pooled predictor Datasets (variables on ``sample`` with a
    ``run`` coord) from ``regression.build_pooled`` (block=None and block=10).
    """
    left_vars = [("tas_global_mean", "Tglob", "C0"),
                 ("tas_interhemispheric_diff", "ΔT_NS", "C1")]
    ref = {v: float(annual[v].values.mean()) for v, _, _ in left_vars}  # shared baseline
    runs = list(dict.fromkeys(annual["run"].values))

    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.9 * len(runs)), squeeze=False,
                             sharex=True, sharey=True)
    twins = [ax.twinx() for ax in axes[:, 0]]
    for twin in twins[1:]:  # one AMOC (Sv) range on every panel, like the K axes
        twin.sharey(twins[0])
    for ax, ax2, run in zip(axes[:, 0], twins, runs):
        for ds, style in [(annual, dict(lw=0.9, alpha=0.65)),
                          (decadal, dict(lw=1.8, marker="o", ms=3))]:
            m = ds["run"].values == run
            yr = ds["sample"].values[m].astype(float)
            order = np.argsort(yr)
            yr = yr[order]
            d = np.diff(yr)
            thresh = 1.5 * np.median(d) if d.size else np.inf  # break only real gaps
            gaps = np.where(d > thresh)[0] + 1
            yr_b = np.insert(yr, gaps, np.nan)
            for var, _lbl, c in left_vars:
                v = ds[var].values[m][order] - ref[var]
                ax.plot(yr_b, np.insert(v, gaps, np.nan), color=c, **style)
            a = ds["amoc_strength"].values[m][order]
            ax2.plot(yr_b, np.insert(a, gaps, np.nan), color="C3", **style)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(run, fontsize=10)
        ax.set_ylabel("Tglob, ΔT_NS anomaly (K)")
        ax2.set_ylabel("AMOC (Sv)", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("year")

    handles = [
        Line2D([], [], color="C0", label="Tglob (left, K anomaly)"),
        Line2D([], [], color="C1", label="ΔT_NS (left, K anomaly)"),
        Line2D([], [], color="C3", label="AMOC (right, Sv)"),
        Line2D([], [], color="0.4", lw=0.9, label="annual"),
        Line2D([], [], color="0.4", lw=1.8, marker="o", ms=3, label="decadal mean"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=8, ncol=5, loc="best", framealpha=0.9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_predictor_scatter(predictors, out_path):
    """Four-panel scatter of the pooled predictors, points styled by case
    (``case_marker_style``).

    Top row uses global-mean tas on the x-axis (AMOC and ΔT_NS on y); bottom row
    uses AMOC strength on the x-axis (global-mean tas and ΔT_NS on y). The pooled
    Pearson r is annotated per panel. ``predictors`` is the pooled Dataset from
    ``regression.build_pooled`` (variables on ``sample``, with a ``run`` coord).
    """
    panels = [
        ("tas_global_mean", "amoc_strength"),
        ("tas_global_mean", "tas_interhemispheric_diff"),
        ("amoc_strength", "tas_global_mean"),
        ("amoc_strength", "tas_interhemispheric_diff"),
    ]
    runs = [case for case in dl.EXPERIMENTS if case in set(predictors["run"].values)]
    run_of = predictors["run"].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (xv, yv) in zip(axes.flat, panels):
        x, y = predictors[xv].values, predictors[yv].values
        for run in runs:
            m = run_of == run
            ax.scatter(x[m], y[m], s=14, alpha=0.7, label=run,
                       **case_marker_style(run, m.sum()))
        r = float(np.corrcoef(x, y)[0, 1])
        ax.set_xlabel(SCALAR_AXIS_LABELS[xv])
        ax.set_ylabel(SCALAR_AXIS_LABELS[yv])
        ax.set_title(f"pooled r = {r:+.2f}", fontsize=10)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8, markerscale=1.6, title="simulation")
    fig.suptitle(
        f"Pooled regression predictors (AMOC-complete sample, {len(runs)} simulations)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_itcz_timeseries(annual, decadal, title, out_path=None, pdf=None):
    """Per-simulation time series of the ITCZ latitude (``precip_max_lat``).

    One panel per simulation; annual values as a thin line and decadal block means
    overlaid as a marked line. Lines break across genuine year gaps (a
    discontinuity left by dropping NaN years). ``annual``/``decadal`` are the pooled response
    DataArrays (on ``sample`` with a ``run`` coord) from
    ``regression.build_pooled_scalar`` (block=None and block=10).
    """
    runs = list(dict.fromkeys(annual["run"].values))
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.6 * len(runs)), squeeze=False,
                             sharex=True, sharey=True)
    for ax, run in zip(axes[:, 0], runs):
        for da, style in [(annual, dict(lw=0.9, alpha=0.65)),
                          (decadal, dict(lw=1.8, marker="o", ms=3))]:
            m = da["run"].values == run
            yr = da["sample"].values[m].astype(float)
            order = np.argsort(yr)
            yr = yr[order]
            d = np.diff(yr)
            thresh = 1.5 * np.median(d) if d.size else np.inf  # break only real gaps
            gaps = np.where(d > thresh)[0] + 1
            yr_b = np.insert(yr, gaps, np.nan)
            v = da.values[m][order]
            ax.plot(yr_b, np.insert(v, gaps, np.nan), color="C2", **style)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(run, fontsize=10)
        ax.set_ylabel("ITCZ lat (°N)")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("year")
    handles = [
        Line2D([], [], color="C2", lw=0.9, label="annual"),
        Line2D([], [], color="C2", lw=1.8, marker="o", ms=3, label="decadal mean"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=8, ncol=2, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, out_path, pdf)


def plot_itcz_predicted_vs_observed(observed, run_of, panels, title, out_path=None, pdf=None):
    """Predicted vs observed ITCZ latitude for one or more multi-predictor fits.

    One subplot per entry of ``panels`` (each a dict with ``label``, ``predicted``
    -- the fitted values on ``sample`` -- and ``r2``). Points are colored by
    simulation; the dashed 1:1 line and R² / n are drawn on shared, equal-aspect
    axes so departures from the 1:1 line read directly as prediction error.
    ``observed`` is the response array and ``run_of`` the per-sample run labels.
    """
    runs = [case for case in dl.EXPERIMENTS if case in set(run_of)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.8),
                             squeeze=False)
    # One range for both axes of every panel, spanning all observed and predicted values.
    values = np.concatenate([observed] + [np.asarray(p["predicted"]) for p in panels])
    pad = 0.05 * (values.max() - values.min())
    lim = (float(values.min() - pad), float(values.max() + pad))
    for ax, panel in zip(axes[0], panels):
        pred = panel["predicted"]
        for run in runs:
            m = run_of == run
            ax.scatter(observed[m], pred[m], s=14, alpha=0.7,
                       label=run, **case_marker_style(run, m.sum()))
        ax.plot(lim, lim, color="k", lw=1.0, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("observed ITCZ lat (°N)")
        ax.set_ylabel("predicted ITCZ lat (°N)")
        ax.set_title(f"{panel['label']}\nR² = {panel['r2']:.2f}, n = {observed.size}",
                     fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, markerscale=1.6, title="simulation")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, out_path, pdf)


def plot_itcz_coefficients(panels, title, out_path=None, pdf=None):
    """Partial-slope bar charts (coef ± SE) for one or more multi-predictor fits.

    One subplot per entry of ``panels`` (each a dict with ``label`` and equal-length
    ``names`` / ``coef`` / ``se`` / ``pvalue`` lists, intercept excluded). Bars are
    the partial slopes with ±SE error bars, blue for positive and red for negative,
    hatched where not significant at p < 0.05. Note the slopes carry mixed units
    (° lat per K, per Sv, or per K·Sv for the interaction), so compare sign and
    significance rather than bar heights across different predictors.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4),
                             squeeze=False, sharey=True)
    for ax, panel in zip(axes[0], panels):
        x = np.arange(len(panel["names"]))
        coef = np.asarray(panel["coef"])
        se = np.asarray(panel["se"])
        sig = np.asarray(panel["pvalue"]) < SIGNIFICANCE_P
        ax.bar(x, coef, yerr=se, capsize=4,
               color=["C0" if c >= 0 else "C3" for c in coef],
               hatch=["" if s else "//" for s in sig],
               edgecolor="k", linewidth=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(panel["names"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("partial slope (° lat / predictor unit)")
        ax.set_title(panel["label"], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    handles = [
        Line2D([], [], marker="s", ls="", color="C0", label="positive"),
        Line2D([], [], marker="s", ls="", color="C3", label="negative"),
        Line2D([], [], marker="s", ls="", color="0.8", mec="k", label="hatched: p≥0.05"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=7, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_figure(fig, out_path, pdf)


def plot_itcz_scatter(predictors, response, fits, single_vars, title, out_path=None, pdf=None):
    """ITCZ latitude vs each single predictor: scatter, OLS line, 95% CI band.

    One panel per variable in ``single_vars`` (the single-predictor sets: Tglob,
    ΔT_NS, AMOC). Points are colored by simulation. The pooled simple-OLS line is
    drawn with a shaded 95% mean-response confidence band, and the slope ± SE, R²
    and p-value (from the corresponding ``fits[var]`` Dataset, i.e.
    ``regression.fit_scalar_ols`` on that single predictor) are annotated.
    ``predictors`` and ``response`` are the pooled Datasets/DataArray on ``sample``.
    """
    runs = [case for case in dl.EXPERIMENTS if case in set(predictors["run"].values)]
    run_of = predictors["run"].values
    y = response.values

    fig, axes = plt.subplots(1, len(single_vars), figsize=(5.2 * len(single_vars), 4.6),
                             squeeze=False, sharey=True)
    for ax, var in zip(axes[0], single_vars):
        x = predictors[var].values
        for run in runs:
            m = run_of == run
            ax.scatter(x[m], y[m], s=14, alpha=0.7, label=run,
                       **case_marker_style(run, m.sum()))

        fit = fits[var]
        b0 = float(fit["coef"].sel(param="intercept"))
        b1 = float(fit["coef"].sel(param=var))
        slope_se = float(fit["se"].sel(param=var))
        pval = float(fit["pvalue"].sel(param=var))
        n, df = fit.attrs["nobs"], fit.attrs["df"]

        # 95% mean-response band: ŷ ± t* · s · sqrt(1/n + (x-x̄)²/Sxx).
        resid = y - (b0 + b1 * x)
        sigma2 = (resid**2).sum() / df
        xbar = x.mean()
        sxx = ((x - xbar) ** 2).sum()
        tcrit = stats.t.ppf(0.975, df)
        xx = np.linspace(x.min(), x.max(), 200)
        yhat = b0 + b1 * xx
        band = tcrit * np.sqrt(sigma2 * (1.0 / n + (xx - xbar) ** 2 / sxx))
        ax.plot(xx, yhat, color="k", lw=1.6)
        ax.fill_between(xx, yhat - band, yhat + band, color="k", alpha=0.15, lw=0)

        ax.set_xlabel(SCALAR_AXIS_LABELS[var])
        ax.set_ylabel("ITCZ lat (°N)")
        ax.set_title(
            f"slope = {b1:+.3g} ± {slope_se:.2g} °/[{SCALAR_AXIS_LABELS[var].split('(')[-1].rstrip(') ')}]\n"
            f"R² = {fit.attrs['r2']:.2f}, p = {pval:.1e}, n = {n}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, markerscale=1.6, title="simulation")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, out_path, pdf)


def plot_tglob_vs_amoc(annual, decadal, out_path):
    """x-y line plot of AMOC strength against global-mean tas, one line per case.

    ``annual`` and ``decadal`` are pooled predictor Datasets from
    ``regression.build_pooled`` (block=None and block=10): each case is drawn as a
    thin line through its annual values and a bold line through its 10-year block
    means, both in time order, with a circle at the first decadal mean. Line style
    and color follow the case convention (``case_line_style``).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for case in dl.EXPERIMENTS:
        style = case_line_style(case)
        a = annual.isel(sample=annual["run"].values == case)
        d = decadal.isel(sample=decadal["run"].values == case)
        ax.plot(a["tas_global_mean"], a["amoc_strength"], lw=0.6, alpha=0.35,
                color=style["color"])
        ax.plot(d["tas_global_mean"], d["amoc_strength"], lw=2.0, label=case, **style)
        ax.plot(d["tas_global_mean"][0], d["amoc_strength"][0], "o", ms=5,
                color=style["color"])
    ax.set_xlabel("global-mean near-surface air temperature, tas (K)")
    ax.set_ylabel("AMOC strength at 26.5°N (Sv)")
    ax.set_title(
        "CESM1.2 AMOC vs global-mean temperature, 2051–2150\n"
        "bold: 10-year block means (○ = first decade); thin: annual means",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, handlelength=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_amoc_timeseries(annual, decadal, out_path):
    """AMOC strength against year, one line per case.

    ``annual`` and ``decadal`` are pooled predictor Datasets from
    ``regression.build_pooled`` (block=None and block=10), whose ``sample``
    coordinate holds the calendar year (the block midpoint for decadal means). Each
    case is a thin line through its annual values and a bold line through its
    10-year block means, styled by ``case_line_style``.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for case in dl.EXPERIMENTS:
        style = case_line_style(case)
        a = annual.isel(sample=annual["run"].values == case)
        d = decadal.isel(sample=decadal["run"].values == case)
        ax.plot(a["sample"], a["amoc_strength"], lw=0.6, alpha=0.35, color=style["color"])
        ax.plot(d["sample"], d["amoc_strength"], lw=2.0, label=case, **style)
    ax.set_xlabel("year")
    ax.set_ylabel("AMOC strength at 26.5°N (Sv)")
    ax.set_title(
        "CESM1.2 AMOC strength at 26.5°N, 2051–2150\n"
        "bold: 10-year block means (at block midpoints); thin: annual means",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, handlelength=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# The four pages drawn per variable by ``plot_case_grid_book``: (page title, map
# of the (co2, hosing) grid to the plotted field). Differences broadcast over the
# grid, so the reference panels of each difference page are identically zero.
CASE_GRID_PAGES = [
    ("raw", lambda grid: grid),
    ("minus piControl (1xCO2, 0 Sv)", lambda grid: grid - grid.sel(co2=1, hosing=0.0)),
    ("minus 1xCO2 at the same hosing (CO2 effect)", lambda grid: grid - grid.sel(co2=1)),
    ("minus no hosing at the same CO2 (hosing effect)", lambda grid: grid - grid.sel(hosing=0.0)),
]


# Resolution of the rasterized case-grid map fields (text and coastlines stay
# vector). The nine small panels need no more, and PdfPages holds every page's
# images until the book closes: at 300 dpi a 4-page book peaked at ~5 GB.
CASE_GRID_RASTER_DPI = 150


def plot_case_grid_page(field, title, cmap, vmin, vmax, pdf, rasterized):
    """One page of ``pdf``: a 3 x 3 map grid of ``field`` (``(co2, hosing, lat,
    lon)``, laid out like ``data_loader.CASE_GRID``) on a shared color scale.

    Each panel is titled with its case name and area-weighted global mean. With
    ``rasterized`` the filled fields are embedded as raster images (small PDF);
    otherwise everything is vector.
    """
    fig, axes = plt.subplots(
        len(dl.CO2_LEVELS), len(dl.HOSING_LEVELS), figsize=(15, 9.5),
        subplot_kw={"projection": PROJECTION}, layout="constrained",
    )
    units = field.attrs["units"]
    for i, co2 in enumerate(dl.CO2_LEVELS):
        for j, hosing in enumerate(dl.HOSING_LEVELS):
            ax, panel = axes[i, j], field.sel(co2=co2, hosing=hosing)
            mesh = draw_field(ax, panel, cmap=cmap, vmin=vmin, vmax=vmax,
                              rasterized=rasterized)
            draw_coastlines(ax)
            ax.set_global()
            ax.set_title(f"{dl.CASE_GRID[i][j]}   global mean = "
                         + value_with_units(f"{float(dl.global_mean(panel)):.4g}", units),
                         fontsize=10)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", shrink=0.5,
                        pad=0.02, aspect=40)
    cbar.set_label(label_with_units(field.name, units))
    fig.suptitle(title, fontsize=12)
    _save_figure(fig, pdf=pdf, dpi=CASE_GRID_RASTER_DPI)


# CO2 x AMOC decomposition book (``scripts/plot_co2_amoc_decomposition.py``). The
# decomposition page has one column per decomposed CO2 multiple (no hosing) and
# four rows of changes from 1xCO2: the three CO2_AMOC_TERM_ROWS and their sum (the
# regression's nxCO2 - 1xCO2; the intercept a cancels). The check page has the same
# columns and rows simulated nxCO2 - 1xCO2, the regression's (as on the
# decomposition page), and simulated minus regression. Titles are formatted with
# the column's ``co2`` and the AMOC transform's name ``amoc``
# (``regression.amoc_transform_label``).
CO2_AMOC_COLUMNS = [2, 4]
CO2_AMOC_TERM_ROWS = [  # (``regression.fit_co2_amoc`` parameter, row title)
    ("log2_co2", "CO2 effect at 1xCO2 AMOC: b·Δlog2(CO2)"),
    ("amoc_anomaly", "AMOC effect at 1xCO2: c·Δ{amoc}"),
    ("log2_co2.amoc_anomaly", "nonlinear: d·Δlog2(CO2)·Δ{amoc}"),
]
CO2_AMOC_FITTED_CHANGE_TITLE = "regression {co2}xCO2 − 1xCO2: sum of the three effects"
CO2_AMOC_SIMULATED_CHANGE_TITLE = "simulated {co2}xCO2 − 1xCO2"
CO2_AMOC_CHANGE_RESIDUAL_TITLE = "simulated − regression ({co2}xCO2 − 1xCO2)"

# A panel of ``plot_field_and_change_grid``: ``change`` panels (differences) share a
# symmetric diverging scale, the others (absolute fields) a ``viridis`` scale;
# panels with a ``pvalue`` (else None) get significance contours (``contour_significant``).
MapPanel = namedtuple("MapPanel", "title field change pvalue")


def _shared_map_style(fields, change, cmap):
    """One color scale shared by ``fields``: for changes a symmetric ``cmap`` scale
    (white = 0), else ``viridis`` over the 1st-99th percentile."""
    values = np.concatenate([f.values.ravel() for f in fields])
    if change:
        bound = symmetric_bound(values)
        return {"cmap": cmap, "vmin": -bound, "vmax": bound}
    low, high = np.nanpercentile(values, [1, 99])
    return {"cmap": "viridis", "vmin": low, "vmax": high}


def plot_field_and_change_grid(panels, shape, name, units, cmap, title, pdf):
    """Two pages of ``pdf``: a ``shape`` (rows, cols) map grid mixing absolute
    fields and changes of one variable, then its zonal statistics in the same layout.

    ``panels`` maps ``(row, col)`` to a ``MapPanel``. The absolute panels share one
    ``viridis`` scale (1st-99th percentile over them) and one zonal y range; the
    change panels share one symmetric ``cmap`` scale (white = 0) and their own y
    range. One horizontal colorbar for each kind present sits under the grid. Each
    panel title gives its area-weighted global mean. Map fields are rasterized at
    ``CASE_GRID_RASTER_DPI``.
    """
    labels = {False: label_with_units(name, units), True: label_with_units(f"Δ{name}", units)}
    fields = {change: [p.field for p in panels.values() if p.change == change]
              for change in {p.change for p in panels.values()}}
    style = {change: _shared_map_style(kind_fields, change, cmap)
             for change, kind_fields in fields.items()}
    levels = {change: colorbar_levels(s["vmin"], s["vmax"]) for change, s in style.items()}
    mean_format = {False: "{:.4g}", True: "{:+.3g}"}

    fig, axes = plt.subplots(*shape, figsize=(5.2 * shape[1], 2.9 * shape[0] + 1.6),
                             squeeze=False, subplot_kw={"projection": PROJECTION},
                             layout="constrained")
    meshes = {}
    for (i, j), panel in panels.items():
        ax = axes[i, j]
        meshes[panel.change] = draw_field(ax, panel.field, rasterized=True, **style[panel.change])
        if panel.pvalue is not None:
            contour_significant(ax, panel.field, panel.pvalue, levels[panel.change])
        draw_coastlines(ax)
        ax.set_global()
        mean_text = mean_format[panel.change].format(float(dl.global_mean(panel.field)))
        ax.set_title(f"{panel.title}\nglobal mean = {value_with_units(mean_text, units)}",
                     fontsize=9)
    for change, mesh in meshes.items():
        fig.colorbar(mesh, ax=axes, orientation="horizontal", shrink=0.5, pad=0.01,
                     aspect=40, ticks=levels[change]).set_label(labels[change])
    fig.suptitle(title, fontsize=11)
    _save_figure(fig, pdf=pdf, dpi=CASE_GRID_RASTER_DPI)

    ylims = {change: zonal_range(kind_fields) for change, kind_fields in fields.items()}
    plot_zonal_grid(
        {cell: ZonalPanel(panel.title, panel.field, ZONAL_NEUTRAL_COLOR,
                          ylims[panel.change], labels[panel.change])
         for cell, panel in panels.items()},
        shape, f"{title}\nzonal statistics over longitude (all cells, significant or not)", pdf,
    )


def plot_co2_amoc_decomposition(decomposition, name, units, cmap, title, pdf):
    """Map page (plus zonal page) of a ``regression.co2_amoc_decomposition`` result:
    one column per ``CO2_AMOC_COLUMNS`` level, headed by its Δlog2(CO2), ΔAMOC and the
    change in transformed AMOC (e.g. Δlog2(AMOC)), and four rows of changes from
    1xCO2: the ``CO2_AMOC_TERM_ROWS`` terms (contoured where their coefficient's
    p <= ``SIGNIFICANCE_P``) and their sum (the regression's nxCO2 - 1xCO2). All
    rows share one diverging scale (``plot_field_and_change_grid``)."""
    amoc = decomposition.attrs["amoc_transform"]
    panels = {}
    for j, co2 in enumerate(CO2_AMOC_COLUMNS):
        column = decomposition.sel(co2=co2)
        header = (f"{co2}xCO2: Δlog2(CO2) = {float(column['delta_log2_co2']):g}, "
                  f"ΔAMOC = {float(column['delta_amoc']):+.2f} Sv,\n"
                  f"Δ{amoc} = "
                  f"{float(column['delta_amoc_transformed']):+.3g}")
        for i, (param, row_title) in enumerate(CO2_AMOC_TERM_ROWS):
            panels[i, j] = MapPanel(row_title.format(amoc=amoc), column["contribution"].sel(param=param), True,
                                    column["pvalue"].sel(param=param))
        panels[0, j] = panels[0, j]._replace(title=f"{header}\n{panels[0, j].title}")
        panels[len(CO2_AMOC_TERM_ROWS), j] = MapPanel(
            CO2_AMOC_FITTED_CHANGE_TITLE.format(co2=co2), column["fitted_change"], True, None)
    plot_field_and_change_grid(panels, (len(CO2_AMOC_TERM_ROWS) + 1, len(CO2_AMOC_COLUMNS)),
                               name, units, cmap, title, pdf)


def plot_co2_amoc_reconstruction_check(actual, decomposition, name, units, cmap, title, pdf):
    """Map page (plus zonal page) checking the regression's CO2 response against the
    no-hosing runs ``actual`` (``(co2, lat, lon)``): one column per
    ``CO2_AMOC_COLUMNS`` level and rows simulated nxCO2 - 1xCO2, the regression's
    (``decomposition["fitted_change"]``), and simulated minus regression. All rows
    share one diverging scale (``plot_field_and_change_grid``)."""
    panels = {}
    for j, co2 in enumerate(CO2_AMOC_COLUMNS):
        simulated = actual.sel(co2=co2) - actual.sel(co2=1)
        regression = decomposition["fitted_change"].sel(co2=co2)
        panels[0, j] = MapPanel(CO2_AMOC_SIMULATED_CHANGE_TITLE.format(co2=co2),
                                simulated, True, None)
        panels[1, j] = MapPanel(CO2_AMOC_FITTED_CHANGE_TITLE.format(co2=co2),
                                regression, True, None)
        panels[2, j] = MapPanel(CO2_AMOC_CHANGE_RESIDUAL_TITLE.format(co2=co2),
                                simulated - regression, True, None)
    plot_field_and_change_grid(panels, (3, len(CO2_AMOC_COLUMNS)), name, units, cmap, title, pdf)


def plot_case_grid_zonal_page(field, title, pdf):
    """One page of ``pdf``: the zonal statistics of ``field`` (``(co2, hosing, lat,
    lon)``) as a 3 x 3 grid laid out like ``data_loader.CASE_GRID``, plus a tall
    right-hand panel overlaying all nine zonal means.

    Grid panels take their case's ``HOSING_COLOR`` and share one y range; the
    overlay styles each case with ``case_line_style`` and has its own y range,
    fitted to the means alone.
    """
    fig = plt.figure(figsize=(19, 9.5), layout="constrained")
    gridspec = fig.add_gridspec(len(dl.CO2_LEVELS), len(dl.HOSING_LEVELS) + 1,
                                width_ratios=[1] * len(dl.HOSING_LEVELS) + [1.3])
    ylim = zonal_range([field])
    for i, co2 in enumerate(dl.CO2_LEVELS):
        for j, hosing in enumerate(dl.HOSING_LEVELS):
            ax = fig.add_subplot(gridspec[i, j])
            draw_zonal_statistics(ax, field.sel(co2=co2, hosing=hosing), HOSING_COLOR[hosing])
            ax.set_ylim(ylim)
            ax.set_title(dl.CASE_GRID[i][j], fontsize=10)
    means = field.mean("lon")
    overlay = fig.add_subplot(gridspec[:, -1])
    for i, co2 in enumerate(dl.CO2_LEVELS):
        for j, hosing in enumerate(dl.HOSING_LEVELS):
            case = dl.CASE_GRID[i][j]
            overlay.plot(means["lat"].values, means.sel(co2=co2, hosing=hosing).values,
                         lw=1.5, label=case, **case_line_style(case))
    overlay.axhline(0, color="0.5", lw=0.5)
    set_sine_latitude_axis(overlay)
    overlay.set_ylim(zonal_range([means]))
    overlay.tick_params(axis="y", labelsize=7)
    overlay.grid(alpha=0.3)
    overlay.set_title("zonal means, all cases (own y range)", fontsize=10)
    overlay.legend(fontsize=8, handlelength=3)
    fig.supylabel(label_with_units(field.name, field.attrs["units"]))
    fig.legend(handles=ZONAL_LEGEND_HANDLES, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{title}\nzonal statistics over longitude", fontsize=12)
    _save_figure(fig, pdf=pdf)


def projected_latitude(lat):
    """y coordinate of latitude ``lat`` (deg) in ``PROJECTION`` (Equal Earth y is
    not proportional to sin(lat)), so a strip beside a map lines up with it."""
    lat = np.asarray(lat, dtype=float)
    return PROJECTION.transform_points(DATA_CRS, np.zeros_like(lat), lat)[:, 1]


# Thin land / ocean zonal-mean lines on zonal strips, colored to tell them from the
# black all-longitude mean.
LAND_OCEAN_STYLE = {
    "land": {"color": "forestgreen", "lw": 0.9},
    "ocean": {"color": "royalblue", "lw": 0.9},
}


def land_ocean_zonal_means(field, land):
    """Zonal means of a ``(lat, lon)`` field over all longitudes, land cells, and
    ocean cells (``land``: boolean mask from ``data_loader.land_mask``), as a dict
    keyed ``all`` / ``land`` / ``ocean``. A latitude with no land (or no ocean)
    gives NaN, a gap in its line."""
    return {"all": field.mean("lon"), "land": field.where(land).mean("lon"),
            "ocean": field.where(~land).mean("lon")}


def draw_zonal_strip(ax, means, band, ylim, xlim, units, band_color=ZONAL_NEUTRAL_COLOR):
    """Zonal strip drawn sideways on ``ax`` against the ``PROJECTION`` y of each
    latitude, so that with ``ylim`` = the neighbouring map's y range the latitudes
    line up: the shaded ``band`` (a ``(low, high)`` pair of ``lat`` DataArrays) in
    ``band_color``, the all-longitude mean (thick black) and thin land / ocean
    means, from ``means`` (a ``land_ocean_zonal_means``-style dict; only the
    ``LAND_OCEAN_STYLE`` surfaces it holds are drawn, so ``{"all": ...}`` alone
    draws just the mean, as for a land-only field)."""
    y = projected_latitude(means["all"]["lat"].values)
    low, high = band
    ax.fill_betweenx(y, low.values, high.values, color=band_color, **ZONAL_STYLE["band"])
    for surface in LAND_OCEAN_STYLE.keys() & means.keys():
        ax.plot(means[surface].values, y, **LAND_OCEAN_STYLE[surface])
    ax.plot(means["all"].values, y, color=ZONAL_NEUTRAL_COLOR, **ZONAL_STYLE["mean"])
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_xlim(xlim)
    _strip_latitude_axis(ax, ylim, units)


def _strip_latitude_axis(ax, ylim, xlabel):
    """Projected-latitude y axis (ticks on the right) of a strip beside a map whose
    y range is ``ylim``, with x label ``xlabel``."""
    ax.set_ylim(ylim)
    ax.set_yticks(projected_latitude(SINE_LATITUDE_TICKS),
                  [latitude_label(t) for t in SINE_LATITUDE_TICKS], fontsize=7)
    ax.yaxis.tick_right()
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(alpha=0.3)


# Ratio panels (AMOC-weakening effect / CO2 effect): the color scale is fixed at
# -1..+1, on a colormap distinct from the difference maps' to flag the different
# (dimensionless) quantity. Beyond it, cells take colors absent from that colormap:
# RATIO_UNDER_COLOR where the ratio is < -1 (AMOC weakening more than cancels the
# CO2 effect) and RATIO_OVER_COLOR where it is > +1 (more than doubles it). The
# strip spans +/-RATIO_STRIP_LIMIT, faintly shaded in those colors beyond +/-1. Its
# band is a confidence interval, not a spread over longitude, so it is shaded in
# its own color. Cells where the ratio is not meaningful are masked in
# RATIO_MASK_COLOR (not contoured): the CO2 effect (the divisor) is not significant,
# or is significant but below RATIO_MIN_DIVISOR_FRACTION of its area-weighted median
# |CO2 effect| (a near-zero divisor; for tas, no cell is that small). Cells where only
# the AMOC effect is not significant stay shown: their ratio is near zero, meaning
# AMOC weakening has no detectable effect there.
RATIO_BOUND = 1.0
RATIO_UNDER_COLOR = "teal"
RATIO_OVER_COLOR = "crimson"
RATIO_MASK_COLOR = "lightgrey"
RATIO_MIN_DIVISOR_FRACTION = 0.1
RATIO_CMAP = plt.colormaps["PuOr_r"].with_extremes(under=RATIO_UNDER_COLOR,
                                                   over=RATIO_OVER_COLOR,
                                                   bad=RATIO_MASK_COLOR)
RATIO_STRIP_LIMIT = 2.0
RATIO_STRIP_SHADE_ALPHA = 0.12
RATIO_BAND_COLOR = "darkorchid"
RATIO_CONFIDENCE = 0.95

ZONAL_STRIP_LEGEND_HANDLES = [
    ZONAL_LEGEND_HANDLES[0],
    *(Line2D([], [], label=f"{surface} mean", **style) for surface, style in LAND_OCEAN_STYLE.items()),
    ZONAL_LEGEND_HANDLES[2],
    Patch(facecolor=RATIO_BAND_COLOR, label=f"{100 * RATIO_CONFIDENCE:.0f}% CI of ratio",
          **ZONAL_STYLE["band"]),
]


def _map_with_strip(fig, map_cell, strip_cell, field, title, cmap, bound, pvalue,
                    strip_means, strip_band, strip_xlim, units, colorbar_label,
                    band_color=ZONAL_NEUTRAL_COLOR):
    """Map of ``field`` on ``±bound`` with its own colorbar (``colorbar_label``;
    ``units`` labels the strip), with significance contours at its colorbar ticks
    (``contour_significant``; none for ``pvalue=None``), and a zonal strip (``draw_zonal_strip``)
    to its right. NaN cells take ``cmap``'s "bad" color."""
    ax = fig.add_subplot(map_cell, projection=PROJECTION)
    mesh = draw_field(ax, field, cmap=cmap, vmin=-bound, vmax=bound, rasterized=True)
    levels = colorbar_levels(-bound, bound)
    if pvalue is not None:
        contour_significant(ax, field, pvalue, levels)
    draw_coastlines(ax)
    ax.set_global()
    ax.set_title(title, fontsize=9)
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.8, pad=0.03,
                        aspect=30, extend="both", ticks=levels)
    cbar.set_label(colorbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    draw_zonal_strip(fig.add_subplot(strip_cell), strip_means, strip_band, ax.get_ylim(),
                     strip_xlim, units, band_color)


def plot_co2_amoc_differences(fields, pvalues, ratio_conf_int, out_path):
    """One-page 2 x 2 PDF decomposing the 4xCO2 response with and without AMOC
    weakening, each map with a zonal strip (``draw_zonal_strip``).

    ``fields`` is a ``data_loader.case_annual_fields`` result for
    ``data_loader.CO2_AMOC_CASES``; panels are differences of its time means:

        4xCO2 - 1xCO2 (total)              4xCO2 - 4xCO2_m03Sv (AMOC effect)
        4xCO2_m03Sv - 1xCO2 (CO2 effect)   AMOC effect / CO2 effect

    ``pvalues`` maps ``total`` / ``amoc_effect`` / ``co2_effect`` to ``(lat, lon)``
    p-values of those differences; each difference map is contoured at its colorbar
    ticks where its difference is significant (p <= ``SIGNIFICANCE_P``). The ratio map
    is masked where the CO2 effect is not significant or is small (see
    ``RATIO_MIN_DIVISOR_FRACTION``) and contoured where the AMOC effect is
    significant, so its contoured cells have both effects significant. The three
    differences share one symmetric bound (99th percentile of |difference|;
    ``RdBu`` for water fluxes, else ``RdBu_r``) and one strip range, with bands
    spanning 5-95% of cells over longitude. The ratio has its own fixed -1..+1
    scale with distinct colors beyond each end (see ``RATIO_CMAP``); its strip
    lines are ratios of zonal means and its
    band is ``ratio_conf_int`` (``low`` / ``high`` on ``lat``, from
    ``regression.difference_ratio_conf_int`` of the zonal means). Titles give
    area-weighted global means (for the ratio, the ratio of global means).
    """
    units = fields.attrs["units"]
    land = dl.land_mask(fields["lat"], fields["lon"])
    means = fields.mean("year")
    differences = {
        "total": means.sel(case="4xCO2") - means.sel(case="1xCO2"),
        "amoc_effect": means.sel(case="4xCO2") - means.sel(case="4xCO2_m03Sv"),
        "co2_effect": means.sel(case="4xCO2_m03Sv") - means.sel(case="1xCO2"),
    }
    divisor_floor = RATIO_MIN_DIVISOR_FRACTION * float(dl.area_weighted_median(
        abs(differences["co2_effect"])))
    ratio_meaningful = ((pvalues["co2_effect"] <= SIGNIFICANCE_P)
                        & (abs(differences["co2_effect"]) >= divisor_floor))
    ratio = (differences["amoc_effect"] / differences["co2_effect"]).where(ratio_meaningful)

    strip_means = {name: land_ocean_zonal_means(d, land) for name, d in differences.items()}
    ratio_means = {surface: strip_means["amoc_effect"][surface] / strip_means["co2_effect"][surface]
                   for surface in strip_means["amoc_effect"]}
    zonal = {name: dl.zonal_statistics(d) for name, d in differences.items()}
    bands = {name: (z["low"], z["high"]) for name, z in zonal.items()}
    diff_bound = symmetric_bound(np.stack([d.values for d in differences.values()]))
    diff_cmap = "RdBu" if units == dl.WATER_FLUX_UNITS else "RdBu_r"
    diff_xlim = zonal_range([edge for band in bands.values() for edge in band]
                            + [m for d in strip_means.values() for m in d.values()])

    def mean_text(field):
        return value_with_units(f"{float(dl.global_mean(field)):+.3g}", units)

    titles = {
        "total": "4xCO2 − 1xCO2 (total 4×CO₂ response)",
        "amoc_effect": "4xCO2 − 4xCO2_m03Sv (AMOC-weakening effect)",
        "co2_effect": "4xCO2_m03Sv − 1xCO2 (4×CO₂ effect, AMOC ~20 Sv)",
    }
    positions = {"total": (0, 0), "amoc_effect": (0, 1), "co2_effect": (1, 0)}

    fig = plt.figure(figsize=(15, 9.5), layout="constrained")
    gridspec = fig.add_gridspec(2, 4, width_ratios=[4.5, 1, 4.5, 1])
    for name, (row, col) in positions.items():
        _map_with_strip(fig, gridspec[row, 2 * col], gridspec[row, 2 * col + 1],
                        differences[name],
                        f"{titles[name]}\nglobal mean = {mean_text(differences[name])}",
                        diff_cmap, diff_bound, pvalues[name], strip_means[name], bands[name],
                        diff_xlim, units, units)
    global_ratio = float(dl.global_mean(differences["amoc_effect"])
                         / dl.global_mean(differences["co2_effect"]))
    _map_with_strip(fig, gridspec[1, 2], gridspec[1, 3], ratio,
                    "(4xCO2 − 4xCO2_m03Sv) / (4xCO2_m03Sv − 1xCO2)\n"
                    f"ratio of global means = {global_ratio:+.3g}\n"
                    f"grey: CO₂ effect not significant, or |CO₂ effect| < "
                    f"{RATIO_MIN_DIVISOR_FRACTION:.0%} of its median ({divisor_floor:.2g} {units})",
                    RATIO_CMAP, RATIO_BOUND, pvalues["amoc_effect"], ratio_means,
                    (ratio_conf_int["low"], ratio_conf_int["high"]),
                    (-RATIO_STRIP_LIMIT, RATIO_STRIP_LIMIT), "ratio",
                    f"ratio ({RATIO_UNDER_COLOR} < −1: AMOC weakening more than cancels the CO₂ "
                    f"effect; {RATIO_OVER_COLOR} > +1: more than doubles it)",
                    band_color=RATIO_BAND_COLOR)
    ratio_strip = fig.axes[-1]
    ratio_strip.axvspan(-RATIO_STRIP_LIMIT, -RATIO_BOUND, color=RATIO_UNDER_COLOR,
                        alpha=RATIO_STRIP_SHADE_ALPHA, lw=0)
    ratio_strip.axvspan(RATIO_BOUND, RATIO_STRIP_LIMIT, color=RATIO_OVER_COLOR,
                        alpha=RATIO_STRIP_SHADE_ALPHA, lw=0)
    fig.legend(handles=ZONAL_STRIP_LEGEND_HANDLES, loc="outside lower right", ncol=5,
               fontsize=8, title="right strips: zonal means; grey band = spread over longitude; "
               "ratio strip: ratios of zonal means, purple band = CI of the all-longitude ratio "
               "(delta method, 10-year means); land = Natural Earth 110m", title_fontsize=8)
    fig.suptitle(f"{fields.name}: {label_with_units(fields.attrs['long_name'], units)}, "
                 f"CESM1.2 means {fields.attrs['time_mean']}\n"
                 f"significance of differences: Welch t-test on 10-year means; "
                 f"{SIGNIFICANCE_CONTOUR_KEY}",
                 fontsize=11)
    _save_figure(fig, out_path=out_path, dpi=CASE_GRID_RASTER_DPI)


def plot_case_grid_book(grid, pdf, rasterized):
    """Append the four ``CASE_GRID_PAGES`` for one variable to ``pdf``, each map
    page followed by its zonal-statistics page (``plot_case_grid_zonal_page``).

    ``grid`` is a ``data_loader.case_grid_time_mean`` result. The raw page uses
    ``viridis`` scaled to the 1st-99th percentile over all nine panels; the
    difference pages use a diverging map with symmetric bounds (±99th percentile
    of |difference| over the page; white = 0): ``RdBu`` (wet = blue) for water
    fluxes (``data_loader.WATER_FLUX_UNITS``), else ``RdBu_r``.
    """
    header = (f"{grid.name}: {label_with_units(grid.attrs['long_name'], grid.attrs['units'])}, "
              f"CESM1.2 mean {grid.attrs['time_mean']}")
    diverging = "RdBu" if grid.attrs["units"] == dl.WATER_FLUX_UNITS else "RdBu_r"
    (raw_label, _), *difference_pages = CASE_GRID_PAGES
    low, high = np.nanpercentile(grid.values, [1, 99])
    plot_case_grid_page(grid, f"{header}\n{raw_label}", "viridis", low, high,
                        pdf, rasterized)
    plot_case_grid_zonal_page(grid, f"{header}\n{raw_label}", pdf)
    for label, transform in difference_pages:
        field = transform(grid).assign_attrs(grid.attrs).rename(grid.name)
        bound = symmetric_bound(field.values)
        plot_case_grid_page(field, f"{header}\n{label}", diverging, -bound, bound,
                            pdf, rasterized)
        plot_case_grid_zonal_page(field, f"{header}\n{label}", pdf)


# Land-only CO2 / AMOC metric pages (``plot_co2_amoc_metrics``). Ocean cells are missing data, drawn in
# OCEAN_COLOR: a pale blue-grey, distinct from the white zero of the diverging maps
# and from the grey of masked land. The two exacerbate / ameliorate metrics (E and
# R) share METRIC_CMAP colors, purple where AMOC slowdown exacerbates the CO2 change
# and green where it ameliorates it; overcompensation (R < -1) is
# OVERCOMPENSATION_COLOR, a color outside that map. The three-part maps blend
# FRACTION_COLORS (following ``cam.FRACTION_NAMES``: AMOC orange, CO2 blue, scale
# white) in proportion to the ``cam.component_fractions``; their key triangle sits
# in the strip slot, where the other maps have their zonal strip.
OCEAN_COLOR = "#dde6ee"
METRIC_CMAP = plt.colormaps["PRGn_r"].with_extremes(bad=OCEAN_COLOR)
SCALE_CMAP = plt.colormaps["viridis"].with_extremes(bad=OCEAN_COLOR)
OVERCOMPENSATION_COLOR = "gold"
AMOC_COLOR = "#e66101"  # AMOC dominates, in the three-part maps; change only with AMOC, on R
FRACTION_COLORS = [AMOC_COLOR, "#2166ac", "white"]
# The ratio R = A / C is drawn on -EFFECT_RATIO_BOUND (cancels) .. +EFFECT_RATIO_BOUND
# (doubles), green to purple; below -1 (overcompensates) cells are
# OVERCOMPENSATION_COLOR, above +1 the darkest purple. Its strip spans
# +-EFFECT_RATIO_STRIP_LIMIT, shaded in OVERCOMPENSATION_COLOR below -1. Land cells
# where R is not shown are AMOC_COLOR where the CO2 effect is not substantive but
# the total is (a change made only with AMOC slowdown, ``paint_cells``), and
# EFFECT_RATIO_MASK_COLOR, a grey between the categories' "not substantive" grey
# and the coastlines, where neither is; ``mask_ocean`` then paints the ocean.
EFFECT_RATIO_BOUND = 1.0
EFFECT_RATIO_STRIP_LIMIT = 2.0
EFFECT_RATIO_STRIP_SHADE_ALPHA = 0.3
EFFECT_RATIO_MASK_COLOR = "0.6"
EFFECT_RATIO_CMAP = plt.colormaps["PRGn_r"].with_extremes(under=OVERCOMPENSATION_COLOR,
                                                          bad=EFFECT_RATIO_MASK_COLOR)
TERNARY_KEY_PIXELS = 300
TERNARY_POINT_STYLE = {"s": 1.5, "c": "black", "alpha": 0.25, "lw": 0}
LAND_STRIP_LEGEND_HANDLES = [
    Line2D([], [], color=ZONAL_NEUTRAL_COLOR, label="land zonal mean", **ZONAL_STYLE["mean"]),
    Patch(facecolor=ZONAL_NEUTRAL_COLOR, label="{:.0f}–{:.0f}% of land cells".format(
        *(100 * q for q in dl.ZONAL_BAND_QUANTILES)), **ZONAL_STYLE["band"]),
    Patch(facecolor=OCEAN_COLOR, edgecolor="0.5", lw=0.5, label="missing: ocean, or σ = 0"),
]
# Metric page grid: columns of (map, strip) pairs in METRIC_WIDTH_RATIOS, each row
# METRIC_ROW_HEIGHT inches tall.
METRIC_COLUMNS = 3
METRIC_WIDTH_RATIOS = [4.5, 1] * METRIC_COLUMNS
METRIC_FIGURE_WIDTH = 24.0
METRIC_ROW_HEIGHT = 5.2
# Effects in page order: dataset name stem, symbol.
EFFECT_SYMBOLS = {"co2_effect": "C", "amoc_effect": "A", "total": "T"}


def _land_map_with_strip(fig, map_cell, strip_cell, field, title, cmap, vmin, vmax, pvalue,
                         strip_xlim, colorbar_label, strip_label, extend="both"):
    """Map of the land-only ``field`` on [``vmin``, ``vmax``] with its colorbar
    (arrows per ``extend``), significance contours at its ticks (none for
    ``pvalue=None``), and a strip of its land zonal mean and 5-95% band over land
    longitudes (``draw_zonal_strip``). Returns the map and strip axes."""
    ax = fig.add_subplot(map_cell, projection=PROJECTION)
    mesh = draw_field(ax, field, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    levels = colorbar_levels(vmin, vmax)
    if pvalue is not None:
        contour_significant(ax, field, pvalue, levels)
    draw_coastlines(ax)
    ax.set_global()
    ax.set_title(title, fontsize=9)
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.8, pad=0.03,
                        aspect=30, extend=extend, ticks=levels)
    cbar.set_label(colorbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    zonal = dl.zonal_statistics(field)
    strip = fig.add_subplot(strip_cell)
    draw_zonal_strip(strip, {"all": zonal["mean"]}, (zonal["low"], zonal["high"]),
                     ax.get_ylim(), strip_xlim, strip_label)
    return ax, strip


def _land_row(fig, cells, panels, cmap, vmin, vmax, extend="both"):
    """One row of ``_land_map_with_strip`` maps sharing [``vmin``, ``vmax``] (colorbar
    arrows per ``extend``) and one strip range (``cells``: (map, strip) pairs;
    ``panels``: (field, title, pvalue, label) per map)."""
    zonal = [dl.zonal_statistics(field) for field, *_ in panels]
    strip_xlim = zonal_range([z[key] for z in zonal for key in ("low", "high")])
    for (map_cell, strip_cell), (field, title, pvalue, label) in zip(cells, panels):
        _land_map_with_strip(fig, map_cell, strip_cell, field, title, cmap, vmin, vmax, pvalue,
                             strip_xlim, label, label, extend)


def _land_mean(field, units):
    """Title line with the area-weighted land mean of ``field``."""
    return f"land mean = {value_with_units(f'{float(dl.global_mean(field)):+.3g}', units)}"


def _effect_cmap(units):
    """Diverging map of the effects: ``RdBu`` for water fluxes (blue = wetter),
    else ``RdBu_r`` (red = increase)."""
    return plt.colormaps["RdBu" if units == dl.WATER_FLUX_UNITS else "RdBu_r"].with_extremes(
        bad=OCEAN_COLOR)


def _effect_definitions():
    """Title text defining C, A and T by case labels, keyed like ``EFFECT_SYMBOLS``."""
    c1, c4, c4slow = (cam.CASE_LABELS[case] for case in dl.CO2_AMOC_CASES)
    return {"co2_effect": f"C = {c4} − {c1} (CO₂ effect, AMOC ~20 Sv)",
            "amoc_effect": f"A = {c4slow} − {c4} (AMOC-slowdown effect)",
            "total": f"T = A + C = {c4slow} − {c1} (total response)"}


def _diverging_row(fig, cells, fields, titles, pvalues, units, labels):
    """``_land_row`` of three effect maps on ``_effect_cmap`` with one symmetric bound."""
    bound = symmetric_bound(np.stack([field.values for field in fields]))
    _land_row(fig, cells, list(zip(fields, titles, pvalues, labels)), _effect_cmap(units),
              -bound, bound)


def _sd_scaled_effect_row(fig, cells, metrics):
    """C / sigma, A / sigma, T / sigma of a ``cam.land_metrics`` result on one
    symmetric bound, each contoured by its own significance."""
    stems = {"co2_effect": "co2", "amoc_effect": "amoc", "total": "total"}
    fields = [metrics[f"{stems[name]}_scaled"] for name in EFFECT_SYMBOLS]
    definitions = _effect_definitions()
    _diverging_row(
        fig, cells, fields,
        [f"{effect} / σ, {definitions[name]}\n{_land_mean(field, 'σ')}"
         for (name, effect), field in zip(EFFECT_SYMBOLS.items(), fields)],
        [metrics[f"{name}_pvalue"] for name in EFFECT_SYMBOLS], metrics.attrs["units"],
        [f"{effect} / σ (σ)" for effect in EFFECT_SYMBOLS.values()])


def _departure_panel(fig, cell, metrics):
    """E = (|T| - |C|) / sigma on its own symmetric bound, contoured by A's significance."""
    c1 = cam.CASE_LABELS[cam.BASELINE_CASE]
    departure = metrics["departure_change"]
    bound = symmetric_bound(departure.values)
    _land_map_with_strip(
        fig, *cell, departure,
        f"E = (|T| − |C|) / σ: > 0, AMOC slowdown moves the\nclimate farther from "
        f"{c1} (exacerbates); < 0, closer (ameliorates)\n{_land_mean(departure, 'σ')}",
        METRIC_CMAP, -bound, bound, metrics["amoc_effect_pvalue"], zonal_range([departure]),
        "E (σ)", "E (σ)")


def paint_cells(ax, cells, color):
    """Paint the cells where the boolean ``(lat, lon)`` ``cells`` is True in
    ``color``, over the fields drawn so far on map ``ax`` and under the coastlines."""
    draw_field(ax, xr.zeros_like(cells, dtype=float).where(cells),
               cmap=mcolors.ListedColormap([color]).with_extremes(bad=(0, 0, 0, 0)),
               vmin=0, vmax=1, rasterized=True)


def mask_ocean(ax, land):
    """Paint the ocean cells (``~land``) of map ``ax`` in ``OCEAN_COLOR``, so that a
    map can use its own "bad" color for masked land cells."""
    paint_cells(ax, ~land, OCEAN_COLOR)


def fraction_colors(fractions):
    """RGB ``(lat, lon, rgb)`` blend of ``FRACTION_COLORS`` weighted by the
    ``cam.FRACTION_NAMES`` variables of ``fractions``; NaN (ocean) cells take
    ``OCEAN_COLOR``."""
    rgb = sum(fractions[name] * xr.DataArray(np.array(mcolors.to_rgb(color)), dims="rgb")
              for name, color in zip(cam.FRACTION_NAMES, FRACTION_COLORS))
    ocean = xr.DataArray(np.array(mcolors.to_rgb(OCEAN_COLOR)), dims="rgb")
    # The float32 fractions can sum to 1 + 1e-7, pushing white just past 1.
    return rgb.fillna(ocean).clip(0, 1).transpose("lat", "lon", "rgb")


def barycentric_xy(fractions):
    """Key-triangle coordinates of the ``cam.FRACTION_NAMES`` shares: AMOC at the
    lower left (0, 0), CO2 at the lower right (1, 0), the scale at the top."""
    amoc, co2, scale = (np.asarray(fractions[name]) for name in cam.FRACTION_NAMES)
    return co2 + 0.5 * scale, np.sqrt(3) / 2 * scale


def draw_ternary_key(ax, fractions, symbol):
    """Color key of the three-part maps: the triangle of ``FRACTION_COLORS`` blends,
    with each land cell of ``fractions`` as a dot."""
    x, y = np.meshgrid(np.linspace(0, 1, TERNARY_KEY_PIXELS),
                       np.linspace(0, np.sqrt(3) / 2, TERNARY_KEY_PIXELS))
    scale = y / (np.sqrt(3) / 2)
    co2 = x - 0.5 * scale
    amoc = 1 - co2 - scale
    inside = (amoc >= 0) & (co2 >= 0)
    key = fraction_colors(xr.Dataset({
        name: (("lat", "lon"), np.where(inside, share, np.nan))
        for name, share in zip(cam.FRACTION_NAMES, (amoc, co2, scale))}))
    ax.imshow(np.dstack([key.values, inside.astype(float)]), origin="lower",
              extent=(0, 1, 0, np.sqrt(3) / 2))
    land_cells = fractions[cam.FRACTION_NAMES].stack(cell=("lat", "lon")).dropna("cell")
    ax.scatter(*barycentric_xy(land_cells), **TERNARY_POINT_STYLE)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color="black", lw=0.6)
    ax.text(0, -0.05, "|A|\nAMOC", ha="center", va="top", fontsize=7)
    ax.text(1, -0.05, "|C|\nCO₂", ha="center", va="top", fontsize=7)
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, symbol, ha="center", va="bottom", fontsize=8)
    ax.text(0.5, -0.36, "key; dots =\nland cells", ha="center", va="top", fontsize=6.5)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.42, np.sqrt(3) / 2 + 0.12)
    ax.set_aspect("equal")
    ax.axis("off")


def _fraction_panel(fig, map_cell, key_cell, scaled, symbol, summary):
    """Three-part map of the ``cam.component_fractions`` in ``scaled`` with its key
    triangle beside it in the strip slot ``key_cell``. The title carries the
    land-wide ``summary`` (``cam.land_summary``)."""
    ax = fig.add_subplot(map_cell, projection=PROJECTION)
    draw_field(ax, fraction_colors(scaled), rasterized=True)
    draw_coastlines(ax)
    ax.set_global()
    ax.set_title(f"|A|, |C| and {symbol} as shares of {symbol} + |A| + |C|\n"
                 f"orange: AMOC slowdown, blue: CO₂, white: within the {symbol} scale\n"
                 f"land-wide: RMS(A) / RMS(C) = {summary['rms_ratio']:.3g}; A vs C pattern\n"
                 f"congruence {summary['pattern_congruence']:+.2f} (uncentered), "
                 f"correlation {summary['pattern_correlation']:+.2f}", fontsize=9)
    draw_ternary_key(fig.add_subplot(key_cell), scaled, symbol)


def _metric_figure(rows):
    """Figure with ``rows`` rows of ``METRIC_COLUMNS`` (map, strip) pairs,
    and a ``cell(row, col)`` function giving a pair."""
    fig = plt.figure(figsize=(METRIC_FIGURE_WIDTH, METRIC_ROW_HEIGHT * rows + 1.5),
                     layout="constrained")
    gridspec = fig.add_gridspec(rows, 2 * METRIC_COLUMNS, width_ratios=METRIC_WIDTH_RATIOS)

    def cell(row, col):
        return gridspec[row, 2 * col], gridspec[row, 2 * col + 1]

    return fig, cell


def _finish_metric_page(fig, data, title_note, significance_note, out_path):
    """Strip legend, suptitle (variable, window, cases, ``title_note``,
    ``significance_note``) and save."""
    c1, c4, c4slow = (cam.CASE_LABELS[case] for case in dl.CO2_AMOC_CASES)
    units = data.attrs["units"]
    fig.legend(handles=LAND_STRIP_LEGEND_HANDLES, loc="outside lower left", ncol=3, fontsize=8,
               title="right strips: statistics over land longitudes at each latitude; "
               "land = Natural Earth 110m", title_fontsize=8)
    fig.suptitle(f"{data.attrs['name']}: {label_with_units(data.attrs['long_name'], units)}, "
                 f"CESM1.2 means {data.attrs['time_mean']}, land only — "
                 f"{c1} = 1xCO2, {c4} = 4xCO2_m03Sv, {c4slow} = 4xCO2\n{title_note}\n"
                 f"significance: Welch t-test on 10-year means ({significance_note}); "
                 f"{SIGNIFICANCE_CONTOUR_KEY}",
                 fontsize=12)
    _save_figure(fig, out_path=out_path, dpi=CASE_GRID_RASTER_DPI)


def plot_co2_amoc_metrics(metrics, summary, out_path):
    """One-page 3 x 3 PDF of land-only metrics of whether AMOC slowdown exacerbates
    or ameliorates the 4xCO2 response (``metrics``: a ``cam.land_metrics`` result; C,
    A, T, sigma, E and R as defined in ``co2_amoc_metrics``; ``summary``:
    ``cam.land_summary`` of its effects):

        C             A             T
        C / sigma     A / sigma     T / sigma
        E             R = A / C     |A|, |C|, sigma shares + key triangle

    Rows 0 and 1 each share one symmetric bound; the effects are contoured by their
    own significance, E and R by A's. R has a fixed
    -1..+1 scale (``EFFECT_RATIO_BOUND``; ``OVERCOMPENSATION_COLOR`` below -1; where
    the CO2 effect is not substantive, ``AMOC_COLOR`` if the total is and grey if
    not). Every map but the three-part one carries a strip of land zonal statistics.
    """
    units = metrics.attrs["units"]
    definitions = _effect_definitions()
    fig, cell = _metric_figure(3)
    row_cells = [[cell(row, col) for col in range(METRIC_COLUMNS)] for row in range(3)]

    _diverging_row(
        fig, row_cells[0], [metrics[name] for name in EFFECT_SYMBOLS],
        [f"{definitions[name]}\n{_land_mean(metrics[name], units)}" for name in EFFECT_SYMBOLS],
        [metrics[f"{name}_pvalue"] for name in EFFECT_SYMBOLS], units,
        [label_with_units(effect, units) for effect in EFFECT_SYMBOLS.values()])

    _sd_scaled_effect_row(fig, row_cells[1], metrics)
    _departure_panel(fig, row_cells[2][0], metrics)

    ratio = metrics["effect_ratio"]
    threshold = metrics.attrs["threshold_sd"]
    overcompensating = cam.land_fraction(ratio < -1, metrics["land"])
    amoc_made = metrics["change_only_with_amoc"] == 1
    ax, strip = _land_map_with_strip(
        fig, *row_cells[2][1], ratio,
        "R = A / C: AMOC effect as a fraction of the CO₂ effect\n"
        "−1: AMOC slowdown cancels the CO₂ effect; −0.5: halves it; +1: doubles it\n"
        f"gold, R < −1: overcompensates ({overcompensating:.1%} of land); "
        f"median R = {float(dl.area_weighted_median(ratio)):+.2f}\n"
        f"orange: C not substantive but T is, a change only with AMOC slowdown "
        f"({cam.land_fraction(amoc_made, metrics['land']):.1%})\n"
        f"grey: neither C nor T substantive (≥ {threshold:g}σ and p ≤ {cam.SIGNIFICANCE_P})",
        EFFECT_RATIO_CMAP, -EFFECT_RATIO_BOUND, EFFECT_RATIO_BOUND, metrics["amoc_effect_pvalue"],
        (-EFFECT_RATIO_STRIP_LIMIT, EFFECT_RATIO_STRIP_LIMIT), "R", "R")
    strip.axvspan(-EFFECT_RATIO_STRIP_LIMIT, -EFFECT_RATIO_BOUND, color=OVERCOMPENSATION_COLOR,
                  alpha=EFFECT_RATIO_STRIP_SHADE_ALPHA, lw=0)
    paint_cells(ax, amoc_made, AMOC_COLOR)
    mask_ocean(ax, metrics["land"])

    _fraction_panel(fig, *row_cells[2][2], metrics, "σ", summary)
    _finish_metric_page(
        fig, metrics,
        f"σ: interannual SD of {cam.CASE_LABELS[cam.BASELINE_CASE]} annual means "
        f"({metrics.attrs['sd_years']}, linearly detrended), the scale of natural variability",
        "C, A, T: own effect; E, R: A", out_path)


# Summary table page closing the CO2 / AMOC metric book
# (``plot_co2_amoc_summary_table``) and its xlsx twin (``write_co2_amoc_summary_xlsx``):
# one row per variable, the ``cam.SUMMARY_COLUMNS`` as columns. Each column has a
# page header, a Python and an Excel number format, and a width in table units;
# SUMMARY_TABLE_GROUPS label runs of adjacent columns.
SummaryColumn = namedtuple("SummaryColumn", "header python_format excel_format width")
_SHARE = ("{:.1%}", "0.0%")
SUMMARY_TABLE_COLUMNS = {
    "description": SummaryColumn("description", "{}", "@", 3.6),
    cam.CLASS_NAMES[0]: SummaryColumn("AMOC effect\nnot substantive", *_SHARE, 1.15),
    cam.CLASS_NAMES[1]: SummaryColumn("exacerbates", *_SHARE, 1.0),
    cam.CLASS_NAMES[2]: SummaryColumn("partly\noffsets", *_SHARE, 1.0),
    cam.CLASS_NAMES[3]: SummaryColumn("over-\ncompensates", *_SHARE, 1.05),
    cam.CLASS_NAMES[4]: SummaryColumn("AMOC-only\n(C not\nsubstantive)", *_SHARE, 1.0),
    "land_fraction_R_below_minus1": SummaryColumn("R < −1\n(gold)", *_SHARE, 0.9),
    "land_fraction_change_only_with_amoc": SummaryColumn("only T\nsubstantive\n(orange)",
                                                         *_SHARE, 1.0),
    "land_median_R": SummaryColumn("median R", "{:+.2f}", "+0.00;-0.00;0.00", 0.9),
    "land_mean_E_in_sd": SummaryColumn("land-mean\nE (σ)", "{:+.3f}", "+0.000;-0.000;0.000",
                                       0.95),
    "rms_ratio": SummaryColumn("RMS(A) /\nRMS(C)", "{:.3f}", "0.000", 0.9),
    "pattern_congruence": SummaryColumn("pattern\ncongruence", "{:+.2f}", "+0.00;-0.00;0.00",
                                        1.0),
    "pattern_correlation": SummaryColumn("pattern\ncorrelation", "{:+.2f}", "+0.00;-0.00;0.00",
                                         1.0),
}
SUMMARY_TABLE_GROUPS = [
    ("categories: share of land area", cam.CLASS_NAMES[0], cam.CLASS_NAMES[4]),
    ("R map: share of land", "land_fraction_R_below_minus1", "land_fraction_change_only_with_amoc"),
    ("R and E over land", "land_median_R", "land_mean_E_in_sd"),
    ("land-wide, A vs C", "rms_ratio", "pattern_correlation"),
]
SUMMARY_VARIABLE_WIDTH = 2.3
SUMMARY_HEADER_ROWS = 3.2
SUMMARY_ROW_INCHES = 0.3
SUMMARY_FIGURE_WIDTH = 18.0
SUMMARY_FONTSIZE = 9
SUMMARY_STRIPE_COLOR = "0.94"


def _one_line_header(column):
    """A ``SummaryColumn`` header on one line ("over-\ncompensates" -> "overcompensates")."""
    return column.header.replace("-\n", "").replace("\n", " ")


def _summary_title(time_mean, threshold_sd):
    """Title of the summary table page and sheet."""
    c1, c4, c4slow = (cam.CASE_LABELS[case] for case in dl.CO2_AMOC_CASES)
    return (f"AMOC slowdown vs. CO₂ over land: summary by variable (CESM1.2 {time_mean} means; "
            f"C = {c4} − {c1}, A = {c4slow} − {c4}, T = A + C; substantive: |effect| ≥ "
            f"{threshold_sd:g}σ and p ≤ {cam.SIGNIFICANCE_P})")


def plot_co2_amoc_summary_table(table, time_mean, threshold_sd, out_path):
    """One-page PDF of the summary ``table`` (a DataFrame indexed by variable with
    the ``cam.SUMMARY_COLUMNS``), formatted per ``SUMMARY_TABLE_COLUMNS``, with
    grouped headers, striped rows and the column definitions beneath."""
    widths = [SUMMARY_VARIABLE_WIDTH] + [column.width for column in SUMMARY_TABLE_COLUMNS.values()]
    left = dict(zip(["variable", *SUMMARY_TABLE_COLUMNS], np.cumsum([0, *widths[:-1]])))
    total_width = sum(widths)
    rows = len(table)
    definitions = [f"{_one_line_header(SUMMARY_TABLE_COLUMNS[name])}: {text}"
                   for name, text in cam.SUMMARY_COLUMNS.items()]
    definition_lines = len(definitions) + 2
    table_inches = (SUMMARY_HEADER_ROWS + rows) * SUMMARY_ROW_INCHES
    fig = plt.figure(figsize=(SUMMARY_FIGURE_WIDTH,
                              table_inches + definition_lines * 0.17 + 1.2))
    ax = fig.add_axes((0.02, 1 - (table_inches + 0.8) / fig.get_figheight(), 0.96,
                       table_inches / fig.get_figheight()))
    ax.set_xlim(0, total_width)
    ax.set_ylim(SUMMARY_HEADER_ROWS + rows, 0)
    ax.axis("off")

    def center(name):
        return left[name] + (widths[0] if name == "variable" else SUMMARY_TABLE_COLUMNS[name].width) / 2

    for label, first, last in SUMMARY_TABLE_GROUPS:
        x0, x1 = left[first] + 0.05, left[last] + SUMMARY_TABLE_COLUMNS[last].width - 0.05
        ax.text((x0 + x1) / 2, 0.35, label, ha="center", va="center",
                fontsize=SUMMARY_FONTSIZE, fontweight="bold")
        ax.plot([x0, x1], [0.7, 0.7], color="black", lw=0.6)
    ax.text(center("variable"), SUMMARY_HEADER_ROWS - 0.15, "variable", ha="center",
            va="bottom", fontsize=SUMMARY_FONTSIZE, fontweight="bold")
    for name, column in SUMMARY_TABLE_COLUMNS.items():
        ax.text(center(name), SUMMARY_HEADER_ROWS - 0.15, column.header, ha="center", va="bottom",
                fontsize=SUMMARY_FONTSIZE, fontweight="bold", linespacing=1.1)
    ax.plot([0, total_width], [SUMMARY_HEADER_ROWS] * 2, color="black", lw=0.8)
    for row, (variable, values) in enumerate(table.iterrows()):
        y = SUMMARY_HEADER_ROWS + row + 0.5
        if row % 2:
            ax.axhspan(y - 0.5, y + 0.5, color=SUMMARY_STRIPE_COLOR, lw=0, zorder=0)
        ax.text(left["variable"] + 0.08, y, variable, va="center", fontsize=SUMMARY_FONTSIZE,
                family="monospace")
        for name, column in SUMMARY_TABLE_COLUMNS.items():
            text = column.python_format.format(values[name])
            if name == "description":
                ax.text(left[name] + 0.08, y, textwrap.shorten(text, 56, placeholder="…"),
                        va="center", fontsize=SUMMARY_FONTSIZE - 0.5)
            else:
                ax.text(center(name), y, text, ha="center", va="center", fontsize=SUMMARY_FONTSIZE)
    ax.plot([0, total_width], [SUMMARY_HEADER_ROWS + rows] * 2, color="black", lw=0.8)
    fig.suptitle(_summary_title(time_mean, threshold_sd), fontsize=11, y=1 - 0.3 / fig.get_figheight())
    definitions_top = (definition_lines * 0.17 + 0.3) / fig.get_figheight()
    fig.text(0.02, definitions_top, "Columns", va="top", fontsize=8.5, fontweight="bold")
    fig.text(0.02, definitions_top - 0.2 / fig.get_figheight(), "\n".join(definitions), va="top",
             fontsize=8, linespacing=1.45)
    _save_figure(fig, out_path=out_path)


def write_co2_amoc_summary_xlsx(table, time_mean, threshold_sd, out_path):
    """The summary ``table`` as an Excel workbook: sheet ``summary`` (title row, one
    row per variable, headers and number formats per ``SUMMARY_TABLE_COLUMNS``,
    panes frozen at the first value) and sheet ``definitions`` (``cam.SUMMARY_COLUMNS``)."""
    headers = {name: _one_line_header(column) for name, column in SUMMARY_TABLE_COLUMNS.items()}
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        table.rename(columns=headers).to_excel(writer, sheet_name="summary", startrow=1)
        sheet = writer.sheets["summary"]
        sheet["A1"] = _summary_title(time_mean, threshold_sd)
        sheet.freeze_panes = "C3"
        sheet.column_dimensions["A"].width = 16
        for index, column in enumerate(SUMMARY_TABLE_COLUMNS.values(), start=2):
            letter = sheet.cell(row=2, column=index).column_letter
            sheet.column_dimensions[letter].width = 7 * column.width + 6
            sheet.cell(row=2, column=index).alignment = openpyxl.styles.Alignment(wrap_text=True)
            for row in range(3, len(table) + 3):
                sheet.cell(row=row, column=index).number_format = column.excel_format
        pd.DataFrame({"column": list(headers.values()),
                      "definition": list(cam.SUMMARY_COLUMNS.values())}).to_excel(
            writer, sheet_name="definitions", index=False)
        writer.sheets["definitions"].column_dimensions["A"].width = 32
        writer.sheets["definitions"].column_dimensions["B"].width = 110


# Guide page opening the CO2 / AMOC metric book (``plot_co2_amoc_metrics_guide``):
# two columns of headed sections, each paragraph wrapped to GUIDE_WRAP_CHARS with a
# hanging indent for its "term:" lead-in.
GUIDE_FIGSIZE = (17, 11)
GUIDE_FONTSIZE = 10.0
GUIDE_LINE_HEIGHT = 1.32 * GUIDE_FONTSIZE / 72 / GUIDE_FIGSIZE[1]  # figure fraction per line
GUIDE_WRAP_CHARS = 104
GUIDE_INDENT = "      "
GUIDE_COLUMNS_X = (0.035, 0.52)
GUIDE_TOP = 0.90


def _guide_column(fig, x, sections):
    """Draw ``sections`` (``(heading, [paragraph, ...])`` pairs) down the page from
    ``GUIDE_TOP`` at figure x ``x``: bold headings, wrapped paragraphs with hanging
    indents, a blank line between paragraphs and two before each heading."""
    y = GUIDE_TOP
    for heading, paragraphs in sections:
        fig.text(x, y, heading, fontsize=GUIDE_FONTSIZE + 1.5, fontweight="bold", va="top")
        y -= 1.6 * GUIDE_LINE_HEIGHT
        for paragraph in paragraphs:
            text = textwrap.fill(paragraph, GUIDE_WRAP_CHARS, subsequent_indent=GUIDE_INDENT)
            fig.text(x, y, text, fontsize=GUIDE_FONTSIZE, va="top", linespacing=1.32)
            y -= (text.count("\n") + 1.5) * GUIDE_LINE_HEIGHT
        y -= GUIDE_LINE_HEIGHT


def plot_co2_amoc_metrics_guide(time_mean, sd_years, threshold_sd, variables, out_path):
    """One-page PDF explaining the metric pages (``plot_co2_amoc_metrics``) of a
    book over ``variables``, for means over ``time_mean``,
    sigma estimated over ``sd_years`` and the category threshold ``threshold_sd``."""
    c1, c4, c4slow = (cam.CASE_LABELS[case] for case in dl.CO2_AMOC_CASES)
    left = [
        ("What this book shows", [
            f"For each climate variable, land maps comparing the effect of AMOC slowdown with "
            f"the effect of quadrupled CO₂ in CESM1.2 (NAHosMIP runs), from {time_mean} means "
            f"(the last 50 years common to the three runs). Oceans are treated as missing data "
            f"and drawn pale blue-grey; land is the Natural Earth 110m land mask. Land cells "
            f"where the variable has no measurable variability in {c1} (σ ≈ 0, e.g. low "
            f"cloud over the high Antarctic and Tibetan plateaus) are missing too, and drawn the same way.",
            f"Pages: one per variable, with natural variability σ as the scale. Variables, in "
            f"order: {', '.join(variables)}. The last page is a summary table, one row per "
            f"variable.",
        ]),
        ("Cases", [
            f"{c1} = 1xCO2: 1×CO₂ control, AMOC ≈ 20 Sv.",
            f"{c4} = 4xCO2_m03Sv: 4×CO₂ with −0.3 Sv hosing, which holds AMOC near 20 Sv.",
            f"{c4slow} = 4xCO2: 4×CO₂ without hosing; AMOC slows.",
        ]),
        ("Effects, per grid cell", [
            f"C = {c4} − {c1}: the CO₂ effect without AMOC slowdown.",
            f"A = {c4slow} − {c4}: the effect of AMOC slowdown. It is measured at 4×CO₂, so it "
            f"includes any interaction between AMOC slowdown and CO₂.",
            f"T = A + C = {c4slow} − {c1}: the total response.",
            "Significance: Welch t-test of the difference between two runs, on their five "
            "10-year means. Maps are contoured at their colorbar ticks, black where p ≤ 0.05 "
            "and light grey where not (on E and R, the contours show A's significance).",
        ]),
        ("Scale: what counts as large", [
            f"σ: year-to-year standard deviation of {c1} annual means over {sd_years}, after "
            f"removing each cell's linear trend. It measures the variability the climate "
            f"already has.",
            "The plain ratio A / C blows up where C ≈ 0, making the AMOC effect look large "
            "wherever the CO₂ effect is small, even if both are negligible. So the ratio R "
            "is shown only where C is substantive, and the other metrics divide by σ.",
        ]),
        ("Land-wide numbers (three-part map title; also in the summary table)", [
            "RMS(A) / RMS(C): area-weighted root-mean-square size of A relative to C over "
            "land. Unlike an average of local ratios, it does not blow up where C ≈ 0.",
            "Pattern congruence = Σ w·A·C / √(Σ w·A² · Σ w·C²) (w = cell area): the uncentered "
            "pattern correlation. −1 if A opposes C everywhere in proportion (AMOC slowdown "
            "counters the CO₂ pattern), +1 if it reinforces it.",
            "Pattern correlation: the usual centered (Pearson) correlation, comparing only "
            "the spatial variations about each field's land mean.",
        ]),
    ]
    right = [
        ("Page rows, from top", [
            "1. C, A, T in the variable's units, on one shared color scale (red = increase; "
            "for water fluxes the colors flip, so blue = wetter), each contoured by its own "
            "significance.",
            "2. C / σ, A / σ, T / σ: the effects in units of the control's variability σ, on "
            "one shared color scale.",
            "3. E, R and the three-part map, described below.",
        ]),
        ("Row 3", [
            "E = (|T| − |C|) / σ: how far AMOC slowdown moves the 4×CO₂ climate farther "
            f"from (E > 0, purple: exacerbates) or closer to (E < 0, green: ameliorates) the "
            f"{c1} climate. |E| ≤ |A| / σ, so E is small wherever the AMOC effect is. If A "
            "opposes C and is more than twice as large, the climate ends up farther away on "
            "the other side, which E counts as exacerbating: R picks out every "
            "overcompensation.",
            "R = A / C: the AMOC effect as a fraction of the CO₂ effect. −1: "
            "AMOC slowdown cancels the CO₂ effect; −0.5: halves it; 0: no effect; +1: doubles "
            "it. Gold, R < −1: AMOC slowdown overcompensates, leaving a change of the opposite "
            f"sign to the CO₂ change. An effect is substantive where |effect| ≥ "
            f"{threshold_sd:g}σ and p ≤ {cam.SIGNIFICANCE_P}. R is shown where C is "
            "substantive. Where it is not, R is meaningless (C ≈ 0), and the cell is orange "
            "if T is substantive (a change that appears only with AMOC slowdown) and dark "
            "grey if neither is. The title gives the gold and orange shares of land and the "
            "area-weighted median of R.",
            "Three-part map: |A|, |C| and σ as shares of σ + |A| + |C| (they sum "
            "to 1). Orange: AMOC slowdown dominates; blue: CO₂ dominates; white: both effects "
            "small compared with σ; mixed colors: mixtures. The triangle beside it is "
            "the color key, with every land cell as a dot.",
        ]),
        ("Strips beside the maps", [
            "Statistics over land longitudes at each latitude, aligned with the map: land "
            "zonal mean (black) and the 5–95% range of land cells (grey). Gaps are latitudes "
            "with no land.",
        ]),
        ("Summary table (last page; also .csv and .xlsx)", [
            "Besides the land-wide numbers, the table gives the share of land in each "
            "category (AMOC effect not substantive; exacerbates: same sign as a substantive "
            "C; partly offsets: opposite sign, |A| < |C|; overcompensates: |A| ≥ |C|; "
            "AMOC-only: substantive A where C is not), the land-mean E, the median R and "
            "the gold and orange shares of the R map. Its columns are defined beneath it.",
        ]),
    ]
    fig = plt.figure(figsize=GUIDE_FIGSIZE)
    fig.text(0.5, 0.965, "AMOC slowdown vs. CO₂ over land: guide to the metrics in this book",
             ha="center", va="top", fontsize=16, fontweight="bold")
    for x, sections in zip(GUIDE_COLUMNS_X, (left, right)):
        _guide_column(fig, x, sections)
    fig.savefig(out_path)
    plt.close(fig)
