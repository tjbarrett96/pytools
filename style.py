import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import pytools.util as util

# ==================================================================================================

# Font options.
size = 14
plt.rcParams["font.size"] = size * 0.75
plt.rcParams["axes.labelsize"] = size
plt.rcParams["axes.titlesize"] = size
plt.rcParams["xtick.labelsize"] = size
plt.rcParams["ytick.labelsize"] = size
plt.rcParams["legend.fontsize"] = size * 0.75

# Rules for switching to scientific notation in axis tick labels.
plt.rcParams["axes.formatter.limits"] = (-2, 3)
plt.rcParams["axes.formatter.offset_threshold"] = 3
plt.rcParams["axes.formatter.use_mathtext"] = True

# Marker and line options.
plt.rcParams["lines.markersize"] = 5
plt.rcParams["lines.linewidth"] = 1

# Draw grid.
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

# Default figure size.
plt.rcParams["figure.figsize"] = 8, 5
plt.rcParams["figure.dpi"] = 100

# Make axis tick marks face inward.
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Draw axis tick marks all around the edges.
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

# Draw minor axis tick marks in between major labels.
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True

# Make all tick marks longer.
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["xtick.minor.size"] = 4
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.minor.size"] = 4

# Dynamically choose the number of histogram bins.
plt.rcParams["hist.bins"] = "auto"

# Space between axes and legend, in font units.
plt.rcParams["legend.borderaxespad"] = 1
plt.rcParams["legend.handlelength"] = 1
plt.rcParams["legend.columnspacing"] = 1.4
plt.rcParams["legend.handletextpad"] = 0.6
plt.rcParams["legend.framealpha"] = 0.6

# Default subplot spacing.
plt.rcParams["figure.subplot.wspace"] = 0.3
plt.rcParams["figure.subplot.hspace"] = 0.3
plt.rcParams["figure.subplot.top"] = 0.93
plt.rcParams["figure.subplot.right"] = 0.93

# ==================================================================================================

def xlabel(label: str, offset: float = 0) -> None:
  """
  Override plt.xlabel with right-alignment.
  :param label: Label for x-axis.
  :param offset: Offset from right edge, as a fraction of the axes width.
  """
  return plt.xlabel(label, ha = "right", x = 1 - offset)

# ==================================================================================================

def ylabel(label: str, offset: float = 0) -> None:
  """
  Override plt.ylabel with top-alignment.
  :param label: Label for y-axis.
  :param offset: Offset from top edge, as a fraction of the axes width.
  """
  return plt.ylabel(label, ha = "right", y = 1 - offset)

# ==================================================================================================

def twinx(shrink = None):
  """
  Switch plotting to y-axis on right side of plot, without duplicating grid lines.
  :param shrink: Amount of space to add for right-side y-axis, as a fraction of the axes width.
  """
  plt.twinx()
  plt.grid(False)
  if shrink is not None:
    plt.subplots_adjust(right = plt.gca().get_position().xmax - shrink)

# ==================================================================================================

def colorbar(mappable = None, label = None, pad = 0.01, fraction = 0.10, aspect = 18, **kwargs):
  """Override plt.colorbar with automatic formatting."""
  cbar = plt.colorbar(mappable = mappable, ax = plt.gca(), pad = pad, fraction = fraction, aspect = aspect, **kwargs)
  if label is not None:
    cbar.set_label(label, ha = "right", y = 1)
  return cbar

# ==================================================================================================

def errorbar(x, y, y_err, x_err = None, ls = "-", marker = "o", ms = 4, capsize = 0, **kwargs):
  """Override plt.errorbar with automatic formatting."""
  return plt.errorbar(x, y, y_err, x_err, fmt = f"{marker}{ls}", ms = ms, capsize = capsize, lw = 0.75, elinewidth = 0.5, mew = 0.5, **kwargs)

# ==================================================================================================

def errorband(x, y, y_err, **kwargs):
  plot_args = {"alpha": 0.3, "color": "k", "ec": None, **kwargs}
  return plt.fill_between(x, y - y_err, y + y_err, **plot_args)

# ==================================================================================================

def make_unique_legend(extend_x = 0, **kwargs):
  """Show a legend on the current plot, containing only unique labels without duplicates."""
  # Get the artist handles and text labels for everything in the current plot.
  handles, labels = plt.gca().get_legend_handles_labels()
  # Make a dictionary mapping labels to handles; this ensures each label only appears with one handle.
  labels_to_handles = {}
  for label, handle in zip(labels, handles):
    if label not in labels_to_handles:
      labels_to_handles[label] = handle
  # Make a legend, as long as there are some labels to show.
  if len(labels_to_handles) > 0:
    if extend_x > 0:
      x_low, x_high = plt.xlim()
      plt.xlim(x_low, x_high + extend_x * (x_high - x_low))
    plt.legend(handles = labels_to_handles.values(), labels = labels_to_handles.keys(), **kwargs)

# ==================================================================================================

def label_and_save(x_label, y_label, output, **legend_kwargs):
  """Shortcut for labeling axes, adding legend, saving the figure, and clearing the figure."""
  xlabel(x_label)
  ylabel(y_label)
  make_unique_legend(**legend_kwargs)
  if isinstance(output, PdfPages):
    output.savefig()
  else:
    plt.savefig(output)
  plt.clf()

# ==================================================================================================

# class Entry:

#   def __init__(self, val, sym, err = None, units = None):
#     (self.val, self.err) = (val, err)
#     (self.sym, self.units) = (sym.symbol, sym.units) if isinstance(sym, const.Quantity) else (sym, units)

#   def format(self, align = False, places = 4):
#     m = "" if align else "$" # math mode boundary character: "" if already inside math env, else "$"
#     amp = "&" if align else "" # alignment character: "&" if inside math align env, else ""
#     err = "" if self.err is None else rf" {m}\pm{m} {self.err:.{places}f}"
#     units = "" if self.units is None else (rf"\;\text{{{self.units}}}" if align else f" {self.units}")
#     return rf"{m}{self.sym}{m} {amp}= {self.val:.{places}f}{err}{units}"

def databox(*entries, left = True, **kwargs):
  plt.text(
    0.03 if left else 0.97,
    0.96,
    "\n".join(entries),
    ha = "left" if left else "right",
    va = "top",
    transform = plt.gca().transAxes,
    bbox = {
      "alpha": plt.rcParams["legend.framealpha"],
      "facecolor": "white",
      "edgecolor": plt.rcParams["legend.edgecolor"],
      "boxstyle": "round"
    },
    **kwargs
  )

# ==================================================================================================

def colormesh(x, y, heights, label = None, cmap = "coolwarm", **kwargs):
  """Override plt.pcolormesh with automatic formatting and colorbar."""
  result = plt.pcolormesh(x, y, heights.T, cmap = cmap, **kwargs)
  cbar = colorbar(label)
  return result, cbar

# ==================================================================================================

def draw_horizontal(y = 0, ls = ":", c = "k", **kwargs):
  """Draw horizontal line at given y-value."""
  return plt.axhline(y, linestyle = ls, color = c, **kwargs)

def draw_vertical(x = 0, ls = ":", c = "k", **kwargs):
  """Draw vertical line at given x-value."""
  return plt.axvline(x, linestyle = ls, color = c, **kwargs)

def horizontal_spread(width, y = 0, color = "k", **kwargs):
  """Draw horizontal band across the range (y - width, y + width)."""
  return plt.axhspan(y - width, y + width, color = color, alpha = 0.1, **kwargs)

def vertical_spread(width, x = 0, color = "k", **kwargs):
  """Draw vertical band across the range (x - width, x + width)."""
  return plt.axvspan(x - width/2, x + width/2, color = color, alpha = 0.1, **kwargs)

# ==================================================================================================

def make_pdf(path):
  """Create and return PDF at given path."""
  return PdfPages(path)

# ==================================================================================================

def make_indexed_color_scale(n: int, cmap = None, special_colors: dict = None) -> tuple[mpl.colors.Colormap, plt.cm.ScalarMappable]:

  indices = np.arange(n)

  if special_colors is None:
    special_colors = {}

  if cmap is None:
    cmap = mpl.colors.ListedColormap([f"C{i % 10}" for i in range(n)])
  else:
    cmap = mpl.colormaps[cmap].resampled(n)
  
  if len(special_colors) > 0:
    discrete_colors = cmap(indices)
    for index, color in special_colors.items():
      discrete_colors[index] = mpl.colors.to_rgba(color)
    cmap = mpl.colors.ListedColormap(discrete_colors)

  norm = mpl.colors.BoundaryNorm(np.arange(-0.5, n), n)

  return cmap, plt.cm.ScalarMappable(norm, cmap)

# ==================================================================================================

# TODO: allow formatting options to apply universally to all lines, or dictionaries specifying certain indices with unique formats
#       e.g. maybe special-color line should also be different thickness or different bar/line/band configuration than others
# TODO: add marker differentiation options

def plot_color_series(
  x,
  y,
  y_err = None,
  x_err = None,
  color = None,
  line = False,
  errorbars = True,
  errorband = False,
  color_label = None,
  color_ticks = None,
  special_colors = None,
  cmap = None,
  label = None,
  alpha = 1
):
  
  if special_colors is None:
    special_colors = {}

  def ensure_list_of_arrays(obj):
    if isinstance(obj, np.ndarray) and obj.ndim == 1 and np.issubdtype(obj.dtype, np.number):
      return [obj]
    elif util.is_iterable(obj) and all(isinstance(item, np.ndarray) and item.ndim == 1 for item in obj):
      return list(obj)
    else:
      raise ValueError("Data provided to plot_color_series must be 1-d arrays or lists of arrays.")

  y = ensure_list_of_arrays(y)
  if y_err is not None:
    y_err = ensure_list_of_arrays(y_err)
    if len(y) != len(y_err) or any(series.shape != series_err.shape for series, series_err in zip(y, y_err)):
      raise ValueError("Mismatch between y-values and y-errors.")
  else:
    y_err = [None for _ in range(len(y))]
    
  x = ensure_list_of_arrays(x)
  if x_err is not None:
    x_err = ensure_list_of_arrays(x_err)
    if len(x) != len(x_err) or any(series.shape != series_err.shape for series, series_err in zip(x, x_err)):
      raise ValueError("Mismatch between x-values and x-errors.")
  else:
    x_err = [None for _ in range(len(x))]

  if len(x) == 1 and len(y) > 1:
    x = [x[0] for _ in range(len(y))]
    x_err = [x_err[0] for _ in range(len(y))]

  cmap, sm = make_indexed_color_scale(len(y), cmap = cmap, special_colors = special_colors)

  for i in range(len(y)):
    c = cmap(i) if color is None else color
    zorder = 0 if i in special_colors else -10
    if errorbars:
      plt.scatter(x[i], y[i], color = c, label = label, zorder = zorder, alpha = alpha)
      errorbar(x[i], y[i], y_err[i] if y_err is not None else None, x_err[i] if x_err is not None else None, color = c, ls = "", marker = "none", alpha = alpha, zorder = zorder)
    if line:
      plt.plot(x[i], y[i], color = c, ls = "-", lw = 0.75, label = label, zorder = zorder, alpha = alpha)
    if errorband and y_err is not None:
      plt.fill_between(x[i], y[i] - y_err[i], y[i] + y_err[i], fc = c, alpha = alpha * 0.3, zorder = (zorder - 1), label = label)

  if len(y) > 1:
    cbar = colorbar(sm, label = color_label)
    cbar.ax.tick_params(labelsize = 10)
    all_ticks = np.arange(len(y))
    all_tick_labels = color_ticks if color_ticks is not None else np.arange(len(y))
    tick_step = 1
    max_ticks = 25
    if len(all_ticks) > max_ticks:
      tick_step = int(np.ceil(len(all_ticks) / max_ticks))
    cbar.set_ticks(all_ticks[::tick_step])
    cbar.minorticks_off()
    cbar.set_ticklabels(all_tick_labels[::tick_step])
