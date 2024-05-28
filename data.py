from dataclasses import dataclass
import numpy as np

@dataclass
# TODO: consider making NamedTuple instead? then supports both member access and ordered unpacking
#       e.g. for plotting API that takes plot(x, y, y_err), can do plot(*data)
class Data:
  """Simple container for x-values, y-values, and optional errorbars."""
  x: np.ndarray
  y: np.ndarray
  y_err: np.ndarray = None
  x_err: np.ndarray = None