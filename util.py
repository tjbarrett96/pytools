import scipy.stats
import numpy as np
import scipy.linalg as lg

# ==================================================================================================

def ensure_type(obj, type):
  """Wraps the given object as the given type if not already an instance of that type."""
  if isinstance(obj, type):
    return obj
  else:
    return type(obj)

# ==================================================================================================

def str_to_bool(string):
  """
  Converts the strings 'true' and 'false' (ignoring capitalization) to booleans True and False,
  or else raises a ValueError.
  """
  if string.lower() == "true":
    return True
  elif string.lower() == "false":
    return False
  else:
    raise ValueError("Boolean string must be either 'true' or 'false'.")

# ==================================================================================================

def p_value(chi2: float, ndf: int) -> float:
  """
  Returns the probability of observing a chi-squared sample larger than the given sample,
  based on the given number of degrees of freedom ('ndf').
  """
  return scipy.stats.chi2.sf(chi2, ndf)

# ==================================================================================================

def fit_linear_combination(
    terms: list[np.ndarray],
    y: np.ndarray,
    y_err: np.ndarray = None,
    error_mode: str = None
  ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Returns the array of best-fit coefficients that model the data (y +/- y_err) as a linear
    combination of the given terms, provided as a list of arrays [y0, y1, ...]. Optionally,
    coefficient errors are returned if error_mode == 'err', or covariance matrix if 'cov'.
    """
    
    if error_mode not in (None, "err", "cov"):
      raise ValueError("fit_linear_combination 'error_mode' must be one of (None, 'err', 'cov').")
    
    # if list of terms is empty, still match last dimension(s) to shape of 'y'
    if len(terms) == 0:
      terms = np.empty(shape = (0, *y.shape))
    
    # treat errors as 1 if not provided
    if y_err is None:
      y_err = np.ones(shape = y.shape)

    # normalize model terms and data by errors
    terms = np.array(terms) / y_err # division applies to last dimension(s) of 'terms', repeating over first
    y = y / y_err

    # if data is multi-dimensional, flatten it
    if len(y.shape) > 1:
      y = y.ravel() # ravel returns flattened view rather than copying
      y_err = y_err.ravel()
      terms = terms.reshape(terms.shape[0], -1) # keep first dimension (i.e. the terms) and flatten rest

    # set up and solve matrix equation Ax = b
    A = terms @ terms.T
    b = y @ terms.T
    result = lg.solve(A, b, assume_a = "sym")

    if error_mode is None:
      return result
    else:
      # compute dp_dy matrix for propagation of data uncertainties to parameters
      # lg.solve treats last dimension of 'terms / y_err' as vector, repeats over first dimension (i.e. p)
      dp_dy = lg.solve(A, terms / y_err, assume_a = "sym")
      if error_mode == "err":
        # shortcut to sqrt(diagonal) of covariance matrix
        result_err = np.sqrt(np.einsum("ki, ki, i -> k", dp_dy, dp_dy, y_err**2))
        return result, result_err
      else:
        # full covariance matrix
        result_cov = np.einsum("ki, li, i -> kl", dp_dy, dp_dy, y_err**2)
        return result, result_cov