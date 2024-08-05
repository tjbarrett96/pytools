import scipy.stats
import numpy as np
import scipy.linalg as lg
import dill
import os
import re

# ==================================================================================================

def ensure_type(obj, type):
  """Wraps the given object as the given type if not already an instance of that type."""
  if isinstance(obj, type):
    return obj
  else:
    return type(obj)

# ==================================================================================================
  
def collect_dict_values_as_lists(*dicts):
  """
  Merges a list of dictionaries with the same key-value structure into one dictionary
  with a list of values for each key.
  """
  
  keys = dicts[0].keys() if len(dicts) > 0 else [] 
  
  result = {
    key: [dictionary[key] for dictionary in dicts]
    for key in keys
  }
  
  # wrap 1D lists of numbers as NumPy arrays
  for key in result:
    if is_numeric_list(result[key]):
      result[key] = np.array(result[key])
  
  return result

# ==================================================================================================
  
def merge_dicts(*dicts):
  """
  Merges a list of dictionaries into one dictionary with all of the unique keys and values together.
  """
  return {
    key: value
    for dictionary in dicts
    for key, value in dictionary.items()
  }

# ==================================================================================================

def order_of_magnitude(number):
  return int(np.floor(np.log10(abs(number)))) if number != 0 else 0

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
  
# use regex to check if argument is a list of the form range(start,end,step) or [item1;item2;...]
regex_number = r"\s*([+-]?(?:[0-9]*[.])?[0-9]+)\s*"
regex_range = fr"range\({regex_number},{regex_number},{regex_number}\)"
regex_list = r"\[(.*)\]"

def parse_as_list(string, type = None):
  
  if type is None:
    type = (lambda x: x)

  if (match_range := re.match(regex_range, str(string))):
    return list(np.arange(*[float(n) for n in match_range.groups()]))
  elif (match_list := re.match(regex_list, str(string))):
    return [parse_as_list(item.strip(), type) for item in match_list.group(1).split(";")]
  else:
    return [type(string)]

# ==================================================================================================
  
def is_iterable(obj):
  """Checks whether or not the given object is an iterable container, excluding strings."""
  try:
    iter(obj)
    return (not isinstance(obj, str))
  except TypeError:
    return False

# ==================================================================================================

def is_numeric_list(obj):
  return is_iterable(obj) and all((np.issubdtype(type(x), np.number) or np.issubdtype(type(x), bool)) for x in obj)

# ==================================================================================================
  
def save(obj: object, filename: str) -> None:
  """Saves given object to given filename using dill (pickle) module."""
  os.makedirs(os.path.dirname(filename), exist_ok = True)
  with open(filename, "wb") as output_file:
    dill.dump(obj, output_file)

# ==================================================================================================
    
def load(filename: str, key: str = None) -> object:
  """Loads object from given filename using dill (pickle) module."""
  with open(filename, "rb") as input_file:
    data = dill.load(input_file)
    return data[key] if key is not None else data

# ==================================================================================================
  
def matrix_to_dict(matrix: np.ndarray, labels: list[str]) -> dict[str, dict[str, float]]:
  """
  For a square matrix whose dimensions can both be labeled by the same sequence of strings (e.g.
  a covariance matrix), returns a nested dictionary allowing element access by string labels.
  For example: cov_matrix[0][1] ==> cov_dict["a"]["b"].
  """
  return {a: {b: matrix[i, j] for j, b in enumerate(labels)} for i, a in enumerate(labels)}

# ==================================================================================================

def p_value(chi2: float, ndf: int) -> float:
  """
  Returns the probability of observing a chi-squared sample larger than the given sample,
  based on the given number of degrees of freedom ('ndf').
  """
  return scipy.stats.chi2.sf(chi2, ndf)

# ==================================================================================================

# TODO: generalize to use data covariance matrix
def fit_linear_combination(
    terms: list[np.ndarray],
    y: np.ndarray,
    y_err: np.ndarray = None,
    error_mode: str = None,
    return_cond: bool = False
  ) -> np.ndarray:
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

    cond = None
    if len(A) > 0:
      try:
        cond = np.linalg.cond(A)
      except np.linalg.LinAlgError:
        cond = -1
    result = lg.solve(A, b, assume_a = "sym")

    if error_mode is None:
      return (result if not return_cond else (result, cond))
    else:
      # compute dp_dy matrix for propagation of data uncertainties to parameters
      # lg.solve treats last dimension of 'terms / y_err' as vector, repeats over first dimension (i.e. p)
      dp_dy = lg.solve(A, terms / y_err, assume_a = "sym")
      if error_mode == "err":
        # shortcut to sqrt(diagonal) of covariance matrix
        result_err = np.sqrt(np.einsum("ki, ki, i -> k", dp_dy, dp_dy, y_err**2))
        return ((result, result_err) if not return_cond else (result, result_err, cond))
      else:
        # full covariance matrix
        result_cov = np.einsum("ki, li, i -> kl", dp_dy, dp_dy, y_err**2)
        return ((result, result_cov) if not return_cond else (result, result_cov, cond))
