import inspect
import numpy as np
from typing import Callable, Self

# =================================================================================================

# TODO: make 2 options for instantiation?
#   (a) from function using named parameters, in which case parameters are detected
#   (b) from function using dictionary, in which case dependent parameter list must be supplied
#   eval set dynamically based on which version, either unpacking ordered parameters from dictionary
#     or passing along restricted view of dictionary
class Expression:
  
  # ===============================================================================================

  def __init__(
    self,
    function: Callable[..., np.ndarray | np.number] | np.ndarray | np.number,
    guesses: dict[str, float] = None,
    bounds: dict[str, float] = None,
    label: str = None
  ):

    # store function for internal use
    if isinstance(function, (np.ndarray, np.number, int, float)):
      self._function = (lambda t: function * np.ones(len(t)))
      signature = {}
    else:
      self._function = function
      # get list of parameter names from function signature
      signature = inspect.signature(function).parameters if function is not None else {}

    # get parameter names and initial guesses from default values in function signature
    self.parameters = {
      name: info.default if info.default != info.empty else None
      for name, info in list(signature.items())[1:]
    }

    # if dictionary of guesses passed, override default initial guesses
    if guesses is not None:
      for name, guess in guesses.items():
        if name in self.parameters:
          self.parameters[name] = guess

    # store parameter bounds if passed
    self.bounds = bounds
    if self.bounds is None:
      self.bounds = {}

    # add label to parameters if passed
    self.add_label(label)

  # ===============================================================================================

  def add_label(self, label: str) -> None:
    """Append underscore and label string to each parameter name."""
    if label is not None:
      self.parameters = {f"{name}_{label}": value for name, value in self.parameters.items()}
      self.bounds = {f"{name}_{label}": value for name, value in self.bounds.items()}
    return self

  # ===============================================================================================
  
  def eval(self, t: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Evaluate expression from global parameter dictionary using this expression's known parameter names."""
    return self._function(t, *(p[name] for name in self.parameters))

  # ===============================================================================================

  # delegate calling to self.eval, which can be changed dynamically (unlike special __call__ method)
  def __call__(self, t: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Evaluate expression from global parameter dictionary using this expression's known parameter names."""
    return self.eval(t, p)

  # ===============================================================================================

  def _merge_skeleton(self, other: Self) -> Self:
    """Construct new Expression which merges the parameters of this Expression and another, but leaves evaluation logic undefined."""
    result = Expression(None)
    result.parameters = {**self.parameters, **other.parameters}
    result.bounds = {**self.bounds, **other.bounds}
    result.eval = None
    return result

  # ===============================================================================================

  def __add__(self, other: Self) -> Self:
    """Construct new Expression which returns the sum of this Expression and another."""
    result = self._merge_skeleton(other)
    result.eval = lambda t, p: self(t, p) + other(t, p)
    return result

  # ===============================================================================================
    
  def __mul__(self, other: Self) -> Self:
    """Construct new Expression which returns the product of this Expression and another."""
    result = self._merge_skeleton(other)
    result.eval = lambda t, p: self(t, p) * other(t, p)
    return result

# =================================================================================================

# Type alias for Expressions or arrays/numbers that can be coerced into Expressions.
ExpressionLike = (Expression | Callable[..., np.ndarray | np.number] | np.ndarray | np.number)