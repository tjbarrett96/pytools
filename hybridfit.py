import numpy as np
import scipy.stats as stats
from typing import Mapping, Callable
from dataclasses import dataclass
import functools
import itertools
import matplotlib.pyplot as plt

import logging
print = logging.info

import iminuit
from iminuit.cost import LeastSquares

import pytools.util as util
from pytools.data import Data

import tabulate as tab

import inspect
import fnmatch
import numpy as np
import math
from typing import Callable, Self

# =================================================================================================

# @dataclass
# # TODO: consider making NamedTuple instead? then supports both member access and ordered unpacking
# #       e.g. for plotting API that takes plot(x, y, y_err), can do plot(*data)
# class Data:
#   """Simple container for x-values, y-values, and optional errorbars."""
#   x: np.ndarray
#   y: np.ndarray
#   y_err: np.ndarray = None
#   x_err: np.ndarray = None

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
      self._function = (lambda t: function * (np.ones(len(t)) if t is not None else 1))
      # self._function = lambda t: function
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
    self.label = label
    self.add_label(label)

    self.t_cache = None
    self.p_cache = None
    self.val_cache = None

  # ===============================================================================================
  
  def copy(self):
    #return Expression(self._function, self.parameters, self.bounds)
    result = Expression(None)
    result.parameters = {**self.parameters}
    result.bounds = {**self.bounds}
    # TODO: this assignment below might not work right, will it permanently bind current 'self' reference to result method call?
    result.eval = self.eval
    return result

  # ===============================================================================================

  def add_label(self, label: str, only: list[str] = None) -> None:
    """Append underscore and label string to each parameter name."""
    
    if self.label is None:

      # if list of restricted names supplied, check that a name matches one of the supplied prefixes
      should_modify = lambda name: True if (only is None) else any(name.startswith(prefix) for prefix in only)
    
      if label is not None:
        self.label = label
        self.parameters = {f"{name}{f'_{label}' if should_modify(name) else ''}": value for name, value in self.parameters.items()}
        self.bounds = {f"{name}{f'_{label}' if should_modify(name) else ''}": value for name, value in self.bounds.items()}

    elif label is not None and label != self.label:
      raise ValueError(f"Attempting to set Expression label to {label}, but already set to {self.label}.")

    return self

  # ===============================================================================================

  # TODO: add strict_cache = True option to check 't' element-wise, disable for looser size/range checking
  # TODO: need to keep this eval to use special arg, instead of replacing in __mul__ etc. below
  # TODO: so modification to __add__ and __mul__ needed
  def eval(self, p: dict[str, float], t: np.ndarray) -> np.ndarray | float:
    """Evaluate expression from global parameter dictionary using this expression's known parameter names."""
    args = [p[name] for name in self.parameters]
    #t_same = (t == self.t_cache).all()
    # t_same = (self.t_cache is not None and t is not None and len(t) == len(self.t_cache))# and t[0] == self.t_cache[0] and t[-1] == self.t_cache[-1])
    # if t_same and (args == self.p_cache):
    if args == self.p_cache:
      return self.val_cache
    else:
      result = self._function(t, *args)
      self.t_cache = t
      self.p_cache = args
      self.val_cache = result
      return result

  # ===============================================================================================

  # delegate calling to self.eval, which can be changed dynamically (unlike special __call__ method)
  def __call__(self, p: dict[str, float], t: np.ndarray = None) -> np.ndarray | float:
    """Evaluate expression from global parameter dictionary using this expression's known parameter names."""
    return self.eval(p, t)

  # ===============================================================================================

  def __add__(self, other: Self) -> Self:
    """Construct new Expression which returns the sum of this Expression and another."""
    return Sum(self, other)

  # ===============================================================================================
    
  def __mul__(self, other: Self) -> Self:
    """Construct new Expression which returns the product of this Expression and another."""
    return Product(self, other)

  # ================================================================================================

  def clear_cache(self):
    """Deletes cached numerical arrays to save space (e.g. for pickling)."""
    self.t_cache = None
    self.p_cache = None
    self.val_cache = None

# Type alias for Expressions or arrays/numbers that can be coerced into Expressions.
ExpressionLike = (Expression | Callable[..., np.ndarray | np.number] | np.ndarray | np.number)

# ==================================================================================================

class MultiExpression(Expression):

  def __init__(self, *expressions: list[Expression], function = sum):

    self.expressions = [util.ensure_type(expr, Expression) for expr in expressions]
    self.function = function

    super().__init__(None)

    self.parameters = {}
    self.bounds = {}
    for expression in self.expressions:
      self.parameters.update(expression.parameters)
      self.bounds.update(expression.bounds)

  def eval(self, p: dict[str, float], t: np.ndarray) -> np.ndarray | float:
    return self.function(expression.eval(p, t) for expression in self.expressions)
  
  def add_label(self, label: str, only: list[str] = None) -> None:
    super().add_label(label, only)
    for expression in self.expressions:
      expression.add_label(label, only)

  def clear_cache(self):
    super().clear_cache()
    for expression in self.expressions:
      expression.clear_cache()

Sum = MultiExpression

class Product(MultiExpression):
  def __init__(self, *expressions: list[Expression]):
    super().__init__(*expressions, function = math.prod)

# ==================================================================================================

# Type hint for a function which takes a dictionary of parameter names/values, and returns a value
# in terms of those parameters. Used to constrain one parameter in terms of others.
ConstraintFunction = Callable[..., float]

# TODO: Constraint integration with Expression feels hacky. Clean up and unify better somehow
class Constraint(Expression):
  """Simple container which wraps a constraint function alongside a list declaring the dependent parameters."""

  def __init__(self, parameters: list[str], constraint: ConstraintFunction):
    super().__init__(None)
    self.parameters = {parameter: None for parameter in parameters}
    self.constraint = constraint
    #self.signature = inspect.signature(constraint).parameters

  def eval(self, p: Mapping[str, float], t: np.ndarray = None) -> float:
    try:
      # restrict parameter dictionary to only the declared dependent parameters
      return self.constraint(*[p[key] for key in self.parameters])
    except KeyError as error:
      error.add_note("Constraint cannot use undeclared parameter.")
      raise

# =================================================================================================

# TODO: add functionality to mask data, and keep track of which points were masked.
#         that way can provide easy access for plotting data with xs for masked points without duplicating manual checks

class HybridFit:
  """
  Fit a model of the form 'scale * (const + a_0 * term_0 + a_1 * term_1 + ...)' to data,
  utilizing a hybrid combination of efficient linear optimization for the a_n coefficients
  and nonlinear optimization for any other parameters in 'scale', 'const', or each 'term_n'. Nonlinear
  parameters (and their default values) in 'scale', 'const', or each 'term_n' are detected from the
  function signatures provided for those expressions.
  """

  # ===============================================================================================

  def __init__(
    self,
    data: Data,
    scale: ExpressionLike = None,
    terms: dict[str, ExpressionLike] = None,
    const: ExpressionLike = None,
    unscaled_terms: dict[str, ExpressionLike] = None,
    unscaled_const: ExpressionLike = None
  ):
    """
    Initialize HybridFit object.

    :param data: Container for measured x-values, y-values, and respective errorbars.
    :param scale: Function or Expression object for multiplicative scale factor outside of linear combination.
    :param const: Function or Expression object for additive term outside of linear combination.
    :param terms: Dictionary mapping coefficient names to functions or Expression objects which form a linear combination.
    :param constraints: Dictionary mapping coefficient names to constraint functions, which take a dictionary of parameter names/values as input, \
                        e.g. {"z": (lambda p: p["x"] + p["y"])} constrains 'z == x + y' wherever 'z' appears in the fit model.
    """
    
    # ensure at least one of scale, terms, or const are provided
    if (scale is None) and (const is None) and (terms is None or len(terms) == 0) and (unscaled_terms is None or len(unscaled_terms) == 0) and (unscaled_const is None):
      raise ValueError("Must provide at least one of 'scale', 'terms', 'const', 'unscaled_terms', or 'unscaled_const.")
    
    self.data = data
    self.metadata = {}

    self.scale = scale if scale is not None else 1
    self.scale = util.ensure_type(self.scale, Expression)

    self.const = const if const is not None else 0
    self.const = util.ensure_type(self.const, Expression)

    self.terms = terms if terms is not None else {}
    self.terms = {name: util.ensure_type(term, Expression) for name, term in self.terms.items()}

    self.unscaled_terms = unscaled_terms if unscaled_terms is not None else {}
    self.unscaled_terms = {name: util.ensure_type(term, Expression) for name, term in self.unscaled_terms.items()}

    self.unscaled_const = unscaled_const if unscaled_const is not None else 0
    self.unscaled_const = util.ensure_type(self.unscaled_const, Expression)

    # internal copies of self.terms and self.const which may be modified to apply parameter constraints
    self._opt_terms = None
    self._opt_const = None
    self._opt_unscaled_terms = None
    self._opt_unscaled_const = None

    # get a list of all nonlinear parameter names, excluding the linear coefficients
    nonlinear_parameters = [
      *self.scale.parameters,
      *self.const.parameters,
      *sum((list(term.parameters) for term in self.terms.values()), []),
      *sum((list(term.parameters) for term in self.unscaled_terms.values()), []),
      *self.unscaled_const.parameters
    ]

    # raise an error if any linear coefficient names also appear in the nonlinear parameter system
    for coeff in [*self.terms, *self.unscaled_terms]:
      if coeff in nonlinear_parameters:
        raise ValueError(f"Linear coefficient '{coeff}' cannot also appear in nonlinear parameters.")

    # merge parameter guesses and bounds from all component expressions
    self.parameters = {**self.scale.parameters}
    self.parameters.update({k: v for k, v in [*self.const.parameters.items(), *self.unscaled_const.parameters.items()] if k not in self.parameters})
    self.bounds = {**self.scale.bounds, **self.const.bounds, **self.unscaled_const.bounds}
    for name, term in [*self.unscaled_terms.items(), *self.terms.items()]:
      self.parameters.update({name: 0, **{k: v for k, v in term.parameters.items() if k not in self.parameters}})
      self.bounds.update(term.bounds)

    # print(self.parameters)
    
    # mapping of nonlinear parameter(s) to Constraint(s), which only allows other nonlinear parameters
    self._nonlinear_constraints = {}

    # mapping of linear parameters to Constraint(s), which only allows linear combinations of unconstrained linear parameters
    self._linear_constraints = {}

    self.cost = LeastSquares(data.x, data.y, data.y_err, self._opt_call)

    self.minuit = iminuit.Minuit(
      self.cost,
      list(self.parameters.values()),
      name = list(self.parameters.keys())
    )

    self.minuit.strategy = 1
    self.minuit.print_level = 1
    self.minuit.tol = 1 # changes EDM from default 0.0002 to 0.002, since 0.0002 is very close to machine limit at typical chi2s for my use case

    # apply parameter bounds to minuit
    for name, value in self.bounds.items():
      self.minuit.limits[name] = value

    # fix linear parameters and those with constraints
    for name in [*self.terms, *self.unscaled_terms]:
      self.minuit.fixed[name] = True

    # internal tracking of fixed/floating parameters, since floating linear & constrained parameters are treated as 'fixed' by minuit
    self.fixed = {name: False for name in self.parameters}
    self.keep_fixed = set()

    self._fit_linear = True

    # covariance matrix of fit parameters
    self.cov = None

    # condition number of linear fit matrix
    self.cond = None

    # extra parameters to include in NDF
    self.ndf_modifier = 0

  # ===============================================================================================
  
  def _fix(self, name: str, value: float = None, permanent = False):
    self.fixed[name] = True
    if (name not in self.terms) and (name not in self.unscaled_terms) and (name not in self._nonlinear_constraints):
      self.minuit.fixed[name] = True
    if value is not None:
      self.minuit.values[name] = value
    if permanent:
      self.keep_fixed.add(name)

  def fix(self, *options: str | dict[str, float], permanent = False):
    """
    Fix given parameter(s) at current value(s). Supports Unix-style wildcards, e.g. "x_*" will fix
    "x_1", "x_2", etc. Constrained parameter values will be frozen too.
    """
    if len(options) == 0:
      for parameter in self.parameters:
        self._fix(parameter, permanent = permanent)
    else:
      for option in options:
        if not isinstance(option, dict):
          option = {option: None}
        for name, value in option.items():
          # allow redirection to self.constrain() if Constraint object is provided
          if isinstance(value, (Constraint, dict)):
            self.constrain({name: value})
            continue
          for parameter in self.parameters:
            if fnmatch.fnmatch(parameter, name):
              self._fix(parameter, value, permanent)

  # ===============================================================================================
  
  def _free(self, name: str, override = False):
    if override:
      self.keep_fixed.discard(name)
    if name not in self.keep_fixed:
      self.fixed[name] = False
      if (name not in self.terms) and (name not in self.unscaled_terms) and (name not in self._nonlinear_constraints):
        self.minuit.fixed[name] = False

  def free(self, *names: str, override = False):
    """
    Allow given parameter(s) to float, after previously being fixed. Supports Unix-style wildcards,
    e.g. "x_*" will free "x_1", "x_2", etc. For constrained parameters, constraints will resume
    after freeing. To remove constraints, use 'HybridFit.unconstrain'.
    """
    if len(names) == 0:
      for parameter in self.parameters:
        self._free(parameter, override)
    else:
      for name in names:
        for parameter in self.parameters:
          if fnmatch.fnmatch(parameter, name):
            self._free(parameter, override)

  # ===============================================================================================

  def mask(self, condition: np.ndarray):
    self.cost.mask(condition)

  def unmask(self):
    self.cost.mask = None

  # ===============================================================================================
        
  def constrain(self, constraints: dict[str, Constraint | dict[str, Constraint]]):
    """Constrain given parameters as functions of other unconstrained parameters."""

    for name, constraint in constraints.items():

      # Simple, singular constraints may be applied to any parameter, as long as the constraint
      # only involves nonlinear parameters.
      if (name not in self.terms) and (name not in self.unscaled_terms) and isinstance(constraint, Constraint):

        if any([dependency in self.terms or dependency in self.unscaled_terms for dependency in constraint.parameters]):
          raise ValueError("Nonlinear constraints may only involve nonlinear parameters.")
        
        self._nonlinear_constraints[name] = constraint
        self.minuit.fixed[name] = True
        self.minuit.errors[name] = 0

      # Linear parameters may be constrained as linear combinations of other linear parameters,
      # supplied as dict[str, Constraint] which maps each linear parameter name in the combination
      # to a constrained coefficient (which may involve nonlinear parameters).
      # e.g. self.constrain({"a_k": {"a_0": c_0, "a_1": c_1, ...}}) applies the constraint
      # a_k = a_0 * c_0(n) + a_1 * c_1(n) + ..., where 'n' is the nonlinear parameter system.
      elif (name in self.terms) or (name in self.unscaled_terms) and isinstance(constraint, dict):
        
        for linear_param in constraint:

          coefficient = util.ensure_type(constraint[linear_param], Expression)
          constraint[linear_param] = coefficient

          if ((linear_param not in self.terms) and (linear_param not in self.unscaled_terms)) or (linear_param in self._linear_constraints):
            raise ValueError(f"Parameter '{linear_param}' is not a linear parameter, or is already constrained.")
          
          if any([dependency in self.terms or dependency in self.unscaled_terms for dependency in coefficient.parameters]):
            raise ValueError("Coefficient in constrained linear combination may only involve nonlinear parameters.")
          
        self._linear_constraints[name] = constraint

      else:

        raise ValueError(f"Unrecognized constraint format for parameter '{name}'.")

      # applying a constraint implies the parameter should float with the constraint, so free it if previously fixed
      self.free(name, override = True)

    # sort nonlinear constraint dictionary so that nested constraints are applied last, after dependencies updated
    independent_constraints = {}
    dependent_constraints = {}
    for name, constraint in self._nonlinear_constraints.items():
      dependent = False
      for dependency in constraint.parameters:
        if dependency in self._nonlinear_constraints:
          dependent_constraints[name] = constraint
          dependent = True
          break 
      if not dependent:
        independent_constraints[name] = constraint
    self._nonlinear_constraints = {**independent_constraints, **dependent_constraints}

  # ===============================================================================================
      
  def unconstrain(self, *names):
    """Remove any constraint relationships from given parameter names."""
    if len(names) == 0:
      names = self.minuit.parameters
    for name in names:
      if name in self._nonlinear_constraints:
        del self._nonlinear_constraints[name]
        if not self.fixed[name]:
          self.minuit.fixed[name] = False
      elif name in self._linear_constraints:
        del self._linear_constraints[name]

  # ===============================================================================================
      
  def unconstrain_dependencies(self, *names):
    """Remove any constraint relationships for dependent parameters that are constrained by the given names."""
    
    for name in names:

      for p, constraint in list(self._nonlinear_constraints.items()):
        if name in constraint.parameters:# and not self.fixed[p]:
          self.unconstrain(p)

      for p, constraint in list(self._linear_constraints.items()):
        if name in constraint:# and not self.fixed[p]:
          self.unconstrain(p)

  # ===============================================================================================
        
  def is_constrained(self, name):
    """Checks if the given parameter is currently constrained."""
    return (name in self._nonlinear_constraints) or (name in self._linear_constraints)

  # ===============================================================================================

  def has_dependencies(self, name):

    has_nonlinear_dependencies = False
    for p, constraint in self._nonlinear_constraints.items():
      if name in constraint.parameters and not self.fixed[p]:
        has_nonlinear_dependencies = True
        break

    has_linear_dependencies = False
    for p, constraint in self._linear_constraints.items():
      if name in constraint and not self.fixed[p]:
        has_linear_dependencies = True
        break

    return (has_nonlinear_dependencies or has_linear_dependencies)

  # ===============================================================================================

  def is_floating(self, name):
    return not self.fixed[name] and not self.is_constrained(name)

  # ===============================================================================================

  # TODO: change this to not use the errors anymore, but the bound range?
  # TODO: it's okay for errors to be very large, just don't want central value to be close to edge!
  def at_limit(self, name, proximity = None, error_scale = 1, left = True, right = True):
    """Checks if the given parameter is stuck at either its lower or upper limit."""

    if self.fixed[name] or self.is_constrained(name):
      return False
    
    value = self.minuit.values[name]
    error = self.minuit.errors[name]
    
    # # otherwise, take 0.1% of bounded range as a heuristic scale for closeness to boundary
    # if not any(np.isinf(limit) for limit in self.minuit.limits[name]):
    #   limit_range = self.minuit.limits[name][1] - self.minuit.limits[name][0]
    #   heuristic_limit = 0.001 * limit_range
    # # otherwise, take 1% of the value
    # else:
    #   heuristic_limit = 0.01 * value

    left_proximity, right_proximity = None, None
    if proximity is not None:
      if isinstance(proximity, tuple):
        left_proximity, right_proximity = proximity
      else:
        left_proximity = proximity
        right_proximity = proximity

    limit_cases = []
    if left:
      limit_cases.append((self.minuit.limits[name][0], left_proximity))
    if right:
      limit_cases.append((self.minuit.limits[name][1], right_proximity))

    for limit, proximity in limit_cases:
      diff = abs(value - limit)
      # errorbars reach limit, and central value also within 20% of limit (heuristic)
      if proximity is None:
        proximity = 0.1 * abs(limit)
        # proximity = np.inf
      if diff <= error_scale * error and (diff <= proximity):
        return True
    
    return False

  # ===============================================================================================

  def _sum_terms(self, p: Mapping[str, float] = None, t: np.ndarray = None) -> np.ndarray:
    term_total = 0
    for name, term in self.terms.items():
      if p[name] != 0:
        term_total += p[name] * term.eval(p, t)
    return term_total
  
  def _sum_opt_terms(self, p: Mapping[str, float] = None, t: np.ndarray = None) -> np.ndarray:
    return sum(p[name] * term.eval(p, t) for name, term in self._opt_terms.items())

  def _sum_unscaled_terms(self, p: Mapping[str, float] = None, t: np.ndarray = None) -> np.ndarray:
    term_total = 0
    for name, term in self.unscaled_terms.items():
      if p[name] != 0:
        term_total += p[name] * term.eval(p, t)
    return term_total
  
  def _sum_opt_unscaled_terms(self, p: Mapping[str, float] = None, t: np.ndarray = None) -> np.ndarray:
    return sum(p[name] * term.eval(p, t) for name, term in self._opt_unscaled_terms.items())

  # TODO: reverse p, t order here?
  def __call__(self, p: Mapping[str, float] = None, t: np.ndarray = None) -> np.ndarray:
    """
    Evaluate the model using the given dictionary of parameter values and independent variable.
    Defaults to current internal state of parameter system and data points if not supplied.
    """

    self.clear_caches()

    if p is None:
      p = self.minuit.values
    if t is None:
      t = self.data.x

    scale, const, unscaled_const = self.scale.eval(p, t), self.const.eval(p, t), self.unscaled_const.eval(p, t)
    return scale * (const + self._sum_terms(p, t)) + self._sum_unscaled_terms(p, t) + unscaled_const
  
  # ===============================================================================================

  def _fit_linear_combination(self, p: Mapping[str, float], t: np.ndarray, return_cond: bool = False, retry: bool = True) -> dict[str, float]:
    """
    Returns a dictionary of best-fit coefficients for the terms in the linear combination
    which are currently floating.
    """

    y = self.data.y if self.cost.mask is None else self.data.y[self.cost.mask]
    err = self.data.y_err if self.cost.mask is None else self.data.y_err[self.cost.mask]

    scale, const, unscaled_const = self.scale.eval(p, t), self._opt_const.eval(p, t), self._opt_unscaled_const.eval(p, t)
    
    try:
      # result = util.fit_linear_combination(
      #   [term.eval(p, t) for term in self._opt_terms.values()],
      #   y / scale - const, # must remove scale and const models from data to isolate linear combination
      #   err / scale, # dividing by scale model also scales data errorbars, but not subtracting const
      #   return_cond = return_cond
      # )
      result = util.fit_linear_combination(
        [term.eval(p, t) * scale for term in self._opt_terms.values()] + [term.eval(p, t) for term in self._opt_unscaled_terms.values()],
        y - scale * const - unscaled_const, # must remove scale and const models from data to isolate linear combination
        err, # dividing by scale model also scales data errorbars, but not subtracting const
        return_cond = return_cond
      )

    except np.linalg.LinAlgError as e:

      print("Failed to solve matrix system. Print current state of parameter system.")
      for p, val in p.items():
        print(f"{p}: {val:.6f}")
      raise e
    
    except ValueError as e:

      if not retry:
        raise e
      
      retry = False
      for parameter in self.parameters:
        if np.isnan(p[parameter]) or np.isnan(self.minuit.errors[parameter]) and self.is_floating(parameter):
          print(f"Floating parameter was NaN. Resetting {parameter} = {self.parameters[parameter]} and trying again.")
          self.minuit.values[parameter] = self.parameters[parameter]
          self.minuit.errors[parameter] = 0
          p[parameter] = self.parameters[parameter]
          retry = True

      if retry:
        return self._fit_linear_combination(p, t, return_cond, retry = False)
      else:
        print("Nothing in fit.minuit.values/errors was NaN.")
        print("Failed to solve matrix system. Print current state of parameter system.")
        for p, val in p.items():
          print(f"{p}: {val:.6f}")
        raise e

    if return_cond:
      coeffs, linear_combination, cond = result
      self.cond = cond
    else:
      coeffs, linear_combination = result

    return dict(zip(list(self._opt_terms) + list(self._opt_unscaled_terms), coeffs)), linear_combination + scale * const + unscaled_const
  
  # ===============================================================================================

  def _opt_call(self, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Internal wrapping of model evaluation which optimizes linear parameters, applies constraints, and
    uses caching to speed up model evaluation when some parameters are unchanged.
    """

    # wrap parameter array into dictionary with parameter names
    p = dict(zip(self.parameters, p))
    
    # update nonlinear constrained parameters
    for name, constraint in self._nonlinear_constraints.items():
      if not self.fixed[name]:
        p[name] = constraint(p)

    # optimize linear coefficient parameters
    result = None
    if self._fit_linear:
      coeffs, result = self._fit_linear_combination(p, t)
      p.update(coeffs)
    else:
      scale = self.scale.eval(p, t)
      const = self._opt_const.eval(p, t)
      unscaled_const = self._opt_unscaled_const.eval(p, t)
      result = scale * (const + self._sum_opt_terms(p, t)) + self._sum_opt_unscaled_terms(p, t) + unscaled_const
      if result.shape != self.data.y.shape:
        # broadcast to match data shape if needed (in edge cases where everything was just a number)
        result = result * np.ones(self.data.y.shape)

    # update linear constrained parameters
    for name, constraint in self._linear_constraints.items():
      if not self.fixed[name]:
        p[name] = sum(
          p[linear_param] * coeff_constraint(self.minuit.values)
          for linear_param, coeff_constraint in constraint.items()
        )
  
    return result
  
  # ===============================================================================================

  def hesse(self):
    # release floating linear parameters in minuit system, call HESSE, then re-fix them
    floating_coeffs = [name for name in [*self._opt_terms, *self._opt_unscaled_terms]]
    if len(floating_coeffs) > 0:
      self.minuit.fixed[*floating_coeffs] = False
    self._fit_linear = False
    self.minuit.hesse()
    if len(floating_coeffs) > 0:
      self.minuit.fixed[*floating_coeffs] = True
    self._fit_linear = True

  # ===============================================================================================

  def fit(self, verbose = True, max_iterations = 3, hesse = True, print_only_floating = False, check_limits = True):
    """Runs chi-squared minimization and covariance estimation for floating parameters."""

    self._opt_terms = {name: [value] for name, value in self.terms.items()}
    self._opt_const = [self.const]

    self._opt_unscaled_terms = {name: [value] for name, value in self.unscaled_terms.items()}
    self._opt_unscaled_const = [self.unscaled_const]
   
    # absorb fixed terms as part of the constant with no coefficient, and remove from system
    for name in self.terms:
      if self.fixed[name]:
        if self.minuit.values[name] != 0:
          self._opt_const.append(Product(self.minuit.values[name], self.terms[name]))
        del self._opt_terms[name]

    for name in self.unscaled_terms:
      if self.fixed[name]:
        if self.minuit.values[name] != 0:
          self._opt_unscaled_const.append(Product(self.minuit.values[name], self.unscaled_terms[name]))
        del self._opt_unscaled_terms[name]

    # modify system of terms according to any linear constraints
    # linear combination in fit model has the form:
    #   a_0 * f_0 + a_1 * f_1 + ... + a_k * f_k + ...
    # if we replace a_k => (a_0 * c_0 + a_1 * c_1 + ...), then linear combination becomes:
    #   a_0 * (f_0 + c_0 * f_k) + a_1 * (f_1 + c_1 * f_k) + ...
    # so each remaining term picks up scaled f_k term from linearly-constrained a_k
    for name in self.terms:

      # if constrained parameter is still in the system (i.e. wasn't fixed)...
      if (name in self._linear_constraints) and (name in self._opt_terms):

        for linear_param, coeff_constraint in self._linear_constraints[name].items():
          # ensure each dependent parameter is still in the constrained system

          #if linear_param not in self._opt_terms:
          #  raise ValueError(f"Linear parameter '{linear_param}' must be floating unconstrained in order to constrain '{name}'.")

          # update term associated with each dependent parameter
          if linear_param in self._opt_terms:
            self._opt_terms[linear_param].append(Product(coeff_constraint, self.terms[name]))
          else:
            # parameter being constrained to was fixed, so this term should go in _opt_const
            self._opt_const.append(Product(self.minuit.values[linear_param], coeff_constraint, self.terms[name]))

        # remove constrained parameter from the constrained linear system
        del self._opt_terms[name]

    for name in self.unscaled_terms:
      if (name in self._linear_constraints) and (name in self._opt_unscaled_terms):
        for linear_param, coeff_constraint in self._linear_constraints[name].items():
          if linear_param in self._opt_unscaled_terms:
            self._opt_unscaled_terms[linear_param].append(Product(coeff_constraint, self.unscaled_terms[name]))
          else:
            self._opt_unscaled_const.append(Product(self.minuit.values[linear_param], coeff_constraint, self.unscaled_terms[name]))
        del self._opt_unscaled_terms[name]

    for name in list(self._opt_terms.keys()):
      self._opt_terms[name] = Sum(*self._opt_terms[name])
    self._opt_const = Sum(*self._opt_const)

    for name in list(self._opt_unscaled_terms.keys()):
      self._opt_unscaled_terms[name] = Sum(*self._opt_unscaled_terms[name])
    self._opt_unscaled_const = Sum(*self._opt_unscaled_const)

    iterations = 0
    success = False
    prev_chi2 = 0
    while not success and iterations < max_iterations:

      # on repeat attempts, also reset any parameters at limits
      if iterations > 0:
        for p in self.parameters:
          if self.at_limit(p):
            print(f"Resetting {p} to default value: {self.parameters[p]}.")
            self.minuit.values[p] = self.parameters[p]
            self.minuit.errors[p] = 0
        self.minuit.simplex()
        
      self.minuit.migrad()

      # update nonlinear constrained parameters in minuit results
      for name, constraint in self._nonlinear_constraints.items():
        if not self.fixed[name]:
          self.minuit.values[name] = constraint(self.minuit.values, None)

      # update linear parameters in minuit results
      coeffs, function_eval = self._fit_linear_combination(
        self.minuit.values,
        self.data.x if self.cost.mask is None else self.data.x[self.cost.mask],
        return_cond = True
      )

      if len(coeffs) > 0:
        self.minuit.values[*coeffs.keys()] = coeffs.values()

      # update linear constrained parameters in minuit results
      for name, constraint in self._linear_constraints.items():
        if not self.fixed[name]:
          self.minuit.values[name] = sum(
            self.minuit.values[a] * coeff(self.minuit.values, None)
            for a, coeff in constraint.items()
          )

      if hesse:
        if verbose:
          print("Running HESSE.")
        self.hesse()

      #self.cov = self.minuit.covariance.to_dict()
      self.cov = self.minuit.covariance
      
      #for parameter in self.minuit.parameters:
      #  for other in self.minuit.parameters:
      #    if (parameter, other) not in self.cov:
      #      self.cov[parameter, other] = self.cov[other, parameter]

      #self.cov = {p1: {p2: self.cov[p1, p2] for p2 in self.minuit.parameters} for p1 in self.minuit.parameters}

      self.errors = self.minuit.errors.to_dict()

      for parameter in self.minuit.parameters:
        if self.fixed[parameter] or self.is_constrained(parameter):
          self.errors[parameter] = 0
          #for other in self.minuit.parameters:
          #  self.cov[parameter, other] = 0
     
      # TODO: is it right to not count fixed parameters in NDF after some rounds of optimizing them?
      # TODO: sometimes chi2 goes down a little, but chi2/ndf goes up a little after freeing lots of parameters in last step
      self.npar = (self.minuit.nfit + len(self._opt_terms) + len(self._opt_unscaled_terms)) + self.ndf_modifier
      self.ndf = self.cost.ndata - self.npar
      self.chi2 = self.minuit.fval
      self.chi2_err = np.sqrt(2 * self.ndf)
      self.chi2ndf = self.chi2 / self.ndf
      self.chi2ndf_err = self.chi2_err / self.ndf
      self.pval = util.p_value(self.chi2, self.ndf)
      self.curve = function_eval #self(self.minuit.values, self.data.x)

      if verbose:
        self.print(only_floating = print_only_floating)

      # re-evaluate fit quality after running HESSE
      success = self.minuit.fmin.is_valid and self.minuit.fmin.has_accurate_covar
      if check_limits:
        success = success and not any(self.at_limit(p) for p in self.parameters)
      success = success or all(self.minuit.fixed)

      if not success:
        if verbose:
          print(f"Minuit is unhappy with fit validity after HESSE. Repeating minimization.")
        # if abs(self.chi2 - prev_chi2) < 1E-6:
        #   print(f"Chi-squared is almost identical after iterating. Further iteration unlikely to help. Breaking early.")
        #   break
        # else:
        iterations += 1
        prev_chi2 = self.chi2

    # for p in self.parameters:
    #   if np.isnan(self.minuit.errors[p]):
    #     print(f"WARNING: parameter {p} error was NaN, resetting to default heuristic.")
    #     # self.minuit.errors[p] = 0
    #     self.hesse()


      # for i in range(len(self.parameters)):
      #   for j in range(i, len(self.parameters)):
      #     corr = self.get_correlation(i, j)
      #     if corr < -0.9:
      #       print(f"WARNING: {self.minuit.parameters[i]} and {self.minuit.parameters[j]} are strongly anticorrelated: {corr:.4f}.")

  # ------------------------------------------------------------------------------------------------
        
  def get_correlation(self, p, q):
    if self.is_floating(p) and self.is_floating(q):
      return self.minuit.covariance[p, q] / (self.minuit.errors[p] * self.minuit.errors[q])
    else:
      return None

  # ================================================================================================
    
  def check_minimum(self, verbose = False):
    """
    Checks for valid minimum based on the following criteria:
    - sufficiently small estimated-distance-to-minimum (EDM),
    - accurate covariance matrix estimation,
    - no parameters stuck at limits,
    - function call limit not exceeded.
    Returns True if all criteria are satisfied, or else False.
    Optionally prints warnings for violated criteria if 'verbose' is True.
    """
    success = True
    if not self.minuit.fmin.is_valid:
      if verbose:
        print("MINIMIZATION DID NOT CONVERGE!")
        if self.minuit.fmin.edm > self.minuit.fmin.edm_goal:
          print(f"EDM (chi2 - chi2_min) ~ {self.minuit.fmin.edm:.4f} exceeds EDM goal of {self.minuit.fmin.edm_goal:.4f}.")
      success = False
    if not self.minuit.fmin.has_accurate_covar:
      if verbose:
        print("Covariance may not be accurate.")
      success = False
    if any(self.at_limit(p) for p in self.parameters):
      if verbose:
        print("Parameter(s) stuck at limits.")
      success = False
    if self.minuit.fmin.has_reached_call_limit:
      if verbose:
        print("Function call limit exceeded.")
      success = False
    if success and verbose:
      print("Minimization was successful.")
    return success

  # ================================================================================================
    
  def print(self, only_floating = False):
    """Prints fit convergence status/warnings, table of parameter information, and chi-squared/p-value."""

    print("\n")
    try:
      self.check_minimum(verbose = True)
      print(f"Time spent: {self.minuit.fmin.time:.6f} seconds.")
    except:
      print("Fit has not yet been optimized.")

    headers = ["index", "name", "value", "error", "limit-", "limit+", "type", "status"]
    rows = []

    for i, name in enumerate(self.minuit.parameters):
      # if name.startswith("center") and self.fixed[name]:
      #   continue
      value = self.minuit.values[name]
      if not self.is_floating(name) and (value == 0 or np.isinf(value) or only_floating):
        continue
      error = self.minuit.errors[name]
      error_order = util.order_of_magnitude(error)
      # TODO: sort out decimal/scientific notation appearance based on sig figs / desired order limits
      decimals = 4
      if error_order < 0:
        new_decimals = abs(error_order) + 3
        if new_decimals > decimals:
          decimals = new_decimals
      rows.append([
        i,
        name,
        f"{value:.{decimals}e}",
        # value,
        f"{error:.{decimals}e}" if not self.fixed[name] and not self.is_constrained(name) else "",
        # error if not self.fixed[name] and not self.is_constrained(name) else np.nan,
        f"{self.minuit.limits[name][0]:.4f}" if not np.isinf(self.minuit.limits[name][0]) else "",
        f"{self.minuit.limits[name][1]:.4f}" if not np.isinf(self.minuit.limits[name][1]) else "",
        "linear" if name in self.terms else "",
        "fixed" if self.fixed[name] else ("constr." if self.is_constrained(name) else "")
      ])

    print(tab.tabulate(rows, headers, tablefmt = "grid", numalign = "left"))
    try:
      print(f"cond = {(self.cond if self.cond is not None else 1):.1f}")
      print(f"npar = {self.npar}")
      print(f"chi2 = {self.chi2:.4f} +/- {self.chi2_err:.4f}")
      print(f"chi2/ndf = {self.chi2ndf:.4f} +/- {self.chi2ndf_err:.4f}")
      pval_format = ".4f" if util.order_of_magnitude(self.pval) > -4 else ".1e"
      print(f"p-value = {self.pval:{pval_format}}")
    except:
      pass
    print("\n")

  # ===============================================================================================
    
  def statbox(self):
    """Returns list of math-formatted strings to display chi2/ndf and p-value on a plot."""
    return [
      rf"$\chi^2$/ndf = {self.chi2ndf:.4f} $\pm$ {self.chi2ndf_err:.4f}",
      rf"$p$ = {self.pval:.2f}"
    ]

  # ===============================================================================================

  def pulls(self):
    """Calculate and return the array of fit pulls [(y - f(x)) / y_err]."""
    return (self.data.y - self.curve) / self.data.y_err

  # ===============================================================================================

  def fft(self, bounds = None):
    """
    Calculate and return the FFT of the fit pulls, scaled as units of the fit's chi2.
    Based on Parseval's theorem: chi^2 = sum(pulls^2) = sum(|FFT|^2) / len(FFT), so the entries of
    |FFT|^2 / len(FFT) yield each FFT frequency bin's contribution to the chi^2.
    """
    pulls = self.pulls()
    if bounds is not None:
      pulls_mask = (self.data.x > bounds[0]) & (self.data.x < bounds[1])
      pulls = pulls[pulls_mask]
    # Only want frequencies up to the Nyquist frequency, so use np.fft.rfft.
    fft = np.abs(np.fft.rfft(pulls))**2 / len(pulls)
    # But Parseval's theorem includes all FFT bins, including those in the 2nd mirrored half.
    # So double the FFT power in the non-zero bins that would have been counted twice in the chi^2.
    fft[1:] *= 2
    frequencies = np.fft.rfftfreq(len(pulls), self.data.x[1] - self.data.x[0])
    return frequencies, fft

  # ===============================================================================================

  def local_frequency_chi2(self, frequency, width):
    f, fft = self.fft()
    select_local_f = (f > frequency - width) & (f < frequency + width)
    return np.sum(fft[select_local_f])
  
  def bounded_chi2(self, start, end):
    select_x = (self.data.x > start) & (self.data.x < end)
    return np.sum(self.pulls()[select_x]**2)

  # ===============================================================================================

  def f_test(
    self,
    parameters: dict[str, float | Constraint],
    defaults = None,
    return_details = False,
    disable_on_failure = False,
    max_iterations = 1,
    threshold = 3,
    verbose = False,
    hesse = False,
    label = None,
    check_parameters = None,
    check_limits = True,
    limit_proximities = None,
    quiet = False,
    test_early = 0,
    test_frequency = None,
    force_pass = False
  ) -> float:

    # hesse = True

    if len(parameters) == 0:
      return False
    
    if limit_proximities is None:
      limit_proximities = {}

    if check_parameters is None:
      check_parameters = []
    else:
      # make a copy so as not to modify in-place
      check_parameters = [*check_parameters]

    if defaults is None:
      defaults = {}
    for p in list(defaults.keys()):
      defaults[p] = util.ensure_list(defaults[p])

    # This seems to happen occasionally, so including a check for it.
    for name, val, err in zip(self.minuit.parameters, self.minuit.values, self.minuit.errors):
      if np.isnan(val):
        print(f"WARNING! {name} was NaN before F-test. Setting to default: {self.parameters[name]}.")
        self.minuit.values[name] = self.parameters[name]
      if np.isnan(err):
        print(f"WARNING! {name} error was NaN before F-test. Setting to default heuristic.")
        self.minuit.errors[name] = 0

    prev_values, prev_errors = np.copy(self.minuit.values), np.copy(self.minuit.errors)

    self.fix(parameters)
    self.fit(verbose = verbose, hesse = hesse, max_iterations = max_iterations)
    
    # prev_values, prev_errors = np.copy(self.minuit.values), np.copy(self.minuit.errors)

    if label:
      print(f"Beginning F-test for {label}. {self.npar} parameters floating.")
    for p, val in parameters.items():
      if isinstance(val, Constraint):
        if not quiet:
          print(f"Constraining {p} in terms of {list(val.parameters.keys())}, currently {[f'{self.minuit.values[q]:.4f}' for q in val.parameters]}.")
        for q in val.parameters:
          if q not in check_parameters:
            check_parameters.append(q)

    prev_chi2 = self.chi2
    prev_cond = self.cond
    null_parameters = {p: self.minuit.values[p] for p in [*parameters, *check_parameters] if p in self.parameters}
    null_errors = {p: self.minuit.errors[p] for p in [*parameters, *check_parameters] if p in self.parameters}
    null_stuck_at_limits = {p: self.at_limit(p, proximity = limit_proximities.get(p, None)) for p in [*parameters, *check_parameters] if p in self.parameters}

    f_before, fft_before = self.fft()

    if test_early > 0:
      before_windowed_chi2 = []
      for window in range(1, 10):
        window_select = (self.data.x >= self.data.x[0] + (window - 1) * test_early) & (self.data.x <= self.data.x[0] + window * test_early)
        before_windowed_chi2.append(np.sum(((self.data.y[window_select] - self()[window_select]) / self.data.y_err[window_select])**2))

    # if prev_chi2 > 10000:
    #   self.print()
    #   exit()

    self.unconstrain(*parameters)
    self.free(*parameters, override = True)

    for default_case in itertools.product(*defaults.values()):

      default_values = dict(zip(defaults.keys(), default_case))
      success = False
      count = 0

      try:

        while not success and count <= 1:

          self.minuit.values = prev_values
          self.minuit.errors = prev_errors

          for name, value in default_values.items():
            # if name in parameters:# and not self.fixed[name]:
            if self.is_floating(name):
              if not quiet:
                print(f"Setting default: {name} = {value:.4f}")
              self.minuit.values[name] = value
              self.minuit.errors[name] = 0

          # print(f"Initial state before test fit:")
          # self.print()
        
          self.fit(verbose = verbose, hesse = hesse, max_iterations = max_iterations)
          # self.fit(verbose = verbose, hesse = True, max_iterations = max_iterations)
       
          new_chi2 = self.chi2
          new_cond = self.cond
          test_parameters = {p: self.minuit.values[p] for p in null_parameters}
          test_errors = {p: self.minuit.errors[p] for p in null_parameters}
          test_stuck_at_limits = {p: self.at_limit(p, proximity = limit_proximities.get(p, None)) for p in null_stuck_at_limits}

          # if (new_chi2 > prev_chi2):
          #   print(f"WARNING: chi2 worse after releasing test parameters. Running HESSE and trying again.")
          #   self.hesse()
          #   count += 1
          # else:
          success = True
            # TODO: add HybridFit.reset_stuck_limits() to reset parameters that got stuck to initial guesses

          #ndf_num = len(parameters)
          #ndf_den = self.ndf

          #f = ((prev_chi2 - new_chi2) / ndf_num) / (new_chi2 / ndf_den)
          #f_pval = stats.f.sf(f, ndf_num, ndf_den)
          #f_pval = stats.chi2.sf(prev_chi2 - new_chi2, len(parameters))

      except (np.linalg.LinAlgError, ValueError):

        print(f"Minimization failed due to singular matrix or invalid parameter values.")
        new_chi2 = prev_chi2
        new_cond = 1
        test_parameters = null_parameters
        test_errors = null_errors
        test_stuck_at_limits = null_stuck_at_limits
        #f_pval = 1

      #passed_test = (f_pval < threshold)
      passed_test = (prev_chi2 - new_chi2) > (threshold * len(parameters) if not force_pass else 0)# or force_pass
      stuck_at_limits = any(test_stuck_at_limits[p] and not null_stuck_at_limits[p] for p in test_stuck_at_limits)
      # stuck_at_limits = any(test_stuck_at_limits[p] for p in test_stuck_at_limits)
      #good_condition = (new_cond < 1E4 or new_cond / prev_cond < 100) if None not in [prev_cond, new_cond] else True
      # good_condition = (new_cond / prev_cond < 1000) if None not in [prev_cond, new_cond] else True
      good_condition = True

      passed_local = True
      if test_frequency is not None:# and not force_pass:
        f_after, fft_after = self.fft()
        select_local_f = (f_before > test_frequency[0]) & (f_before < test_frequency[1])
        local_chi2_before = np.sum(fft_before[select_local_f])
        local_chi2_after = np.sum(fft_after[select_local_f])
        local_delta_chi2 = local_chi2_after - local_chi2_before
        print(f"Local-frequency delta(chi2) per parameter: {local_delta_chi2/len(parameters):.4g}.")
        # if local_delta_chi2/len(parameters) > -threshold:
        #   passed_local = False
        #   print(f"Local-frequency delta(chi2) failed test. Overriding decision.")
        # if (new_chi2 - prev_chi2)/len(parameters) > local_delta_chi2/len(parameters):
        #   passed_local = False
        #   print(f"Global delta(chi2) is worse than local-frequency delta(chi2). Overriding decision.")
        # if "tau_beta_(CBO-a)" in parameters:
        #   plt.plot(f_before, fft_after - fft_before)
        #   plt.show()

      # TODO: make this windowed chi2 a default feature of fitting, just have it always available with configurable window size
      passed_early = True
      if test_early > 0:
        after_windowed_chi2 = []
        for window in range(1, 10):
          window_select = (self.data.x >= self.data.x[0] + (window - 1) * test_early) & (self.data.x <= self.data.x[0] + window * test_early)
          after_windowed_chi2.append(np.sum(((self.data.y[window_select] - self()[window_select]) / self.data.y_err[window_select])**2))
        delta_windowed_chi2 = [float(f'{after - before:.2f}') for after, before in zip(after_windowed_chi2, before_windowed_chi2)]
        print(f"Windowed delta(chi2): {delta_windowed_chi2}")
        early_delta_chi2 = delta_windowed_chi2[0]
        # if early_delta_chi2 / len(parameters) > threshold:
        #   passed_early = False

      passed_test = (passed_test and passed_early and good_condition and passed_local) #and not stuck_at_limits

      # don't apply the stuck-at-limits check for constraints, since we want to unconstrain and re-test independently if it releases but gets stuck
      if not check_limits or (len(parameters) == 1 and isinstance(list(parameters.values())[0], Constraint)):
        pass
      else:
        passed_test = passed_test and not stuck_at_limits

      passed_test = passed_test or force_pass

      if passed_test:
        break

    if not quiet:

      for pset, prefix in [(parameters, ""), (check_parameters, "[info] ")]:
        for p in pset:
          if p not in self.parameters:
            continue
          dp = test_parameters[p] - null_parameters[p]
          dp_p = dp / test_errors[p] if test_errors[p] > 0 else np.nan
          delta_string = f"// {dp:.4f} ({dp_p*100:.2f}%)" if not np.isinf(dp) else ""
          print(f"{prefix}delta({p}): ({null_parameters[p]:.4f} +/- {null_errors[p]:.4f}) --> ({test_parameters[p]:.4f} +/- {test_errors[p]:.4f}) {delta_string}")

    if None not in [prev_cond, new_cond]:
      print(f"delta(cond): {new_cond/prev_cond:.2f}x ({prev_cond:.4f} --> {new_cond:.4f})")
    print(f"delta(chi2): {new_chi2 - prev_chi2:.2e} ({prev_chi2:.4f} --> {new_chi2:.4f})")
    if test_early > 0:
      print(f"delta(chi2)/delta(ndf) in first {test_early} x-units: {early_delta_chi2/len(parameters):.2f}")
      if not passed_early:
        print(f"WARNING: chi2 increased too much in first {test_early} x-units. Overriding decision.")
    # for p, error_frac in check_parameters.items():
    #   if error_frac is not None and null_errors[p] > 0:
    #     dp = test_parameters[p] - null_parameters[p]
    #     dp_p = dp / test_errors[p]
    #     print(f"delta({p}): {dp:.4f} ({dp_p*100:.2f}%, threshold {error_frac*100:.2f}%)")
    #     passed_test = passed_test or (new_chi2 < prev_chi2 and abs(dp_p) > error_frac)

    #print(f"chance of significance: {(1-f_pval)*100:.4f}% (threshold {(1-threshold)*100}%)")
    print(f"delta(chi2) per parameter: {(new_chi2 - prev_chi2)/len(parameters):.2e}")
    if label and stuck_at_limits:
      print(f"WARNING: parameter(s) stuck at limits.")
    if force_pass and passed_test:
      print(f"Passed by force.")

    # if new_chi2 > 10000:
    #   self.print()
    #   exit()
    
    if disable_on_failure and label:
      if not passed_test:
        print(f"FAILED: {label}")
        self.minuit.values = prev_values
        self.minuit.errors = prev_errors
        self.fix(parameters, permanent = True)
        # fitting with hesse after disabling seems to fix some occasional problems with minimum status being bad (despite being exactly the same minimum as when it was good?)
        # self.fit(verbose = False, max_iterations = 1, hesse = True)
        # self.hesse()
      else: 
        print(f"PASSED: {label}")
    
    print("")

    if not return_details:
      return passed_test
    else:
      #return passed_test, f_pval
      return passed_test, new_chi2 - prev_chi2

  # ===============================================================================================

  def results(self, prefix = ""):
    if prefix != "":
      prefix = f"{prefix}_"
    fft_x, fft_y = self.fft()
    return {
      **self.metadata,
      f"{prefix}fit_chi2": self.chi2,
      f"{prefix}fit_chi2_err": self.chi2_err,
      f"{prefix}fit_ndf": self.ndf,
      f"{prefix}fit_npar": self.npar,
      f"{prefix}fit_chi2ndf": self.chi2ndf,
      f"{prefix}fit_chi2ndf_err": self.chi2ndf_err,
      f"{prefix}fit_pvalue": self.pval,
      **{f"{prefix}{name}": value for name, value in self.minuit.values.to_dict().items()},
      **{f"{prefix}{name}_err": error for name, error in self.errors.items()},
      #**{f"{prefix}{name}_err": 0 for name in self.minuit.parameters if self.is_constrained(name)},
      #**{f"{prefix}{name}_err": 0 for name in self.minuit.parameters if self.fixed[name]},
      **{f"{prefix}{name}_valid": (not self.at_limit(name) and self.minuit.fmin.has_accurate_covar) for name in self.minuit.parameters},
      f"{prefix}fit_x": self.data.x,
      f"{prefix}fit_y": self.data.y,
      f"{prefix}fit_y_err": self.data.y_err,
      f"{prefix}fit_curve": self.curve,
      f"{prefix}fit_residuals": self.data.y - self.curve,
      f"{prefix}fit_pulls": self.pulls(),
      f"{prefix}fit_fft_x": fft_x,
      f"{prefix}fit_fft_y": fft_y,
      f"{prefix}fit_converged": self.minuit.fmin.is_valid,
      f"{prefix}err_accurate": self.minuit.fmin.has_accurate_covar,
      f"{prefix}fit_valid": self.check_minimum(verbose = False),
      f"{prefix}cov": np.array(self.cov),
      f"{prefix}cov_labels": list(self.minuit.parameters)
      #f"{prefix}cov_labels": ",".join(self.minuit.parameters),
      #f"{prefix}cov": np.array([self.cov[i, j] for j in self.minuit.parameters for i in self.minuit.parameters])
    }

  # ================================================================================================

  def clear_caches(self):
    """Clear numerical caches in Expression objects to save space when pickling."""
    self.scale.clear_cache()
    self.const.clear_cache()
    self.unscaled_const.clear_cache()
    for term in [*self.terms.values(), *self.unscaled_terms.values()]:
      term.clear_cache()

  # ================================================================================================

  def save(self, filename: str):
    """Wrapper for util.save which clears HybridFit's numerical caches to save space."""
    self.clear_caches()
    util.save(self, filename)
