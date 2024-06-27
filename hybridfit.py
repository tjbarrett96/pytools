import numpy as np
from typing import Mapping, Callable
from dataclasses import dataclass
import functools

import iminuit
from iminuit.cost import LeastSquares

import pytools.util as util
from pytools.data import Data

import tabulate as tab

import inspect
import fnmatch
import numpy as np
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
      signature = {}
    else:
      self._function = function
      # get list of parameter names from function signature
      signature = inspect.signature(function).parameters if function is not None else {}

    # TODO: lru_cache does not work with numpy array argument. is there a workaround?
    #if self._function is not None:
    #  cache_wrapper = functools.lru_cache(maxsize = 1)
    #  self._function = cache_wrapper(self._function)

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

    self.t_cache = None
    self.p_cache = None
    self.val_cache = None

    # list of references to sub-expressions involved in adding or multiplying
    self._sub_exprs = []

  # ===============================================================================================

  def add_label(self, label: str) -> None:
    """Append underscore and label string to each parameter name."""
    if label is not None:
      self.parameters = {f"{name}_{label}": value for name, value in self.parameters.items()}
      self.bounds = {f"{name}_{label}": value for name, value in self.bounds.items()}
    return self

  # ===============================================================================================
 
  # TODO: add strict_cache = True option to check 't' element-wise, disable for looser size/range checking
  # TODO: need to keep this eval to use special arg, instead of replacing in __mul__ etc. below
  # TODO: so modification to __add__ and __mul__ needed
  def eval(self, p: dict[str, float], t: np.ndarray) -> np.ndarray | float:
    """Evaluate expression from global parameter dictionary using this expression's known parameter names."""
    args = [p[name] for name in self.parameters]
    #t_same = (t == self.t_cache).all()
    t_same = (self.t_cache is not None and len(t) == len(self.t_cache) and t[0] == self.t_cache[0] and t[-1] == self.t_cache[-1])
    if t_same and (args == self.p_cache):
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

  def _merge_skeleton(self, other: Self) -> Self:
    """Construct new Expression which merges the parameters of this Expression and another, but leaves evaluation logic undefined."""
    result = Expression(None)
    result.parameters = {**self.parameters, **other.parameters}
    result.bounds = {**self.bounds, **other.bounds}
    result.eval = None
    result._sub_exprs.extend([self, other])
    return result

  # ===============================================================================================

  def __add__(self, other: Self) -> Self:
    """Construct new Expression which returns the sum of this Expression and another."""
    result = self._merge_skeleton(other)
    result.eval = lambda p, t: self(p, t) + other(p, t)
    return result

  # ===============================================================================================
    
  def __mul__(self, other: Self) -> Self:
    """Construct new Expression which returns the product of this Expression and another."""
    result = self._merge_skeleton(other)
    result.eval = lambda p, t: self(p, t) * other(p, t)
    return result

  # ================================================================================================

  def clear_cache(self):
    """Deletes cached numerical arrays to save space (e.g. for pickling)."""
    self.t_cache = None
    self.p_cache = None
    self.val_cache = None
    for sub_expr in self._sub_exprs:
      sub_expr.clear_cache()

# Type alias for Expressions or arrays/numbers that can be coerced into Expressions.
ExpressionLike = (Expression | Callable[..., np.ndarray | np.number] | np.ndarray | np.number)

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

  def eval(self, p: Mapping[str, float], t: np.ndarray = None) -> float:
    # restrict parameter dictionary to only the declared dependent parameters
    try:
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
    const: ExpressionLike = None
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
    if scale is None and const is None and (terms is None or len(terms) == 0):
      raise ValueError("Must provide at least one of 'scale', 'terms', or 'const'.")
    
    self.data = data
    self.metadata = {}

    self.scale = scale if scale is not None else 1
    self.scale = util.ensure_type(self.scale, Expression)

    self.const = const if const is not None else 0
    self.const = util.ensure_type(self.const, Expression)

    self.terms = terms if terms is not None else {}
    self.terms = {name: util.ensure_type(term, Expression) for name, term in self.terms.items()}

    # internal copies of self.terms and self.const which may be modified to apply parameter constraints
    self._opt_terms = None
    self._opt_const = None

    # get a list of all nonlinear parameter names, excluding the linear coefficients
    nonlinear_parameters = [
      *self.scale.parameters,
      *self.const.parameters,
      *sum((list(term.parameters) for term in self.terms.values()), [])
    ]

    # raise an error if any linear coefficient names also appear in the nonlinear parameter system
    for coeff in self.terms:
      if coeff in nonlinear_parameters:
        raise ValueError(f"Linear coefficient '{coeff}' cannot also appear in nonlinear parameters.")

    # merge parameter guesses and bounds from all component expressions
    self.parameters = {**self.scale.parameters}
    self.parameters.update({k: v for k, v in self.const.parameters.items() if k not in self.parameters})
    self.bounds = {**self.scale.bounds, **self.const.bounds}
    for name, term in self.terms.items():
      self.parameters.update({**{k: v for k, v in term.parameters.items() if k not in self.parameters}, name: 0})
      self.bounds.update(term.bounds)
    
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

    self.minuit.print_level = 1

    # apply parameter bounds to minuit
    for name, value in self.bounds.items():
      self.minuit.limits[name] = value

    # fix linear parameters and those with constraints
    for name in self.terms:
      self.minuit.fixed[name] = True

    # internal tracking of fixed/floating parameters, since floating linear & constrained parameters are treated as 'fixed' by minuit
    self.fixed = {name: False for name in self.parameters}
    self._fit_linear = True

    # covariance matrix of fit parameters
    self.cov = None

    # condition number of linear fit matrix
    self.cond = None

  # ===============================================================================================
  
  def _fix(self, name: str):
    self.fixed[name] = True
    if (name not in self.terms) and (name not in self._nonlinear_constraints):
      self.minuit.fixed[name] = True

  def fix(self, *options: str | dict[str, float]):
    """
    Fix given parameter(s) at current value(s). Supports Unix-style wildcards, e.g. "x_*" will fix
    "x_1", "x_2", etc. Constrained parameter values will be frozen too.
    """
    if len(options) == 0:
      for parameter in self.parameters:
        self._fix(parameter)
    else:
      for option in options:
        if not isinstance(option, dict):
          option = {option: None}
        for name, value in option.items():
          if value is not None:
            self.minuit.values[name] = value
          for parameter in self.parameters:
            if fnmatch.fnmatch(parameter, name):
              self._fix(parameter)

  # ===============================================================================================
  
  def _free(self, name: str):
    self.fixed[name] = False
    if (name not in self.terms) and (name not in self._nonlinear_constraints):
      self.minuit.fixed[name] = False

  def free(self, *names: str):
    """
    Allow given parameter(s) to float, after previously being fixed. Supports Unix-style wildcards,
    e.g. "x_*" will free "x_1", "x_2", etc. For constrained parameters, constraints will resume
    after freeing. To remove constraints, use 'HybridFit.unconstrain'.
    """
    if len(names) == 0:
      for parameter in self.parameters:
        self._free(parameter)
    else:
      for name in names:
        for parameter in self.parameters:
          if fnmatch.fnmatch(parameter, name):
            self._free(parameter)

  # ===============================================================================================
        
  def constrain(self, constraints: dict[str, Constraint | dict[str, Constraint]]):
    """Constrain given parameters as functions of other unconstrained parameters."""

    for name, constraint in constraints.items():

      # Simple, singular constraints may be applied to any parameter, as long as the constraint
      # only involves nonlinear parameters.
      if (name not in self.terms) and isinstance(constraint, Constraint):

        if any([dependency in self.terms for dependency in constraint.parameters]):
          raise ValueError("Nonlinear constraints may only involve nonlinear parameters.")
        
        self._nonlinear_constraints[name] = constraint
        self.minuit.fixed[name] = True

      # Linear parameters may be constrained as linear combinations of other linear parameters,
      # supplied as dict[str, Constraint] which maps each linear parameter name in the combination
      # to a constrained coefficient (which may involve nonlinear parameters).
      # e.g. self.constrain({"a_k": {"a_0": c_0, "a_1": c_1, ...}}) applies the constraint
      # a_k = a_0 * c_0(n) + a_1 * c_1(n) + ..., where 'n' is the nonlinear parameter system.
      elif (name in self.terms) and isinstance(constraint, dict):
        
        for linear_param in constraint:

          coefficient = util.ensure_type(constraint[linear_param], Expression)
          constraint[linear_param] = coefficient

          if (linear_param not in self.terms) or (linear_param in self._linear_constraints):
            raise ValueError(f"Parameter '{linear_param}' is not a linear parameter, or is already constrained.")
          
          if any([dependency in self.terms for dependency in coefficient.parameters]):
            raise ValueError("Coefficient in constrained linear combination may only involve nonlinear parameters.")
          
        self._linear_constraints[name] = constraint

      else:

        raise ValueError(f"Unrecognized constraint for parameter '{name}'.")

  # ===============================================================================================
      
  def unconstrain(self, *names):
    """Remove any constraint relationships from given parameter names."""
    for name in names:
      if name in self._nonlinear_constraints:
        del self._nonlinear_constraints[name]
        if not self.fixed[name]:
          self.minuit.fixed[name] = False
      elif name in self._linear_constraints:
        del self._linear_constraints[name]

  # ===============================================================================================
        
  def is_constrained(self, name):
    """Checks if the given parameter is currently constrained."""
    return (name in self._nonlinear_constraints) or (name in self._linear_constraints)

  # ===============================================================================================

  def at_limit(self, name):
    """Checks if the given parameter is stuck at either its lower or upper limit."""
    value, error = self.minuit.values[name], self.minuit.errors[name]
    if self.minuit.limits[name] is not None:
      for limit in self.minuit.limits[name]:
        if abs(value - limit) <= 0.5 * error:
          return True

  # ===============================================================================================

  # TODO: reverse p, t order here
  def __call__(self, p: Mapping[str, float] = None, t: np.ndarray = None) -> np.ndarray:
    """
    Evaluate the model using the given dictionary of parameter values and independent variable.
    Defaults to current internal state of parameter system and data points if not supplied.
    """

    if p is None:
      p = self.minuit.values
    if t is None:
      t = self.data.x

    scale, const = self.scale(p, t), self.const(p, t)
    term_total = 0
    for name, term in self.terms.items():
      term_total += p[name] * term(p, t)

    return scale * (const + term_total)
  
  # ===============================================================================================

  def _fit_linear_combination(self, p: Mapping[str, float], t: np.ndarray, return_cond: bool = False) -> dict[str, float]:
    """
    Returns a dictionary of best-fit coefficients for the terms in the linear combination
    which are currently floating.
    """

    y = self.data.y if self.cost.mask is None else self.data.y[self.cost.mask]
    err = self.data.y_err if self.cost.mask is None else self.data.y_err[self.cost.mask]

    scale, const = self.scale(p, t), self._opt_const(p, t)
    result = util.fit_linear_combination(
      [term(p, t) for term in self._opt_terms.values()],
      y / scale - const, # must remove scale and const models from data to isolate linear combination
      err / scale, # dividing by scale model also scales data errorbars, but not subtracting const
      return_cond = return_cond
    )

    if return_cond:
      coeffs, cond = result
      self.cond = cond
    else:
      coeffs = result

    return dict(zip(self._opt_terms.keys(), coeffs))
  
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
    if self._fit_linear:
      coeffs = self._fit_linear_combination(p, t)
      p.update(coeffs)

    # update linear constrained parameters
    for name, constraint in self._linear_constraints.items():
      if not self.fixed[name]:
        p[name] = sum(
          p[linear_param] * coeff_constraint(self.minuit.values)
          for linear_param, coeff_constraint in constraint.items()
        )
    
    # evaluate model with updated parameter dictionary
    return self(p, t)
  
  # ===============================================================================================

  def hesse(self):
    # release floating linear parameters in minuit system, call HESSE, then re-fix them
    floating_coeffs = [name for name in self._opt_terms]
    if len(floating_coeffs) > 0:
      self.minuit.fixed[*floating_coeffs] = False
    self._fit_linear = False
    self.minuit.hesse()
    if len(floating_coeffs) > 0:
      self.minuit.fixed[*floating_coeffs] = True
    self._fit_linear = True

  # ===============================================================================================

  def fit(self, verbose = True, max_iterations = 3, hesse = True):
    """Runs chi-squared minimization and covariance estimation for floating parameters."""

    self._opt_terms = {name: value for name, value in self.terms.items()}
    self._opt_const = self.const

    # absorb fixed terms as part of the constant with no coefficient, and remove from system
    for name in self.terms:
      if self.fixed[name]:
        self._opt_const = self._opt_const + util.ensure_type(self.minuit.values[name], Expression) * self.terms[name]
        del self._opt_terms[name]

    # after fixed terms removed, modify system of terms according to any linear constraints
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
          if linear_param not in self._opt_terms:
            raise ValueError(f"Linear parameter '{linear_param}' must be floating unconstrained in order to constrain '{name}'.")
          # update term associated with each dependent parameter
          self._opt_terms[linear_param] = self._opt_terms[linear_param] + coeff_constraint * self.terms[name]
        # remove constrained parameter from the constrained linear system
        del self._opt_terms[name]

    iterations = 0
    success = False
    while not success and iterations < max_iterations:

      self.minuit.migrad()

      # update nonlinear constrained parameters in minuit results
      for name, constraint in self._nonlinear_constraints.items():
        if not self.fixed[name]:
          self.minuit.values[name] = constraint(self.minuit.values)

      # update linear parameters in minuit results
      coeffs = self._fit_linear_combination(
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
            self.minuit.values[a] * coeff(self.minuit.values)
            for a, coeff in constraint.items()
          )

      if hesse:
        if verbose:
          print("Running HESSE.")
        self.hesse()

      self.cov = self.minuit.covariance.to_dict()
      
      for parameter in self.minuit.parameters:
        for other in self.minuit.parameters:
          if (parameter, other) not in self.cov:
            self.cov[parameter, other] = self.cov[other, parameter]

      self.cov = {p1: {p2: self.cov[p1, p2] for p2 in self.minuit.parameters} for p1 in self.minuit.parameters}

      self.errors = self.minuit.errors.to_dict()

      for parameter in self.minuit.parameters:
        if self.fixed[parameter] or self.is_constrained(parameter):
          self.errors[parameter] = 0
          for other in self.minuit.parameters:
            self.cov[parameter][other] = 0
     
      # TODO: is it right to not count fixed parameters in NDF after some rounds of optimizing them?
      # TODO: sometimes chi2 goes down a little, but chi2/ndf goes up a little after freeing lots of parameters in last step
      self.ndf = self.cost.ndata - (self.minuit.nfit + len(self._opt_terms))
      self.chi2 = self.minuit.fval
      self.chi2_err = np.sqrt(2 * self.ndf)
      self.chi2ndf = self.chi2 / self.ndf
      self.chi2ndf_err = self.chi2_err / self.ndf
      self.pval = util.p_value(self.chi2, self.ndf)
      self.curve = self()

      if verbose:
        self.print()

      # re-evaluate fit quality after running HESSE
      if not (self.minuit.fmin.is_valid and self.minuit.fmin.has_accurate_covar) and not all(self.minuit.fixed):
        if verbose:
          print(f"Minuit is unhappy with fit validity after HESSE. Repeating minimization.")
        iterations += 1
      else:
        success = True

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
    if self.minuit.fmin.has_parameters_at_limit:
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
    
  def print(self):
    """Prints fit convergence status/warnings, table of parameter information, and chi-squared/p-value."""

    print()
    self.check_minimum(verbose = True)
    print(f"Time spent: {self.minuit.fmin.time:.6f} seconds.")

    headers = ["index", "name", "value", "error", "limit-", "limit+", "type", "status"]
    rows = []

    for i, name in enumerate(self.minuit.parameters):
      value = self.minuit.values[name]
      error = self.minuit.errors[name]
      error_order = util.order_of_magnitude(error)
      # TODO: sort out decimal/scientific notation appearance based on sig figs / desired order limits
      decimals = 4
      if error_order < 0:
        new_decimals = abs(error_order) + 2
        if new_decimals > decimals:
          decimals = new_decimals
      rows.append([
        i,
        name,
        f"{value:.{decimals}f}",
        # value,
        f"{error:.{decimals}f}" if not self.fixed[name] and not self.is_constrained(name) else "",
        # error if not self.fixed[name] and not self.is_constrained(name) else np.nan,
        f"{self.minuit.limits[name][0]:.4f}" if not np.isinf(self.minuit.limits[name][0]) else "",
        f"{self.minuit.limits[name][1]:.4f}" if not np.isinf(self.minuit.limits[name][1]) else "",
        "linear" if name in self.terms else "",
        "fixed" if self.fixed[name] else ("constr." if self.is_constrained(name) else "")
      ])

    print(tab.tabulate(rows, headers, tablefmt = "grid", numalign = "left"))
    print(f"cond = {(self.cond if self.cond is not None else 1):.1f}")
    print(f"chi2 = {self.chi2:.4f} +/- {self.chi2_err:.4f}")
    print(f"chi2/ndf = {self.chi2ndf:.4f} +/- {self.chi2ndf_err:.4f}")
    pval_format = ".4f" if util.order_of_magnitude(self.pval) > -4 else ".1e"
    print(f"p-value = {self.pval:{pval_format}}")
    print()

  # ===============================================================================================
    
  def statbox(self):
    """Returns list of math-formatted strings to display chi2/ndf and p-value on a plot."""
    return [
      rf"$\chi^2$/ndf = {self.chi2ndf:.4f} $\pm$ {self.chi2ndf_err:.4f}",
      rf"$p$ = {self.pval:.4f}"
    ]

  # ===============================================================================================

  def pulls(self):
    """Calculate and return the array of fit pulls [(y - f(x)) / y_err]."""
    return (self.data.y - self.curve) / self.data.y_err

  # ===============================================================================================

  def fft(self):
    """
    Calculate and return the FFT of the fit pulls, scaled as units of the fit's chi2.
    Based on Parseval's theorem: chi^2 = sum(pulls^2) = sum(|FFT|^2) / len(FFT), so the entries of
    |FFT|^2 / len(FFT) yield each FFT frequency bin's contribution to the chi^2.
    """
    pulls = self.pulls()
    # Only want frequencies up to the Nyquist frequency, so use np.fft.rfft.
    fft = np.abs(np.fft.rfft(pulls))**2 / len(pulls)
    # But Parseval's theorem includes all FFT bins, including those in the 2nd mirrored half.
    # So double the FFT power in the non-zero bins that would have been counted twice in the chi^2.
    fft[1:] *= 2
    frequencies = np.fft.rfftfreq(len(pulls), self.data.x[1] - self.data.x[0])
    return frequencies, fft

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
      f"{prefix}cov": self.cov
      #f"{prefix}cov_labels": ",".join(self.minuit.parameters),
      #f"{prefix}cov": np.array([self.cov[i, j] for j in self.minuit.parameters for i in self.minuit.parameters])
    }

  # ================================================================================================

  def clear_caches(self):
    """Clear numerical caches in Expression objects to save space when pickling."""
    self.scale.clear_cache()
    self.const.clear_cache()
    for term in self.terms.values():
      term.clear_cache()

  # ================================================================================================

  def save(self, filename: str):
    """Wrapper for util.save which clears HybridFit's numerical caches to save space."""
    self.clear_caches()
    util.save(self, filename)
