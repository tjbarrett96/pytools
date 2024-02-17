import numpy as np
from typing import Mapping, Callable
from dataclasses import dataclass

import iminuit
from iminuit.cost import LeastSquares

from pytools.expression import Expression, ExpressionLike
import pytools.util as util

import tabulate as tab
import json

# =================================================================================================

@dataclass
class Data:
  """Simple container for x-values, y-values, and optional errorbars."""
  x: np.ndarray
  y: np.ndarray
  y_err: np.ndarray = None
  x_err: np.ndarray = None

# ==================================================================================================

# Type hint for a function which takes a dictionary of parameter names/values, and returns a value
# in terms of those parameters. Used to constrain one parameter in terms of others.
ConstraintFunction = Callable[[Mapping[str, float]], float]

# TODO: maybe switch order of Expression parameters to (p, t = None) so that t optional for constraint-type Expressions
class Constraint(Expression):
  """Simple container which wraps a constraint function alongside a list declaring the dependent parameters."""

  def __init__(self, parameters: list[str], constraint: ConstraintFunction):
    super().__init__(None)
    self.parameters = {parameter: None for parameter in parameters}
    self.constraint = constraint

  def eval(self, p: Mapping[str, float]) -> float:
    # restrict parameter dictionary to only the declared dependent parameters
    return self.constraint({key: p[key] for key in self.parameters})
  
  def __call__(self, t: np.ndarray, p: Mapping[str, float]) -> float:
    return self.eval(p)

# =================================================================================================

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
    
    self.scale = scale if scale is not None else 1
    self.scale = HybridFit._ensure_expression(self.scale)

    self.const = const if const is not None else 0
    self.const = HybridFit._ensure_expression(self.const)

    self.terms = terms if terms is not None else {}
    self.terms = {name: HybridFit._ensure_expression(term) for name, term in self.terms.items()}

    # internal copies of self.terms and self.const which may be modified to apply parameter constraints
    self._constrained_terms = None
    self._constrained_const = None

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
    self.parameters = {**self.scale.parameters, **self.const.parameters}
    self.bounds = {**self.scale.bounds, **self.const.bounds}
    for name, term in self.terms.items():
      self.parameters.update({name: 0, **term.parameters})
      self.bounds.update(term.bounds)
    
    # mapping of nonlinear parameter(s) to Constraint(s), which only allows other nonlinear parameters
    self._nonlinear_constraints = {}

    # mapping of linear parameters to Constraint(s), which only allows linear combinations of unconstrained linear parameters
    self._linear_constraints = {}

    self.minuit = iminuit.Minuit(
      LeastSquares(data.x, data.y, data.y_err, self._opt_call),
      list(self.parameters.values()),
      name = list(self.parameters.keys())
    )

    # self.minuit.print_level = 2

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

  # ===============================================================================================
  
  def fix(self, *names: str):
    """
    Fix given parameter(s) at current value(s). Constrained parameter values will be frozen and constraints
    no longer applied while fixed.
    """
    for name in names:
      self.fixed[name] = True
      if (name not in self.terms) and (name not in self._nonlinear_constraints):
        self.minuit.fixed[name] = True

  # ===============================================================================================
  
  def free(self, *names: str):
    """
    Allow given parameter(s) to float, after previously being fixed. Constrained parameters will still
    have their constraints applied; to remove constraints, use 'HybridFit.remove_constraint'.
    """
    for name in names:
      self.fixed[name] = False
      if (name not in self.terms) and (name not in self._nonlinear_constraints):
        self.minuit.fixed[name] = False

  # ===============================================================================================
        
  def add_constraint(self, constraints: dict[str, Constraint | dict[str, Constraint]]):
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
      # e.g. self.add_constraint({"a_k": {"a_0": c_0, "a_1": c_1, ...}}) applies the constraint
      # a_k = a_0 * c_0(n) + a_1 * c_1(n) + ..., where 'n' is the nonlinear parameter system.
      elif (name in self.terms) and isinstance(constraint, dict):
        
        for linear_param, coefficient in constraint.items():

          if (linear_param not in self.terms) or (linear_param in self._linear_constraints):
            raise ValueError(f"Parameter '{linear_param}' is not a linear parameter, or is already constrained.")
          
          if any([dependency in self.terms for dependency in coefficient.parameters]):
            raise ValueError("Coefficient in constrained linear combination may only involve nonlinear parameters.")
          
          self._linear_constraints[name] = constraint

      else:

        raise ValueError(f"Unrecognized constraint for parameter '{name}'.")

  # ===============================================================================================
      
  def remove_constraint(self, *names):
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

  @staticmethod
  def _ensure_expression(expr):
    """Wraps the given expression in an Expression object if not already one."""
    if isinstance(expr, Expression):
      return expr
    else:
      return Expression(expr)

  # ===============================================================================================

  def __call__(self, t: np.ndarray, p: Mapping[str, float]) -> np.ndarray:
    """Evaluate the model using the given independent variable and dictionary of parameter values."""
    return self.scale(t, p) * sum(
      (p[name] * term(t, p) for name, term in self.terms.items()),
      self.const(t, p)
    )
  
  # ===============================================================================================

  def fit_linear_combination(self, t: np.ndarray, p: Mapping[str, float]) -> dict[str, float]:
    """
    Returns a dictionary of best-fit coefficients for the terms in the linear combination
    which are currently floating.
    """

    scale, const = self.scale(t, p), self._constrained_const(t, p)
    coeffs = util.fit_linear_combination(
      [term(t, p) for term in self._constrained_terms.values()],
      self.data.y / scale - const, # must remove scale and const models from data to isolate linear combination
      self.data.y_err / scale # dividing by scale model also scales data errorbars, but not subtracting const
    )

    return dict(zip(self._constrained_terms.keys(), coeffs))
  
  # ===============================================================================================

  def _opt_call(self, t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Internal wrapping of model evaluation which optimizes linear parameters and applies constraints."""
    
    # wrap parameter array into dictionary with parameter names
    p = dict(zip(self.parameters, p))
    
    # update nonlinear constrained parameters
    for name, constraint in self._nonlinear_constraints.items():
      if not self.fixed[name]:
        p[name] = constraint.eval(p)

    # optimize linear coefficient parameters
    if self._fit_linear:
      coeffs = self.fit_linear_combination(t, p)
      p.update(coeffs)

    # update linear constrained parameters
    for name, constraint in self._linear_constraints.items():
      if not self.fixed[name]:
        p[name] = sum(
          p[linear_param] * coeff_constraint.eval(self.minuit.values)
          for linear_param, coeff_constraint in constraint.items()
        )
    
    # evaluate model with updated parameter dictionary
    return self(t, p)
  
  # ===============================================================================================

  def fit(self):
    """Runs chi-squared minimization and covariance estimation for floating parameters."""

    self._constrained_terms = {name: value for name, value in self.terms.items()}
    self._constrained_const = self.const

    # absorb fixed terms as part of the constant with no coefficient, and remove from system
    for name in self.terms:
      if self.fixed[name]:
        self._constrained_const = self._constrained_const + self.minuit.values[name] * self.terms[name]
        del self._constrained_terms[name]

    # after fixed terms removed, modify system of terms according to any linear constraints
    # linear combination in fit model has the form:
    #   a_0 * f_0 + a_1 * f_1 + ... + a_k * f_k + ...
    # if we replace a_k => (a_0 * c_0 + a_1 * c_1 + ...), then linear combination becomes:
    #   a_0 * (f_0 + c_0 * f_k) + a_1 * (f_1 + c_1 * f_k) + ...
    # so each remaining term picks up scaled f_k term from linearly-constrained a_k
    for name in self.terms:
      # if constrained parameter is still in the system (i.e. wasn't fixed)...
      if (name in self._linear_constraints) and (name in self._constrained_terms):
        for linear_param, coeff_constraint in self._linear_constraints[name].items():
          # ensure each dependent parameter is still in the constrained system
          if linear_param not in self._constrained_terms:
            raise ValueError(f"Linear parameter '{linear_param}' must be floating unconstrained in order to constrain '{name}'.")
          # update term associated with each dependent parameter
          self._constrained_terms[linear_param] = self._constrained_terms[linear_param] + coeff_constraint * self.terms[name]
        # remove constrained parameter from the constrained linear system
        del self._constrained_terms[name]

    iterations = 0
    success = False
    while not success and iterations < 2:

      self.minuit.migrad()

      # update linear parameters in minuit results
      coeffs = self.fit_linear_combination(self.data.x, self.minuit.values)
      if len(coeffs) > 0:
        self.minuit.values[*coeffs.keys()] = coeffs.values()

      # update nonlinear constrained parameters in minuit results
      for name, constraint in self._nonlinear_constraints.items():
        if not self.fixed[name]:
          self.minuit.values[name] = constraint.eval(self.minuit.values)

      # update linear constrained parameters in minuit results
      for name, constraint in self._linear_constraints.items():
        if not self.fixed[name]:
          self.minuit.values[name] = sum(
            self.minuit.values[a] * coeff.eval(self.minuit.values)
            for a, coeff in constraint.items()
          )

      # release floating linear parameters in minuit system, call HESSE, then re-fix them
      floating_coeffs = [name for name in self._constrained_terms]
      self.minuit.fixed[*floating_coeffs] = False
      self._fit_linear = False
      self.minuit.hesse()
      self.minuit.fixed[*floating_coeffs] = True
      self._fit_linear = True

      # re-evaluate EDM after calling HESSE
      if self.minuit.fmin.edm > self.minuit.fmin.edm_goal:
        print(f"After HESSE, updated EDM exceeds target for convergence. Repeating minimization.")
        iterations += 1
      else:
        success = True

    self.cov = np.array(self.minuit.covariance)
    self.errors = self.minuit.errors.to_dict()

    self.ndf = len(self.data.y) - self.minuit.nfit - len(floating_coeffs)
    self.chi2 = self.minuit.fval
    self.chi2_err = np.sqrt(2 * self.ndf)
    self.chi2ndf = self.chi2 / self.ndf
    self.chi2ndf_err = self.chi2_err / self.ndf
    self.pval = util.p_value(self.chi2, self.ndf)

    self.print()

  # ================================================================================================
    
  def print(self):
    """Prints fit convergence status/warnings, table of parameter information, and chi-squared/p-value."""

    print()
    success = True
    if not self.minuit.fmin.is_valid:
      print("MINIMIZATION DID NOT CONVERGE!")
      success = False
      if self.minuit.fmin.edm > self.minuit.fmin.edm_goal:
        print(f"EDM (chi2 - chi2_min) ~ {self.minuit.fmin.edm:.4f} exceeds EDM goal of {self.minuit.fmin.edm_goal:.4f}.")
    if not self.minuit.fmin.has_accurate_covar:
      print("Covariance may not be accurate.")
      success = False
    if self.minuit.fmin.has_parameters_at_limit:
      print("Parameter(s) stuck at limits.")
      success = False
    if self.minuit.fmin.has_reached_call_limit:
      print("Function call limit exceeded.")
      success = False
    if success:
      print("Minimization was successful.")
    print(f"Time spent: {self.minuit.fmin.time:.6f} seconds.")

    headers = ["index", "name", "value", "error", "limit-", "limit+", "type", "status"]
    rows = [
      [
        i,
        name,
        f"{self.minuit.values[name]:.4f}",
        f"{self.minuit.errors[name]:.4f}" if not self.fixed[name] and not self.is_constrained(name) else "N/A",
        f"{self.minuit.limits[name][0]:.4f}" if not np.isinf(self.minuit.limits[name][0]) else "",
        f"{self.minuit.limits[name][1]:.4f}" if not np.isinf(self.minuit.limits[name][1]) else "",
        "constr." if self.is_constrained(name) else ("linear" if name in self.terms else ""),
        "fixed" if self.fixed[name] else ""
      ]
      for i, name in enumerate(self.minuit.parameters)
    ]

    print(tab.tabulate(rows, headers, tablefmt = "grid", numalign = "left"))
    print(f"chi2 = {self.chi2:.4f} +/- {self.chi2_err:.4f}")
    print(f"chi2/ndf = {self.chi2ndf:.4f} +/- {self.chi2ndf_err:.4f}")
    print(f"p-value = {self.pval:.4f}")
    print()

  # ===============================================================================================
    
  def results(self):
    return {
      "chi2": self.chi2,
      "chi2_err": self.chi2_err,
      "chi2ndf": self.chi2ndf,
      "chi2ndf_err": self.chi2ndf_err,
      "pval": self.pval,
      **self.minuit.values.to_dict(),
      **{f"{name}_err": error for name, error in self.minuit.errors.to_dict().items()},
      **{f"{name}_err": 0 for name in self.minuit.parameters if self.fixed[name] or self.is_constrained(name)}
    }