import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pytools.style as style

nodes = 10
eps = 0.1

x = np.linspace(-5, 5, 25)
f = x**3 + 2*x**2 - 5*x + 2
# f = np.cos(x)

relu = lambda x: np.maximum(x, 0)
def hidden(p, x = x):
  a = p[0::3]
  b = p[1::3]
  c = p[2::3]
  return (c * relu((a * np.subtract.outer(x, b)).T).T).T

def model(p,x=x):
  h = hidden(p,x)
  y = np.sum(h, axis = 0)
  return y

# def loss(p):
#   return np.max(np.abs(model(p,x) - f))

def loss(p,x=x):
  return np.sum((model(p,x) - f)**2)

p0 = np.zeros(3 * nodes)
a0 = ((np.random.random(nodes) - 0.5)*2) * 5
b0 = ((np.random.random(nodes) - 0.5)*2) * 4
# b0 = np.linspace(-4.5, 4.5, nodes)
c0 = ((np.random.random(nodes) - 0.5)*2)
# c0 = np.ones(nodes)

p0[0::3] = a0
p0[1::3] = b0
p0[2::3] = c0

bounds = []
for i in range(nodes):
  bounds.append((None, None))
  # bounds.append((None, None))
  bounds.append((x[0], x[-1]))
  bounds.append((-1, 1))
result = opt.minimize(loss, p0, bounds=bounds, options = {"disp": True})

p = result.x
a = p[0::3]
b = p[1::3]
c = p[2::3]

extended_xlim = np.max(np.abs(b)) + 2
extended_x = np.linspace(-extended_xlim, +extended_xlim, 500)
plot_xlim = (-extended_xlim, +extended_xlim)

f_range = np.max(f) - np.min(f)
extended_ylim = np.max(np.abs(f)) + 0.05*f_range
plot_ylim = (-extended_ylim, +extended_ylim)

node_order = np.argsort(b)
h = hidden(p, extended_x)[node_order, :]
h0 = hidden(p0, extended_x)[node_order, :]

interp_x = np.linspace(x[0], x[-1], 500)
y = model(p, interp_x)

with style.make_pdf("demo_randomized_relu_fit.pdf") as pdf:

  plt.scatter(x, f, c = "k", label = r"Target $f(x) = x^2$")
  style.draw_horizontal()
  style.xlabel("x")
  style.ylabel("y")
  style.make_unique_legend(loc = "upper center")
  plt.xlim(*plot_xlim)
  plt.ylim(*plot_ylim)
  pdf.savefig()

  # for offset in b:
  #   plt.axvline(offset, 0, 1, ls = "--", lw = 1, c = "k")

  plt.plot(interp_x, y, label = r"Model Output")
  style.make_unique_legend(loc = "upper center")
  pdf.savefig()
  plt.clf()

  for offset in b0:
    plt.axvline(offset, 0, 1, ls = "--", lw = 1, c = "k")

  for i, node in enumerate(h0):
    # node = np.where(node > 0, node, np.nan)
    # node -= i*0.1
    plt.plot(extended_x, node, c = f"C{i}", lw = 1)

  style.draw_horizontal()
  plt.xlim(*plot_xlim)
  plt.ylim(*plot_ylim)
  style.xlabel("x")
  style.ylabel("y")
  pdf.savefig()
  plt.clf()

  for offset in b:
    plt.axvline(offset, 0, 1, ls = "--", lw = 1, c = "k")

  for i, node in enumerate(h):
    # node = np.where(node > 0, node, np.nan)
    # node -= i*0.1
    plt.plot(extended_x, node, c = f"C{i}", lw = 1)

  style.draw_horizontal()
  plt.xlim(*plot_xlim)
  plt.ylim(*plot_ylim)
  style.xlabel("x")
  style.ylabel("y")
  pdf.savefig()