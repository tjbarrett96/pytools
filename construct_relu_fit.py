import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import pytools.style as style
from draw_network import Network

depth = True

nodes = 4 if depth else 10

x = np.linspace(-3, 2, 40)
f = x**3 + 2*x**2 - 2*x + 10

noise = np.random.normal(0, 0.15, len(x))
f += noise

x_fine = np.linspace(-7, 7, 10_000)

# ------------------------------------------------------------------------------------------------

def relu(x):
  return np.maximum(x, 0)

def hidden(a, b, x = x, sigma = relu):
  return sigma(a*(x - b))

# ------------------------------------------------------------------------------------------------

# partition domain into one bin for each node
x_edges = np.linspace(x[0], x[-1], nodes + 1)
x_centers = (x_edges[1:] + x_edges[:-1]) / 2
x_width = x_edges[1] - x_edges[0]

# accumulate basis functions one at a time
basis_data = []
basis_fine = []

# start with constant output shift parameter to set f[0] -> 0 for ReLUs
basis_data.append(np.ones(len(x)) * f[0])
basis_fine.append(np.ones(len(x_fine)) * f[0])

# cumulative model result
result_fine = basis_fine[0]
result_solid_line = None
result_faded_line = None

# ------------------------------------------------------------------------------------------------

with style.make_pdf("demo_construct_relu_fit" + ("_depth" if depth else "") + ".pdf") as pdf:

  fig = style.stretched_figure(1.25)
  gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace = 0.05)
  ax = fig.add_subplot(gs[0, 0])
  net = fig.add_subplot(gs[0, 1])
  net.set_axis_off()
  net.set_aspect("equal")
  plt.subplot(1, 2, 1)

  # ------------------------------------------------------------------------------------------------

  x_range = x[-1] - x[0]
  x_pad = 0.15 * x_range
  plt.xlim(x[0] - x_pad, x[-1] + x_pad)

  y_min = min(0, np.min(f))
  y_max = max(0, np.max(f))
  y_abs_max = max(abs(y_min), abs(y_max))
  y_range = 2 * y_abs_max
  y_pad = 0.15 * y_range
  plt.ylim(-y_abs_max - y_pad, y_abs_max + y_pad)

  style.xlabel("x")
  style.ylabel("y")

  # ------------------------------------------------------------------------------------------------

  style.draw_horizontal(lw=1)
  plt.scatter(x, f, c = "darkgray")
  pdf.savefig()

  # ------------------------------------------------------------------------------------------------

  plt.subplot(1, 2, 2)

  network = Network()
  if not depth:
    network.add_layer(nodes)
    network.add_layer(1, fc = "k")
    network.add_layer(0)
    network.label_node('"basis"\nfunctions', 0, 0, dy = 1.25)
    network.label_node("output", 1, 0, dy = 1.25)
  else:
    network.add_layer(10)
    network.label_node('construct\n"basis"\nfunctions', 0, 0, dy = 1.25)
    network.add_layer(nodes)
    network.add_layer(1, fc = "k")
    network.label_node('final\n"basis"\nfunctions', 1, 0, dy = 1.25)
    network.label_node("output", 2, 0, dy = 1.25)
  
  network.draw()

  plt.subplot(1, 2, 1)
  for xi in x_edges:
    style.draw_vertical(xi, lw=1)
    
  pdf.savefig()

  # ------------------------------------------------------------------------------------------------

  select_fine_to_bin = (x_fine <= x[0])
  result_faded_line = plt.plot(x_fine, result_fine, alpha = 0.2, c = "k")[0]
  result_solid_line = plt.plot(x_fine[select_fine_to_bin], result_fine[select_fine_to_bin], c = "k")[0]
  pdf.savefig()

  # ------------------------------------------------------------------------------------------------

  for i in range(nodes):

    # x-values of bin edges
    x0, x1 = x_edges[i:i+2]

    # find data within this bin
    select_data_in_bin = (x >= x0) & (x <= x1)
    select_fine_in_bin = (x_fine >= x0) & (x_fine <= x1)
    
    x_bin = x[select_data_in_bin]
    y_bin = f[select_data_in_bin]

    # compute cumulative model already within this bin
    model_bin = 0
    for basis_fn in basis_data:
      model_bin += basis_fn[select_data_in_bin]

    if not depth:

      # # use np.polyfit for best line through residuals
      # a, y_int = np.polyfit(x_bin, y_bin - model_bin, 1)

      # # convert y-intercept to x-intercept (translation parameter b)
      # b = -y_int / a

      # # check if we need to flip sign using output weights C
      # if a < 0:
      #   a *= -1
      #   c = -1
      # else:
      #   c = +1

      # h_data = c * hidden(a, b)
      # h_fine = c * hidden(a, b, x_fine)

      def model(x, s, a):
        return s * relu(a*(x - x0))
      
      p_opt, p_cov = opt.curve_fit(
        model,
        x_bin,
        y_bin - model_bin,
        bounds = (
          [-1, -np.inf],
          [1, np.inf]
        )
      )

      h_data = model(x, *p_opt)
      h_fine = model(x_fine, *p_opt)

    else:

      # # depth case

      # # use np.polyfit for best line through residuals
      # p0 = np.polyfit(x_bin, y_bin - model_bin, 2)

      # roots = np.roots(p0)
      # print(roots)
      # # taper_data = np.ones(x)
      # # taper_fine = np.ones(x_fine)
      # # if len(roots) > 1:
      # #   for root in roots[1:]:
      # #     root_bin = (root - x[0]) // x_width
          
      # y0 = np.polyval(p0, x)[select_data_in_bin]
      # dy0 = y0[-1] - y0[0]

      # # check if we need to flip sign using output weights C
      # if dy0 < 0:
      #   p0 *= -1
      #   c = -1
      # else:
      #   c = +1

      # h_data = c * hidden(1, 0, x = np.polyval(p0, x))
      # h_fine = c * hidden(1, 0, x = np.polyval(p0, x_fine))
      # # h_data = np.polyval(p0, x)
      # # h_fine = np.polyval(p0, x_fine)

      # h_data[x < roots[0]] = 0
      # h_fine[x_fine < roots[0]] = 0

      def model(x, s, a, b):
        temp = s * relu(a*(x-x0)**2 + b*(x-x0))
        # temp = s * relu(a * (x - x0) * (x - b))
        temp[x < x0] = 0
        return temp
      
      p_opt, p_cov = opt.curve_fit(
        model,
        x_bin,
        y_bin - model_bin,
        bounds = (
          [-1, -np.inf, -np.inf],
          [1, np.inf, np.inf]
        )
      )

      h_data = model(x, *p_opt)
      h_fine = model(x_fine, *p_opt)

    plt.plot(x_fine, h_fine, alpha=0.2, c = f"C{i}")
    plt.plot(x_fine[select_fine_in_bin], h_fine[select_fine_in_bin], c = f"C{i}")
    # network.layers[1 if depth else 0][i].set_facecolor(f"C{i}")
    network.color_node(1 if depth else 0, i, f"C{i}")

    # compute ReLU that models this bin
    basis_data.append(h_data)
    basis_fine.append(h_fine)

    result_fine += h_fine

    if result_faded_line is not None:
      result_faded_line.remove()
    if result_solid_line is not None:
      result_solid_line.remove()

    select_fine_to_bin = (x_fine <= x1)
    result_faded_line = plt.plot(x_fine, result_fine, alpha = 0.2, c = "k")[0]
    result_solid_line = plt.plot(x_fine[select_fine_to_bin], result_fine[select_fine_to_bin], c = "k")[0]
    pdf.savefig()

  plt.show()