import numpy as np
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patch
import matplotlib.lines as mpl_line
import matplotlib.text as mpl_text
import matplotlib.axes as mpl_axes

from typing import Any, Callable

# TODO: add unit option to relpos, defaults to internal size/width, switch for absolute
# TODO: add gap variables to npad, lpad to distinguish node/layer sep

# ------------------------------------------------------------------------------------------------

_default_patch_opts = {
  "color": "white",
  "ec": "k",
  "lw": 0.5
}

_default_line_opts = {
  "color": "black",
  "alpha": 0.25,
  "lw": 1,
  "zorder": 0
}

_default_text_opts = {
  "color": "black",
  "ha": "center",
  "va": "center",
  "clip_on": False,
  "usetex": True
}

# ------------------------------------------------------------------------------------------------

# wrap matplotlib Patch constructors with common center-width argument structure: x, y, width, **kwargs
_create_patch = {
  "circle": lambda x, y, s, **kwargs: mpl_patch.Circle((x, y), s/2, **kwargs),
  "square": lambda x, y, s, **kwargs: mpl_patch.Rectangle((x-s/2, y-s/2), s, s, **kwargs)
}

_default_node_opts = {
  "shape": "circle",
  "size": 1
}

# ------------------------------------------------------------------------------------------------

"""Sets viewing range for given or current axes to include all artists."""
def autoscale(ax: mpl_axes.Axes = None):
  
  if ax is None:
    ax = plt.gca()

  ax.set_aspect("equal")
  ax.autoscale_view()
  ax.figure.draw_without_rendering()

  # update data limits to account for any text annotations
  artists = [artist for artist in ax.get_children()]
  for artist in artists:
    if not isinstance(artist, mpl_text.Text) or not artist.get_text():
      continue
    bbox = artist.get_window_extent().transformed(ax.transData.inverted())
    ax.update_datalim(bbox.corners())

  ax.autoscale_view()

# ------------------------------------------------------------------------------------------------

class Node:

  def __init__(
    self,
    xy: tuple[float, float],
    text: str | mpl_text.Text = "",
    shape: str = _default_node_opts["shape"],
    size: float = _default_node_opts["size"],
    layer: "Layer" = None,
    **kwargs
  ):

    if shape not in _create_patch:
      raise ValueError(f"Node shape must be one of {list(_create_patch)}.")

    if size < 0:
      raise ValueError(f"Node radius must be >= 0.")
    
    # reference to parent layer and network
    self.layer = layer
    self.network = layer.network

    # position and radius
    self.x = xy[0]
    self.y = xy[1]
    self.size = size

    # shortcuts to common anchor points
    self.center = self.relpos(0, 0)
    self.left = self.relpos(-1, 0)
    self.right = self.relpos(1, 0)
    self.bottom = self.relpos(0, -1)
    self.top = self.relpos(0, 1)

    # create shape patch object with default options, forwarding any extra keywords
    kwargs = {**_default_patch_opts, **kwargs}
    self.patch: mpl_patch.Patch = _create_patch[shape](self.x, self.y, self.size, **kwargs)

    # create label text object
    if isinstance(text, mpl_text.Text):
      self.text = text
    else:
      self.text = mpl_text.Text(self.x, self.y, text, **_default_text_opts)

    # additional text annotations other than the central label
    self.annotations: list[mpl_text.Text] = []

  """Returns 2D coordinate shifted from node center by (dx, dy) in units of the node radius."""
  def relpos(self, dx: float, dy: float, abs: bool = False) -> np.ndarray:
    if abs:
      return np.array([self.x + dx, self.y + dy])
    else:
      return np.array([self.x + dx * self.size/2, self.y + dy * self.size/2])
  
  """Add text annotation at the given relative position from node center."""
  def annotate(self, text: str, relpos: tuple[float, float], **kwargs):
    kwargs = {**_default_text_opts, **kwargs}
    self.annotations.append(mpl_text.Text(*self.relpos(*relpos), text, **kwargs))

  """Make a Connection from this node to another."""
  def connect(self, other: "Node", **kwargs):
    return Connection(self, other, **kwargs)

  """Add this node to the given or current axes."""
  def draw(self, ax: mpl_axes.Axes = None):
    if ax is None:
      ax = plt.gca()
    ax.add_patch(self.patch)
    ax.add_artist(self.text)
    for annotation in self.annotations:
      ax.add_artist(annotation)

  """Highlights all connections to this node."""
  def highlight(self, enable: bool = True):
    if self.network is None:
      raise ValueError("Highlighted node is missing reference to parent network.")
    for pair, connection in self.network.connections.items():
      if self in pair and enable:
        connection.highlight()
      else:
        connection.highlight(False)
    return self

  """Set the face color of the node."""
  def color(self, color: str):
    self.patch.set_facecolor(color)
    return self
  
# ------------------------------------------------------------------------------------------------
    
class Layer:

  def __init__(
    self,
    xy: tuple[float, float],
    nodes: int,
    symbol: str = "",
    size: float = _default_node_opts["size"],
    shape: str = _default_node_opts["shape"],
    nstep: float = None,
    label_format: Callable[[int], str] = None,
    network: "Network" = None,
    **kwargs
  ):
    
    if nodes < 0:
      raise ValueError(f"Number of nodes in a layer must be >= 0.")

    # reference to parent network
    self.network = network

    # centroid of layer
    self.x = xy[0]
    self.y = xy[1]

    # size of nodes and gap between them (as fraction of size)
    self.size = size
    self.shape = shape

    if nstep is None:
      nstep = 1.2 * self.size
    self.nstep = nstep

    self.height = (nodes - 1) * self.nstep
    y_start = self.y + self.height/2

    # label format function
    if label_format is None:
      if symbol:
        label_format = lambda node: fr"${symbol}_{{{node}}}$"
      else:
        label_format = lambda node: ""

    self.nodes: list[Node] = []
    for n in range(nodes):
      self.nodes.append(Node(
        (self.x, y_start - n * self.nstep),
        label_format(n),
        size = self.size,
        shape = self.shape,
        layer = self,
        **kwargs
      ))

    # vertical position of first and last nodes
    self.ymax = self.nodes[0].y
    self.ymin = self.nodes[-1].y

    self.annotations: list[mpl_text.Text] = []

  """Returns 2D coordinate shifted from layer center by (dx, dy) in units of the layer width/height."""
  def relpos(self, dx: float, dy: float, abs: bool = False) -> np.ndarray:
    if abs:
      return np.array([self.x + dx, self.y + dy])
    else:
      return np.array([self.x + dx * self.size/2, self.y + dy * self.height/2])

  """Add text annotation at the given relative position from node center."""
  def annotate(self, text: str, relpos: tuple[float, float], **kwargs):
    kwargs = {**_default_text_opts, **kwargs}
    self.annotations.append(mpl_text.Text(*self.relpos(*relpos), text, **kwargs))

  """Make a mapping from (NodeA, NodeB) -> Connection from each node in this layer to another layer."""
  def connect(self, other: "Layer", direct = False, **kwargs):
    result = {}
    node_mapping = zip if direct else itertools.product
    for this_node, other_node in node_mapping(self.nodes, other.nodes):
      result[this_node, other_node] = this_node.connect(other_node, **kwargs)
    return result

  """Add this layer of nodes to the given or current axes."""
  def draw(self, ax: mpl_axes.Axes = None):
    for node in self.nodes:
      node.draw(ax)
    for annotation in self.annotations:
      ax.add_artist(annotation)

# ------------------------------------------------------------------------------------------------
      
class Connection:

  def __init__(
    self,
    start: Node,
    end: Node,
    label: str = "",
    start_relpos: tuple[float, float] = None,
    end_relpos: tuple[float, float] = None,
    label_relpos: float = 0.5,
    line_kwargs = None,
    text_kwargs = None
  ):
    
    self.start_node = start
    self.end_node = end

    if start_relpos is None:
      start_relpos = (1, 0)
    if end_relpos is None:
      end_relpos = (-1, 0)

    self.start_pos = self.start_node.relpos(*start_relpos)
    self.end_pos = self.end_node.relpos(*end_relpos)
    self.vector = self.end_pos - self.start_pos

    x0, y0 = self.start_pos
    x1, y1 = self.end_pos

    if line_kwargs is None:
      line_kwargs = {}
    line_kwargs = {**_default_line_opts, **line_kwargs}
    self.line = mpl_line.Line2D([x0, x1], [y0, y1], **line_kwargs)

    if text_kwargs is None:
      text_kwargs = {}
    text_kwargs = {**_default_text_opts, "bbox": {"color": "white"}, **text_kwargs}
    self.text = mpl_text.Text(*self.relpos(label_relpos), label, **text_kwargs)

  """Returns 2D coordinate at given fraction along the line."""
  def relpos(self, frac: float = 0.5) -> np.ndarray:
    return self.start_pos + frac * self.vector
  
  """Add this connection between nodes to the given or current axes."""
  def draw(self, ax: mpl_axes.Axes = None):
    if ax is None:
      ax = plt.gca()
    ax.add_line(self.line)
    ax.add_artist(self.text)

  """Set highlighted state on or off."""
  def highlight(self, enable: bool = True):
    if enable:
      self.line.set_linewidth(1.5)
      self.line.set_alpha(1)
    else:
      self.line.set_linewidth(1)
      self.line.set_alpha(0.25)

# ------------------------------------------------------------------------------------------------

class Network:

  def __init__(
    self,
    xy: tuple[float, float] = None,
    lstep: float = 3,
    **kwargs
  ):
    
    # centroid of first layer
    if xy is None:
      xy = (0, 0)
    self.x = xy[0]
    self.y = xy[1]

    # gap between layers
    self.lstep = lstep
    if self.lstep < 0:
      raise ValueError(f"Network layer gap size must be >= 0.")
    
    # extra options passed to layers
    self.options = kwargs
    
    self.layers: list[Layer] = []
    self.connections: dict[tuple[Node, Node], Connection] = {}

    # minimum and maximum extent of all layers
    self.xmin, self.xmax = None, None
    self.ymin, self.ymax = None, None
  
  # ------------------------------------------------------------------------------------------------

  def add_layer(self, *args, lstep: float = None, connect: str = "full", **kwargs):

    prev_layer = self.layers[-1] if len(self.layers) > 0 else None

    if lstep is None:
      lstep = self.lstep

    if prev_layer is not None:
      new_x = prev_layer.x + lstep
    else:
      new_x = self.x

    kwargs = {**self.options, **kwargs}
    new_layer = Layer((new_x, self.y), *args, network = self, **kwargs)

    self.layers.append(new_layer)
    if connect and prev_layer is not None:
      self.connections.update(prev_layer.connect(new_layer, direct = (connect == "direct")))

    if self.xmin is None or new_x < self.xmin:
      self.xmin = new_x
    if self.xmax is None or new_x > self.xmax:
      self.xmax = new_x
    if self.ymin is None or new_layer.ymin < self.ymin:
      self.ymin = new_layer.ymin
    if self.ymax is None or new_layer.ymax > self.ymax:
      self.ymax = new_layer.ymax

    return self

  # ------------------------------------------------------------------------------------------------

  def draw(self, ax: mpl_axes.Axes = None):

    if ax is None:
      ax = plt.gca()

    for layer in self.layers:
      layer.draw(ax)
    for connection in self.connections.values():
      connection.draw(ax)

    autoscale(ax)
    ax.set_axis_off()

# ------------------------------------------------------------------------------------------------

# TODO: Attention could be subclass of Network? would require some updates/generalization of Network, e.g. draw bounding patch, relpos for labels, etc.
# TODO: tell each network an input vector label, so attention can fetch it
# TODO: make each thing subclass a common Region(?) class, with indiviudal (x,y), width/height, and relpos method, can update as network grows
class Attention:

  def __init__(
    self,
    multinetwork: "MultiNetwork" = None,
    width: float = 3,
    pad: float = None
  ):

    self.multinetwork = multinetwork
    self.width = width

    lower_left_node = self.multinetwork.networks[-1].layers[-1].nodes[-1]
    upper_left_node = self.multinetwork.networks[0].layers[-1].nodes[0]

    self.xmin, self.ymin = lower_left_node.relpos(1, -1)
    _, self.ymax = upper_left_node.relpos(1, 1)
    self.xmax = self.xmin + self.width

    self.x = (self.xmin + self.xmax) / 2
    self.y = (self.ymin + self.ymax) / 2

    if pad is None:
      pad = upper_left_node.size
    self.ymin -= pad
    self.ymax += pad
    self.height = self.ymax - self.ymin

    patch_args = {**_default_patch_opts, "color": "lightgray"}
    self.patch = mpl_patch.Rectangle(
      (self.xmin, self.ymin),
      self.width,
      self.height,
      zorder = 0,
      **patch_args
    )

    self.inputs = Layer(
      (self.xmin + pad, (self.ymin + self.ymax)/2),
      len(self.multinetwork.networks),
      nstep = self.multinetwork.netstep,
      shape = "square",
      label_format = lambda n: fr"$\mathbf{{x}}_{{{n}}}$",
      alpha = 0,
      lw = 0
    )

    self.outputs = Layer(
      (self.xmax - pad, (self.ymin + self.ymax)/2),
      len(self.multinetwork.networks),
      nstep = self.multinetwork.netstep,
      shape = "square",
      label_format = lambda n: fr"$\mathbf{{x}}_{{{n}}}$",
      alpha = 0,
      lw = 0
    )

    self.connections = self.inputs.connect(self.outputs)

    # self.inputs: list[Node] = []
    # self.outputs = list[Node] = []
    # for network in self.multinetwork.networks:

  def draw(self, ax: mpl_axes.Axes = None):
    if ax is None:
      ax = plt.gca()
    ax.add_patch(self.patch)
    self.inputs.draw(ax)
    self.outputs.draw(ax)
    for connection in self.connections.values():
      connection.draw(ax)

  """Returns 2D coordinate shifted from attention center by (dx, dy) in units of the attention width/height."""
  def relpos(self, dx: float, dy: float, abs: bool = False) -> np.ndarray:
    if abs:
      return np.array([self.x + dx, self.y + dy])
    else:
      return np.array([self.x + dx * self.width/2, self.y + dy * self.height/2])

# ------------------------------------------------------------------------------------------------

class MultiNetwork:

  def __init__(
    self,
    networks: int,
    netstep: float = 3,
    **kwargs
  ):
    self.netstep = netstep
    self.networks = [Network((0, -i*netstep), **kwargs) for i in range(networks)]
    self.attentions: list[Attention] = []

  def add_layer(self, *args, **kwargs):
    for i, network in enumerate(self.networks):
      network.add_layer(*args, **kwargs)

  def add_attention(self):

    input_len = len(self.networks[0].layers[0].nodes)
    input_shape = self.networks[0].layers[0].shape
    input_size = self.networks[0].layers[0].size

    full_connect = (len(self.networks[0].layers) > 1)
    self.add_layer(input_len, shape = input_shape, size = input_size, connect = True if full_connect else "direct")
    attention = Attention(self)
    self.attentions.append(attention)
    self.add_layer(input_len, lstep = attention.width + input_size, connect = False, shape = input_shape, size = input_size)

  def draw(self, ax: mpl_axes.Axes = None):
    if ax is None:
      ax = plt.gca()
    for network in self.networks:
      network.draw(ax)
    for attention in self.attentions:
      attention.draw(ax)

# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":

  fig = plt.figure()
  ax = fig.add_subplot()

  # network = Network((0, 0), gap = 5)
  # network.add_layer(4, "x", shape = "square")
  # network.add_layer(10, "h")
  # network.add_layer(2, "y", shape = "square")
  # network.layers[0].annotate(r"$\mathbf{x}_0$", (-3, 0))
  # network.layers[-1].annotate(r"$\mathbf{y}_0$", (+3, 0))
  # network.layers[1].nodes[3].color("C0").highlight()
  # network.draw(ax)

  # network2 = Network((0, -10), gap = 5, size = 0.6)
  # network2.add_layer(3, shape = "square")
  # network2.add_layer(5)
  # network2.add_layer(2, shape = "square")
  # network2.layers[0].annotate(r"$\mathbf{x}_1$", (-5, 0))
  # network2.layers[-1].annotate(r"$\mathbf{y}_1$", (+5, 0))
  # network2.layers[1].nodes[1].color("C1").highlight()
  # network2.draw(ax)

  network = MultiNetwork(4, netstep = 4, size = 0.5, nstep = 0.5)
  network.add_layer(3, shape = "square")
  network.add_attention()
  network.add_layer(5)
  network.add_attention()
  network.add_layer(5)
  network.add_attention()
  network.add_layer(5)
  network.add_layer(2, shape = "square")
  network.draw(ax)

  plt.show()