import numpy as np
import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patch
import matplotlib.lines as mpl_line
import matplotlib.text as mpl_text
import matplotlib.axes as mpl_axes

from typing import Any, Callable

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
  def relpos(self, dx: float, dy: float) -> np.ndarray:
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
    x: float,
    nodes: int,
    symbol: str = "",
    size: float = 1,
    gap: float = 0.2,
    label_format: Callable[[int, int], str] = None,
    network: "Network" = None,
    **kwargs
  ):
    
    if nodes < 0:
      raise ValueError(f"Number of nodes in a layer must be >= 0.")

    # reference to parent network
    self.network = network

    # position and spacing
    self.x = x
    self.y = 0
    self.gap = gap
    self.size = size

    dy = self.size + self.gap * self.size
    self.height = nodes * self.size + (nodes - 1) * (self.gap * self.size)
    y_start = self.height / 2 - self.size/2

    # label format function
    if label_format is None:
      if symbol:
        label_format = lambda node: fr"${symbol}_{{{node}}}$"
      else:
        label_format = lambda node: ""

    self.nodes: list[Node] = []
    for n in range(nodes):
      self.nodes.append(Node(
        (self.x, y_start - n * dy),
        label_format(n),
        size = self.size,
        layer = self,
        **kwargs
      ))

    self.annotations: list[mpl_text.Text] = []

  """Returns 2D coordinate shifted from layer center by (dx, dy) in units of the layer width/height."""
  def relpos(self, dx: float, dy: float) -> np.ndarray:
    return np.array([self.x + dx * self.size/2, self.y + dy * self.height/2])

  """Add text annotation at the given relative position from node center."""
  def annotate(self, text: str, relpos: tuple[float, float], **kwargs):
    kwargs = {**_default_text_opts, **kwargs}
    self.annotations.append(mpl_text.Text(*self.relpos(*relpos), text, **kwargs))

  """Make a mapping from (NodeA, NodeB) -> Connection from each node in this layer to another layer."""
  def connect(self, other: "Layer", **kwargs):
    result = {}
    for this_node in self.nodes:
      for other_node in other.nodes:
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
    gap: float = 3
  ):

    self.layers: list[Layer] = []
    self.connections: dict[tuple[Node, Node], Connection] = {}

    self.gap = gap
    if self.gap < 0:
      raise ValueError(f"Network layer gap size must be >= 0.")
  
  # ------------------------------------------------------------------------------------------------

  def add_layer(self, *args, connect = True, **kwargs):
    
    x = len(self.layers) * self.gap

    prev_layer = self.layers[-1] if len(self.layers) > 0 else None
    new_layer = Layer(x, *args, **{**kwargs, "network": self})

    self.layers.append(new_layer)
    if connect and prev_layer is not None:
      self.connections.update(prev_layer.connect(new_layer))

    return new_layer

  # ------------------------------------------------------------------------------------------------

  def draw(self, ax: mpl_axes.Axes = None):

    if ax is None:
      ax = plt.gca()

    for layer in self.layers:
      layer.draw(ax)
    for connection in self.connections.values():
      connection.draw(ax)

    # to get accurate artist bboxes below, must set aspect ratio & autoscale, then draw.
    # without autoscaling here, the first two text artists encountered in the iterable
    # always seem to have an incorrect, almost point-like bbox (??).
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
    ax.set_axis_off()

# ------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":

  fig = plt.figure()
  ax = fig.add_subplot()

  network = Network(gap = 5)

  input = network.add_layer(4, "x", shape = "square")
  input.annotate(r"$\mathbf{x}$", (-3, 0))

  hidden = network.add_layer(10, "h")

  output = network.add_layer(2, "y", shape = "square")
  output.annotate(r"$\mathbf{y}$", (+3, 0))

  hidden.nodes[3].color("C0").highlight()

  network.draw(ax)

  plt.show()