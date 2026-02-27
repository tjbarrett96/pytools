import numpy as np

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
  "va": "center"
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
    label: str = "",
    shape: str = _default_node_opts["shape"],
    size: float = _default_node_opts["size"],
    patch_kwargs: dict[str, Any] = None,
    text_kwargs: dict[str, Any] = None
  ):

    if shape not in _create_patch:
      raise ValueError(f"Node shape must be one of {list(_create_patch)}.")

    if size < 0:
      raise ValueError(f"Node radius must be >= 0.")
    
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

    # create shape patch object
    if patch_kwargs is None:
      patch_kwargs = {}
    patch_kwargs = {**_default_patch_opts, **patch_kwargs}
    self.patch = _create_patch[shape](self.x, self.y, self.size, **patch_kwargs)

    # create label text object
    if text_kwargs is None:
      text_kwargs = {}
    text_kwargs = {**_default_text_opts, **text_kwargs}
    self.label = mpl_text.Text(self.x, self.y, label, **text_kwargs)

  """Returns 2D coordinate shifted from center by (dx, dy) in units of the radius."""
  def relpos(self, dx: float, dy: float) -> np.ndarray:
    return np.array([self.x + dx * self.size/2, self.y + dy * self.size/2])
  
  """Make a Connection from this node to another."""
  def connect(self, other, **kwargs):
    return Connection(self, other, **kwargs)

  """Add this node to the given or current axes."""
  def draw(self, ax: mpl_axes.Axes = None):
    if ax is None:
      ax = plt.gca()
    ax.add_patch(self.patch)
    ax.add_artist(self.label)

# ------------------------------------------------------------------------------------------------
    
class Layer:

  def __init__(
    self,
    x: float,
    nodes: int,
    symbol: str = "",
    gap: float = 0.2,
    label_format: Callable[[int, int], str] = None,
    **kwargs
  ):
    
    if nodes < 0:
      raise ValueError(f"Number of nodes in a layer must be >= 0.")

    # position and spacing
    self.x = x
    self.gap = gap

    # label format function
    if label_format is None:
      if symbol:
        label_format = lambda node: fr"${symbol}_{{{node}}}$"
      else:
        label_format = lambda node: ""

    size = kwargs.get("radius", _default_node_opts["size"])
    dy = size + self.gap * size
    y_length = nodes * size + (nodes - 1) * (self.gap * size)
    y_start = y_length / 2

    self.nodes: list[Node] = []
    for n in range(nodes):
      self.nodes.append(Node(
        (self.x, y_start - n * dy),
        label_format(n),
        **kwargs
      ))

  """Make a mapping from (NodeA, NodeB) -> Connection from each node in this layer to another layer."""
  def connect(self, other, **kwargs):
    result = {}
    for this_node in self.nodes:
      for other_node in other.nodes:
        result[this_node, other_node] = this_node.connect(other_node, **kwargs)
    return result

  """Add this layer of nodes to the given or current axes."""
  def draw(self, ax: mpl_axes.Axes = None):
    for node in self.nodes:
      node.draw(ax)

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

# ------------------------------------------------------------------------------------------------

class Network:

  # ------------------------------------------------------------------------------------------------

  def __init__(
    self,
    gap: float = 3
  ):

    self.layers = []
    self.connections = {}

    self.gap = gap
    if self.gap < 0:
      raise ValueError(f"Network layer gap size must be >= 0.")
  
  # ------------------------------------------------------------------------------------------------

  def add_layer(self, *args, connect = True, **kwargs):
    
    x = len(self.layers) * self.gap

    prev_layer = self.layers[-1] if len(self.layers) > 0 else None
    new_layer = Layer(x, *args, **kwargs)

    self.layers.append(new_layer)

    if connect and prev_layer is not None:
      self.connections.update(prev_layer.connect(new_layer))

  # ------------------------------------------------------------------------------------------------

  def draw(self, ax: mpl_axes.Axes = None):

    if ax is None:
      ax = plt.gca()

    for layer in self.layers:
      layer.draw(ax)
    for connection in self.connections.values():
      connection.draw(ax)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()

  # ------------------------------------------------------------------------------------------------
    
  def color_node(self, layer, index, color, connections = True, highlight = True):

    self.layers[layer].nodes[index].patch.set_facecolor(color)

    if connections:
      for key, connection in self.connections.items():
        if (layer, index) in key:
          connection.line.set_color(color)
          connection.line.set_alpha(0.25)

    if highlight:
      self.highlight_connections(layer, index)

  # ------------------------------------------------------------------------------------------------
          
  def highlight_connections(self, layer, index):
    for key, connection in self.connections.items():
      if (layer, index) in key:
        connection.line.set_linewidth(1.5)
        connection.line.set_alpha(1)
      else:
        connection.line.set_linewidth(1)
        connection.line.set_alpha(0.25)

  # ------------------------------------------------------------------------------------------------
        
  def label_node(self, text, layer, index, dy = 0):
    node = self.layers[layer].nodes[index]
    plt.annotate(text, (node.x, node.y + dy*node.r), fontsize = 10, va = "bottom", ha = "center")

# ------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":

  fig = plt.figure()
  ax = fig.add_subplot()

  network = Network(gap = 5)
  network.add_layer(3, "x", shape = "square")
  network.add_layer(10, "h")
  network.add_layer(2, "y", shape = "square")
  network.draw(ax)

  # node = Node((0, 0), "x")
  # node.draw(ax)

  # ax.set_aspect("equal")
  # # ax.autoscale_view()
  # plt.xlim(-1.5, 1.5)
  # plt.ylim(-1.5, 1.5)

  plt.show()