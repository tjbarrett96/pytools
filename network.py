# numeric modules
import numpy as np

# python modules
import itertools

# matplotlib modules
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.textpath as mpl_textpath
import matplotlib.font_manager as mpl_font
import matplotlib.patches as mpl_patch
import matplotlib.lines as mpl_line
import matplotlib.text as mpl_text
import matplotlib.axes as mpl_axes
import matplotlib.transforms as mpl_transforms
import matplotlib.artist as mpl_artist

# type hints
from typing import Any, Callable, Self
from collections.abc import Iterable
from numbers import Number

# local pytools
import pytools.util as util

# TODO: add unit option to relpos, defaults to internal size/width, switch for absolute
# TODO: add gap variables to npad, lpad to distinguish node/layer sep

# ------------------------------------------------------------------------------------------------

_default_patch_opts = {
  "color": "white",
  "ec": "k",
  "lw": 0.5
}

_blank_patch_opts = {
  "fill": False,
  "lw": 0
}

_default_line_opts = {
  "color": "black",
  "alpha": 0.25,
  "lw": 1,
  "zorder": 0
}

_default_text_opts = {
  "color": "black",
  "lw": 0
}

_default_node_opts = {
  "shape": "round",
  "size": 1,
  "anchor": "c"
}

_default_margin = 0.15
_default_pad = 0.15
_default_rounding_size = 0.1

_text_size_conversion = 0.2

# ------------------------------------------------------------------------------------------------

# positions of named anchor points relative to the node center, in units of (width/2, height/2)
_default_anchor_rules = {
  "c": np.array([ 0,  0]),
  "l": np.array([-1,  0]),
  "r": np.array([+1,  0]),
  "t": np.array([ 0, +1]),
  "b": np.array([ 0, -1])
}

# add corner anchors with names concatenated like "tr" (top-right)
for y in ("t", "b"):
  for x in ("l", "r"):
    _default_anchor_rules[f"{y[0]}{x[0]}"] = _default_anchor_rules[y] + _default_anchor_rules[x]

# ------------------------------------------------------------------------------------------------

"""Creates a Figure and Axes with no margins, equal aspect ratio, and no spines."""
def make_canvas():
  fig = plt.figure()
  ax: mpl_axes.Axes = fig.add_axes([0, 0, 1, 1])
  ax.set_aspect("equal")
  ax.set_axis_off()
  return fig, ax

# ------------------------------------------------------------------------------------------------

"""Sets viewing range for given or current axes to include all artists."""
def autoscale(ax: mpl_axes.Axes = None):
  
  if ax is None:
    ax = plt.gca()

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
    size: Number | tuple[Number, Number] | Self | Iterable[Self] = _default_node_opts["size"],
    shape: str = _default_node_opts["shape"],
    anchor: str = _default_node_opts["anchor"],
    margin: float = _default_margin,
    pad: float = _default_pad,
    **kwargs
  ):
    
    # default center position (before placement)
    self.xy = np.array([0.0, 0.0])

    # z-order
    self.z = 0

    # margin (outer spacing) and padding (inner spacing)
    self.margin = margin
    self.pad = pad

    # list of dependent Nodes placed relative to this Node
    self.relatives: list[Self] = []

    # list of other Nodes that should be drawn with this one
    self.children: list[Self] = []

    # wrap single-Node argument as iterable list
    if isinstance(size, Node):
      size = [size]

    # parse node size option
    if not util.is_iterable(size):
      # one float (width == height)
      self.width, self.height = size, size
    elif isinstance(size[0], Number):
      # tuple[float, float] ~ (width, height)
      self.width, self.height = size
    else:
      # minimal bounding box around list of other Nodes
      bbox = mpl_transforms.Bbox.union([node._get_bbox() for node in size])
      # width and height including padding
      self.width = bbox.width + 2*self.pad
      self.height = bbox.height + 2*self.pad
      # center point from lower-left corner
      self.xy[0] = (bbox.min[0] - self.pad) + self.width/2
      self.xy[1] = (bbox.min[1] - self.pad) + self.height/2
      # keep track of all sub-Nodes contained inside this wrapper
      for node in size:
        self.children.append(node)
      # set z-order below children
      if len(self.children) > 0:
        self.z = min(node.z for node in self.children) - 1

    # dictionary of named anchor positions
    self.anchors = {anchor: np.array([0.0, 0.0]) for anchor in _default_anchor_rules}
    self._anchor_basis = np.array([self.width/2, self.height/2])
    self._outer_basis = self._anchor_basis + (self.margin, self.margin)
    self._inner_basis = self._anchor_basis - (self.pad, self.pad)
    self._update_anchors()

    # default anchor for this node
    self.anchor = anchor

    # create shape patch object with default options, forwarding any extra keywords
    patch_kwargs = {**_default_patch_opts, **kwargs}
    self.box = mpl_patch.FancyBboxPatch(
      self["bl"],
      self.width,
      self.height,
      f"{shape}, pad = 0" \
        + (f", rounding_size = {_default_rounding_size}" if shape == "round" else ""),
      zorder = self.z,
      **patch_kwargs
    )

    # initialize placement
    self.place(self.xy, "c")

  """Update anchor positions relative to the current center position."""
  def _update_anchors(self):
    self.x, self.y = self.xy
    for anchor, relpos in _default_anchor_rules.items():
      self.anchors[anchor] = self.xy + self._anchor_basis * relpos
      self.anchors[f"o{anchor}"] = self.xy + self._outer_basis * relpos
      self.anchors[f"i{anchor}"] = self.xy + self._inner_basis * relpos

  """Shorthand for accessing named anchor points, e.g. node['center'] or node['c']."""
  def __getitem__(self, anchor: str):
    # TODO: add some dynamically calculated divisions, like "r(1/4)", "r(2/4)", etc. divides right edge
    return self.anchors[anchor]

  """Returns 2D coordinate shifted from node center by (dx, dy) in units of (width/2, height/2)."""
  def relpos(self, dx: float, dy: float) -> np.ndarray:
    # TODO: consider harmonizing with other usage of tuples instead of separate args
    return self.xy + self._anchor_basis * (dx, dy)
  
  """Positions this node (and moves dependents) so that its anchor is located at the given (x, y)."""
  def place(self, xy: tuple[float, float], anchor: str = None):
    if anchor is None:
      anchor = self.anchor
    # calculate translation amount, shift center and anchors
    dr = xy - self.anchors[anchor]
    self.xy = self.xy + dr
    self._update_anchors()
    # shift bbox patch
    self.box.set_x(self["bl"][0])
    self.box.set_y(self["bl"][1])
    # shift text annotations
    for node in self.children:
      node.place(node.anchors[node.anchor] + dr)
    # shift other relative nodes
    for node in self.relatives:
      node.place(node.anchors[node.anchor] + dr)
    return self

  """Parse a specified location as either an anchor string, or relpos tuple."""
  def _parse_loc(self, loc: str | tuple[float, float]):
    if isinstance(loc, str):
      return self.anchors[loc]
    else:
      return self.relpos(*loc)

  """Add text annotation at the given location (anchor name or relpos tuple)."""
  def label(self, text: str, loc: str | tuple[float, float] = "c", **kwargs):
    loc = self._parse_loc(loc)
    kwargs = {**_default_text_opts, **kwargs}
    node = Text(text, **kwargs).place(loc)
    self.children.append(node)
    return self

  """Make a Connection from this node to another."""
  def connect(self, other: "Node", **kwargs):
    return Connection(self, other, **kwargs)

  """Add this node to the given or current axes."""
  def draw(self, ax: mpl_axes.Axes = None):
    if ax is None:
      ax = plt.gca()
    ax.add_patch(self.box)
    for node in self.children:
      node.draw(ax)
    return self

  """Returns minimal bounding box (Bbox) around this node and all children, including text."""
  def _get_bbox(self):
    return mpl_transforms.Bbox.union([node.box.get_bbox() for node in [self, *self.children]])

  # """Highlights all connections to this node."""
  # def highlight(self, enable: bool = True):
  #   if self.network is None:
  #     raise ValueError("Highlighted node is missing reference to parent network.")
  #   for pair, connection in self.network.connections.items():
  #     if self in pair and enable:
  #       connection.highlight()
  #     else:
  #       connection.highlight(False)
  #   return self

  """Set the face color of the node."""
  def color(self, color: str):
    self.box.set_facecolor(color)
    return self
  
# ------------------------------------------------------------------------------------------------

class Text(Node):

  def __init__(
    self,
    text: str,
    anchor: str = _default_node_opts["anchor"],
    size: float = 1,
    **kwargs
  ):

    # create text path with bottom-left corner near (0, 0)
    self.path = mpl_textpath.TextPath((0, 0), text, size = size * _text_size_conversion, usetex = True)

    # convert path into patch, passing any keyword args for color, etc.
    patch_opts = {**_default_text_opts, **kwargs}
    self.patch = mpl_patch.PathPatch(self.path, **patch_opts)

    # get actual position and dimensions of text patch, won't be precisely at (0, 0)
    bbox = self.patch.get_path().get_extents()
    x0, y0 = bbox.min
    width, height = bbox.width, bbox.height

    # translate text patch to be centered on (0, 0) to match initialized Node
    self._translate_text_patch(-width/2 - x0, -height/2 - y0)

    # create empty bounding Node
    node_opts = {**_blank_patch_opts, "pad": 0, "margin": 0}
    super().__init__(size = (width, height), anchor = anchor, **node_opts)

  """Translate the text patch alone."""
  def _translate_text_patch(self, dx: float, dy: float):
    translate = mpl_transforms.Affine2D().translate(dx, dy)
    self.patch.set_path(self.patch.get_path().transformed(translate))

  """Overrides Node.place() to additionally place the text patch."""
  def place(self, xy: tuple[float, float], anchor: str = None):
    if anchor is None:
      anchor = self.anchor
    dr = xy - self.anchors[anchor]
    self._translate_text_patch(*dr)
    super().place(xy, anchor)
    return self

  """Overrides Node.draw() to additionally draw the text patch."""
  def draw(self, ax: mpl_axes.Axes = None):
    super().draw(ax)
    if ax is None:
      ax = plt.gca()
    ax.add_patch(self.patch)
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
    start: Node | tuple[Node, str],
    end: Node | tuple[Node, str],
    label: str = "",
    label_relpos: float = 0.5,
    line_kwargs = None,
    text_kwargs = None
  ):
    
    if isinstance(start, Node):
      start = (start, start.anchor)
    if isinstance(end, Node):
      end = (end, end.anchor)
    
    self.start_node, self.start_anchor = start
    self.end_node, self.end_anchor = end

    self.start_pos = self.start_node[self.end_anchor]
    self.end_pos = self.end_node[self.end_anchor]
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