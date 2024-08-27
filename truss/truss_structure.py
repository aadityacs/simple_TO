"""Truss skeleton and connectivities."""


import dataclasses
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


@dataclasses.dataclass
class TrussSkeleton:
  """
  Attributes:
    node_xy: Array of (num_nodes, 2) that contain the x-y coordinates of the
      nodes of the truss.
    connectivity: Array of (num_bars, 2) that provide the connectivity of
      the nodes.
    bar_center: Array of (num_bars, 2) that contain the xy coordinate of the
      center of the bars.
    bar_length: Array of (num_bars,) that contain the length of the bars.
    bar_orientation: Array of (num_bars,) that contain the orientation (wrt x)
      of the bars in radians.
  """
  node_xy: jnp.ndarray
  connectivity: jnp.ndarray
  bar_center: jnp.ndarray = dataclasses.field(init=False)
  bar_length: jnp.ndarray = dataclasses.field(init=False)
  bar_orientation: jnp.ndarray = dataclasses.field(init=False)

  def __post_init__(self):
    self.bar_center = 0.5*(self.node_xy[self.connectivity[:, 0]] +
                           self.node_xy[self.connectivity[:, 1]] )

    self.bar_length = jnp.sqrt(
                          (self.node_xy[self.connectivity[:, 0]][:, 0] -
                          self.node_xy[self.connectivity[:, 1]][:, 0])**2 +
                          (self.node_xy[self.connectivity[:, 0]][:, 1] -
                          self.node_xy[self.connectivity[:, 1]][:, 1])**2 )

    self.bar_orientation = jnp.arctan2(
                          (self.node_xy[self.connectivity[:, 1]][:, 1] -
                          self.node_xy[self.connectivity[:, 0]][:, 1]),
                          (self.node_xy[self.connectivity[:, 1]][:, 0] -
                          self.node_xy[self.connectivity[:, 0]][:, 0]))



  @property
  def num_nodes(self)->int:
    return self.node_xy.shape[0]


  @property
  def num_dim(self)->int:
    return 2


  @property
  def num_bars(self)->int:
    return self.connectivity.shape[0]


@dataclasses.dataclass
class BC:
  """
  Attributes:
    fixed_x_nodes: Array of size (num_fixed_x_nodes,) that contain the nodes that
      are fixed along x.
    fixed_y_nodes: Array of size (num_fixed_y_nodes,) that contain the nodes that
      are fixed along y.
    force_nodes: Array of (num_forced_nodes,) that contain the nodes on which
      a force is applied.
    force_x: Array of (num_forced_nodes,) that contain the forces along X for
      each of the `force_nodes`. Provide `0` in case there is no x force on the
        node.
    force_y: Array of (num_forced_nodes,) that contain the forces along Y for
      each of the `force_nodes`. Provide `0` in case there is no y force on the
        node.
  """
  fixed_x_nodes: jnp.ndarray
  fixed_y_nodes: jnp.ndarray
  force_nodes: jnp.ndarray
  force_x: jnp.ndarray
  force_y: jnp.ndarray


def plot_truss(truss_structure: TrussSkeleton,
               bc: BC,
               area: jnp.ndarray,
               node_displacements: jnp.ndarray=None,
               ax=None,
               annotate_nodes: bool=False,
               show_fixtures: bool=False,
               show_forces: bool=False,
               title_str: str='',
               thresold: float = 0.1):

  if ax is None:
    _, ax = plt.subplots(1,1)

  area_min = np.amin(area)


  # plot the bars undeformed and if true the deformed
  for i in range(truss_structure.num_bars):
    n1, n2 = truss_structure.connectivity[i, 0], truss_structure.connectivity[i, 1]
    sx, sy = truss_structure.node_xy[n1, 0], truss_structure.node_xy[n1, 1]
    ex, ey = truss_structure.node_xy[n2, 0], truss_structure.node_xy[n2, 1]


    if area[i] < thresold*np.amax(area):
      thkns = 0.
    else:
      thkns = 1. + 3.*np.log(area[i]/area_min)
    plt.plot([sx,ex],[sy,ey], color = 'black', linewidth = thkns, alpha = 0.5)

    if node_displacements is not None:
      len_scale = np.amax(truss_structure.bar_length)
      scale = 0.1*len_scale/np.max(np.abs(node_displacements))
      dx1, dx2 = node_displacements[2*n1], node_displacements[2*n2]
      dy1, dy2 = node_displacements[2*n1+1], node_displacements[2*n2+1]
      plt.plot([sx + scale*dx1,
                ex + scale*dx2],
               [sy + scale*dy1,
                ey + scale*dy2],
                color = 'black',
                linestyle = 'dashed')

  if annotate_nodes :
    for i in range(truss_structure.num_nodes):
      plt.annotate('{:d}'.format(i), (truss_structure.node_xy[i, 0],
                                      truss_structure.node_xy[i, 1]))

  if show_fixtures:
    for i in bc.fixed_x_nodes:
      plt.scatter(truss_structure.node_xy[i, 0], truss_structure.node_xy[i,1],
                  marker=4, s=100, c = 'orange')

    for i in bc.fixed_y_nodes:
      plt.scatter(truss_structure.node_xy[i,0], truss_structure.node_xy[i,1],
                  marker=6, s=100, c = 'green')

  if show_forces:
    for ctr, nd in enumerate(bc.force_nodes):
      plt.quiver( truss_structure.node_xy[nd, 0],
                  truss_structure.node_xy[nd, 1],
                  bc.force_x[ctr],
                  bc.force_y[ctr],
                  color = 'purple')


  ax.set_aspect('equal')
  ax.grid(False)
  ax.set_title(title_str)
  return ax