"""Sample truss problems in 2D."""


import dataclasses
from enum import Enum, auto
from typing import Tuple
import numpy as np
import jax.numpy as jnp

import truss_structure as trs


class Samples(Enum):
  """Sample truss problems in 2D."""
  ONE_TRIANGLE = auto()
  TWO_TRIANGLES = auto()
  GRID_TRUSS = auto()


def get_sample_truss(sample: Samples)->Tuple[trs.TrussSkeleton, trs.BC]:

  if sample == Samples.TWO_TRIANGLES:
    node_xy = jnp.array([[0.5, 1.5, 1., 0., 2.],
                        [1., 1., 0., 0., 0.]]).T

    connectivity = np.array([[0, 0, 0, 1, 1, 2, 2],
                            [1, 2, 3, 2, 4, 3, 4]]).T.astype(np.int32)

    truss_structure = trs.TrussSkeleton(node_xy = node_xy,
                                    connectivity = connectivity)

    bc = trs.BC(fixed_x_nodes = np.array([3, 4]).astype(np.int32),
            fixed_y_nodes = np.array([3, 4]).astype(np.int32),
            force_nodes = np.array([0, 1]).astype(np.int32),
            force_x = 1.e3*jnp.array([1., 2.]),
            force_y = 1.e3*jnp.array([-2., 0.])
            )

  elif sample == Samples.ONE_TRIANGLE:
    node_xy = jnp.array([[0., 0., 1.],
                        [0., 1., 0.]]).T
    connectivity = np.array([[0, 0, 1],
                            [1, 2, 2]]).T.astype(np.int32)
    truss_structure = trs.TrussSkeleton(node_xy = node_xy,
                                    connectivity = connectivity)

    bc = trs.BC(fixed_x_nodes = np.array([0, 1]).astype(np.int32),
            fixed_y_nodes = np.array([0, 1]).astype(np.int32),
            force_nodes = np.array([2]).astype(np.int32),
            force_x = 1.e3*jnp.array([1.,]),
            force_y = 1.e3*jnp.array([1.,]),)
  
  elif sample == Samples.GRID_TRUSS:
    node_xy = jnp.array(
      [[0., 0., 0., 1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.],
        [0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2.]]).T

    connectivity = np.array([
        (0, 1), (1, 2),  # Vertical members on the left
        (3, 4), (4, 5),  # Vertical members in the middle-left
        (6, 7), (7, 8),  # Vertical members in the middle-right
        (9, 10), (10, 11),  # Vertical members on the right
        (12, 13), (13, 14),  # Vertical members on the far right
        (0, 3), (3, 6), (6, 9), (9, 12),  # Horizontal members on the bottom
        (1, 4), (4, 7), (7, 10), (10, 13),  # Horizontal members in the middle
        (2, 5), (5, 8), (8, 11), (11, 14),  # Horizontal members on the top
        (0, 4), (1, 3), (1, 5), (2, 4),  # Diagonal members in the first section
        (3, 7), (4, 6), (4, 8), (5, 7),  # Diagonal members in the second section
        (6, 10), (7, 9), (7, 11), (8, 10),  # Diagonal members in the third section
        (9, 13), (10, 12), (10, 14), (11, 13)  # Diagonal members in the fourth section
    ]).astype(np.int32)

    truss_structure = trs.TrussSkeleton(node_xy = node_xy,
                                    connectivity = connectivity)
    
    bc = trs.BC(fixed_x_nodes = np.array([0, 1, 2]).astype(np.int32),
        fixed_y_nodes = np.array([0, 1, 2]).astype(np.int32),
        force_nodes = np.array([13]).astype(np.int32),
        force_x = 1.e3*jnp.array([-1.]),
        force_y = 1.e3*jnp.array([0.])
        )

  return truss_structure, bc