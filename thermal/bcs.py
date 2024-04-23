
import dataclasses
from enum import Enum, auto
import numpy as np
import jax.numpy as jnp
import mesher

_Mesher = mesher.BilinearThermalMesher

@dataclasses.dataclass
class BC:
  """
  Attributes:
    force: Array of size (num_dofs,) that contain the imposed load on each dof.
    fixed_dofs: Array of size (num_fixed_dofs,) that contain all the dof numbers
      that are fixed.
  """
  force: jnp.ndarray
  fixed_dofs: np.ndarray

  @property
  def num_dofs(self):
    return self.force.shape[0]

  @property
  def free_dofs(self):
    return np.setdiff1d(np.arange(self.num_dofs), self.fixed_dofs)
  

class SampleThermalBoundaryConditions(Enum):
  LEFT_EDGE_FIXED_RIGHT_TOP_HEAT = auto()
  ALL_EDGE_FIXED_ALL_NODE_HEAT = auto()
  LEFT_MID_EDGE_FIXED_ALL_NODE_HEAT = auto()
  LEFT_AND_RIGHT_EDGE_FIXED = auto()


def get_sample_bc(mesh:_Mesher, sample:SampleThermalBoundaryConditions)->BC:
  force = np.zeros((mesh.num_nodes,1))

  if(sample == SampleThermalBoundaryConditions.LEFT_EDGE_FIXED_RIGHT_TOP_HEAT):
    fixed_dofs = np.arange(0, int(mesh.nely))
    force = np.zeros((mesh.num_nodes, 1))
    force[int(mesh.nelx*(mesh.nely+1) + mesh.nely/2):
              (mesh.nely+1)*(mesh.nelx+1), :] = 1.

  if(sample == SampleThermalBoundaryConditions.ALL_EDGE_FIXED_ALL_NODE_HEAT):
    force = np.ones((mesh.num_nodes, 1))
    left_edge = np.arange(0, mesh.nely+1, 1)
    right_edge =  np.arange(mesh.nelx*(mesh.nely+1), (mesh.nelx+1)*(mesh.nely+1), 1)
    top_edge =  np.arange(mesh.nely, (mesh.nelx)*(mesh.nely+1), mesh.nely+1)
    btm_edge = np.arange(0, (1+mesh.nely)*(mesh.nelx+1), mesh.nely+1)
    fixed_dofs = np.union1d(left_edge, right_edge)
    fixed_dofs = np.union1d(fixed_dofs, top_edge)
    fixed_dofs = np.union1d(fixed_dofs, btm_edge)

  if(sample == SampleThermalBoundaryConditions.LEFT_MID_EDGE_FIXED_ALL_NODE_HEAT):
    force = 1e-5*np.ones((mesh.num_nodes, 1))
    left_mid_edge = np.arange(int(0.5*mesh.nely) - 5 ,
                              int(0.5*mesh.nely) + 5, 1)
    fixed_dofs = left_mid_edge

  if(sample == SampleThermalBoundaryConditions.LEFT_AND_RIGHT_EDGE_FIXED):
    
    right_btm = mesh.nelx*(mesh.nely+1)
    right_mid = int(right_btm + 0.5*mesh.nely)

    left_mid = int(0.5*mesh.nely)
    left_top = mesh.nely

    fixed_dofs = np.arange(left_mid, left_top)
                            #np.union1d(np.arange(left_mid, left_top),
                            # np.arange(right_btm, right_mid)
                          #  )
    force[np.arange(right_btm, right_mid), 0] = 1.e-2

  return BC(force=force, fixed_dofs=fixed_dofs)