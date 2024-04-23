import numpy as np
import jax.numpy as jnp
import jax

from typing import Tuple
import matplotlib.pyplot as plt

import mesher
import material
import bcs


_BilinMesh = mesher.BilinearThermalMesher


class FEA:
  def __init__(self,
               mesh: _BilinMesh,
               material: material.ThermalMaterial,
               bc: bcs.BC):
    self.mesh, self.material, self.bc = mesh, material, bc
    self.D0 = self.FE_compute_element_stiffness()


  def FE_compute_element_stiffness(self) -> np.ndarray:
    KE = np.array(
          [ 2./3., -1./6., -1./3., -1./6.,
          -1./6.,  2./3., -1./6., -1./3.,
          -1./3., -1./6.,  2./3., -1./6.,
          -1./6., -1./3., -1./6.,  2./3.]).reshape((4, 4))
    return KE


  def compute_elem_stiffness_matrix(self,
                                    therm_cond: jnp.ndarray)->jnp.ndarray:
    """
    Args:
      therm_cond: Array of size (num_elems,) which contain the modulus
        of each element
    Returns: Array of size (num_elems, 4, 4) which is the structual
      stiffness matrix of each of the bilinear quad elements. Each element has
      8 dofs corresponding to the x and y displacements of the 4 noded quad
      element.
    """
    # e - element, i - elem_nodes j - elem_nodes
    return jnp.einsum('e, ij->eij', therm_cond, self.D0)


  def assemble_stiffness_matrix(self, elem_stiff_mtrx: jnp.ndarray):
    """
    Args:
      elem_stiff_mtrx: Array of size (num_elems, 8, 8) which is the structual
        stiffness matrix of each of the bilinear quad elements. Each element has
        8 dofs corresponding to the x and y displacements of the 4 noded quad
        element.
    Returns: Array of size (num_dofs, num_dofs) which is the assembled global
      stiffness matrix.
    """
    glob_stiff_mtrx = jnp.zeros((self.mesh.num_dofs, self.mesh.num_dofs))
    glob_stiff_mtrx = glob_stiff_mtrx.at[(self.mesh.iK, self.mesh.jK)].add(
                                      elem_stiff_mtrx.T.flatten('F'))
    return glob_stiff_mtrx


  def solve(self, glob_stiff_mtrx):
    """Solve the system of Finite element equations.
    Args:
      glob_stiff_mtrx: Array of size (num_dofs, num_dofs) which is the assembled
        global stiffness matrix.
    Returns: Array of size (num_dofs,) which is the displacement of the nodes.
    """
    k_free = glob_stiff_mtrx[self.bc.free_dofs,:][:,self.bc.free_dofs]

    u_free = jax.scipy.linalg.solve(
          k_free,
          self.bc.force[self.bc.free_dofs],
          assume_a = 'pos', check_finite=False)
    u = jnp.zeros((self.mesh.num_dofs))
    u = u.at[self.bc.free_dofs].add(u_free.reshape(-1))
    return u


  def compute_compliance(self, u:jnp.ndarray)->jnp.ndarray:
    """Objective measure for structural performance.
    Args:
      u: Array of size (num_dofs,) which is the displacement of the nodes
        of the mesh.
    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    return jnp.dot(u, self.bc.force.flatten())


  def loss_function(self, density:jnp.ndarray)->float:
    """Wrapper function that takes in density field and returns compliance.
    Args:
      density: Array of size (num_elems,) that contain the density of each
        of the elements for FEA.
    Returns: Structural compliance, which is a measure of performance. Lower
      compliance means stiffer and stronger design.
    """
    therm_cond = self.material.compute_SIMP_material_modulus(density)
    elem_stiffness_mtrx = self.compute_elem_stiffness_matrix(therm_cond)
    glob_stiff_mtrx = self.assemble_stiffness_matrix(elem_stiffness_mtrx)
    u = self.solve(glob_stiff_mtrx)
    return self.compute_compliance(u)


  def plot_temperature(self, u, density = None)->None:
    x, y = np.mgrid[:self.mesh.nelx, :self.mesh.nely]

    if density is not None:
      delta = delta*np.round(density)

    z = delta.reshape(self.mesh.nelx+1, self.mesh.nely+1)
    im = plt.pcolormesh(x, y, z, cmap='coolwarm')
    plt.title('temperature')
    plt.colorbar(im)
