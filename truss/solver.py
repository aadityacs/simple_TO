
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jax_sprs
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spy_sprs

import truss_structure as trs


def sparse_solver(stiff_matrix: jax_sprs.BCOO,
                  force: jnp.ndarray):

  def mv(u):
    Ku = stiff_matrix @ u
    return Ku

  def solver_wrapper(b):

    K_sp = spy_sprs.coo_matrix((jax.lax.stop_gradient(stiff_matrix.data),
                                     (stiff_matrix.indices[:,0],
                                      stiff_matrix.indices[:,1])),
                                  shape=stiff_matrix.shape)
    x = spy_sprs.linalg.spsolve(K_sp, b)
    return x.astype(np.float32)

  result_shape = jax.ShapeDtypeStruct(force.shape, np.float32)

  cust_solver = lambda mv, b: jax.pure_callback(solver_wrapper, result_shape, b)
  sol = jax.lax.custom_linear_solve(mv, force, cust_solver, symmetric=True)
  return sol.reshape(-1)

class TrussSolver:

  def __init__(self, truss: trs.TrussSkeleton,
               bc: trs.BC):
    self.truss, self.bc = truss, bc

    self.compute_global_connectivity()
    self.compute_elem_stiffness_matrices()
    self.assemble_boundary_conditions()


  @property
  def dofs_per_node(self)->int:
    return 2


  @property
  def num_dofs(self)->int:
    return self.truss.num_nodes*self.dofs_per_node


  @property
  def num_dofs_per_bar(self)->int:
    return 4


  def compute_global_connectivity(self):
    """Compute connectivity information for performing FEA.
    Attributes:
      elem_dof_mat: Array of (num_bars, num_dofs_per_bar) that contain the
        global dof number associated with each bar.
      iK:
      jK:

    """
    self.elem_dof_mat=np.zeros((self.truss.num_bars,
                                self.num_dofs_per_bar), dtype=int)

    for br in range(self.truss.num_bars):
      n1, n2 = self.truss.connectivity[br, 0], self.truss.connectivity[br, 1]
      self.elem_dof_mat[br, :] = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1])

    self.iK = np.kron(self.elem_dof_mat,
                      np.ones((self.num_dofs_per_bar ,1))).flatten()
    self.jK = np.kron(self.elem_dof_mat,
                      np.ones((1, self.num_dofs_per_bar))).flatten()


  def compute_elem_stiffness_matrices(self):
    """
    Attributes:
      K0: Array of (num_bars, num_dofs_per_bar, num_dofs_per_bar) that contain
        the stiffness matrices of the bars with unit area and Young's modulus.
    """
    rot_mtrx = (
        jnp.einsum('i,jk->ijk',jnp.cos(self.truss.bar_orientation),
                               jnp.array([[1,0,0,0],
                                          [0,0,1,0]])) +
        jnp.einsum('i,jk->ijk',jnp.sin(self.truss.bar_orientation),
                              jnp.array([[0,1,0,0],
                                         [0,0,0,1]]))
        )

    k0 = jnp.array([[1., -1.],
                    [-1., 1.]])

    self.K0 = jnp.einsum('b, bfr -> bfr', 1./self.truss.bar_length,
                            jnp.einsum('bfw, bwr -> bfr',
                                jnp.einsum('btf, tw -> bfw', rot_mtrx, k0),
                                rot_mtrx))


  def assemble_boundary_conditions(self, fixture_penalty=1e12):

    # force
    self.force = jnp.zeros((self.num_dofs,))
    self.force = self.force.at[2*self.bc.force_nodes].set(self.bc.force_x)
    self.force = self.force.at[2*self.bc.force_nodes + 1].set(self.bc.force_y)

    #fixtures
    self.fixed_dofs = np.hstack((2*self.bc.fixed_x_nodes,
                               2*self.bc.fixed_y_nodes + 1))

    self.free_dofs = np.setdiff1d(np.arange(self.num_dofs), self.fixed_dofs)


    fixed_indices = np.stack((self.fixed_dofs, self.fixed_dofs), axis=-1)
    fixture_penalty_values = fixture_penalty*np.ones((self.fixed_dofs.shape[0]))
    fixture_shape = (self.num_dofs, self.num_dofs)
    self.stiff_dirichlet = jax_sprs.BCOO((fixture_penalty_values, fixed_indices),
                                          shape = fixture_shape)


  def solve(self, youngs_mod: jnp.ndarray, area: jnp.ndarray)->jnp.ndarray:
    node_idx = np.stack((self.iK, self.jK)).astype(np.int32).T
    elem_stiff_mtrx = jnp.einsum('b, bjk->bjk', youngs_mod*area, self.K0)

    stiff_glob = jax_sprs.BCOO((elem_stiff_mtrx.flatten(order='C'), node_idx),
                               shape=(self.num_dofs, self.num_dofs))

    u = sparse_solver(stiff_glob + self.stiff_dirichlet,
                      self.force)

    return u
  

  def compute_compliance(self, u: jnp.ndarray)->jnp.ndarray:
    return jnp.einsum('i, i ->', u, self.force)

  def get_volume(self, area: jnp.ndarray)->jnp.ndarray:
    return jnp.einsum('i,i->',self.truss.bar_length, area)