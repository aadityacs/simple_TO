import numpy as np
import jax.numpy as jnp

class StructuralMaterial:
  def __init__(self, youngs_modulus: float, poisson_ratio: float,
               mass_density: float):
    self.youngs_modulus, self.poisson_ratio = youngs_modulus, poisson_ratio
    self.mass_density = mass_density
  
  def compute_SIMP_material_modulus(self, density: jnp.ndarray,
                                    penal: float = 3.,
                                    young_min: float = 1e-3)->jnp.ndarray:
    """
      E = rho_min + E0*( density)^penal
    Args:
      density: Array of size (num_elems,) with values in range [0,1]
      penal: SIMP penalization constant, usually assumes a value of 3
      young_min: Small value added to the modulus to prevent matrix singularity
    Returns: Array of size (num_elems,) which contain the penalized modulus
      at each element
    """
    return young_min + self.youngs_modulus*(density**penal)