import numpy as np
import jax.numpy as jnp



class ThermalMaterial:
  def __init__(self, conductivity: float, heat_capacity: float,
               mass_density: float):
    self.conductivity, self.heat_capacity = conductivity, heat_capacity
    self.mass_density = mass_density
  
  def compute_SIMP_material_modulus(self, density: jnp.ndarray,
                                    penal: float = 3.,
                                    cond_min: float = 1e-3)->jnp.ndarray:
    """
      K = rho_min + K0*( density)^penal
    Args:
      density: Array of size (num_elems,) with values in range [0,1]
      penal: SIMP penalization constant, usually assumes a value of 3
      cond_min: Small value added to the modulus to prevent matrix singularity
    Returns: Array of size (num_elems,) which contain the penalized conductivity
      at each element
    """
    return cond_min + self.conductivity*(density**penal)