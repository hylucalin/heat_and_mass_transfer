import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit

# Constants
MAX = 1
MIN = 0
alpha = 10.09 * 1E-5    # m2/s
pho_cp = 8960 * 410     # kg/m3 * J/kgK = J/m3K
g_dot = 1*1E5           # W/m3

# Recrtangular window setup
Window_Width = 16
Window_Height = 7

# Simulation boundaries
## Note: Object should be bounded in x and y
x_lim = [-1, 1]
y_lim = [-1, 1]

# Simulation parameters
cell_size_x = 0.01
cell_size_y = 0.01
time_step = 0.1
sim_time = 1500
Num_of_iterations = int(sim_time/time_step)
Num_of_iteration = 0
time = 0

# Temperature grid initialisation
Num_of_x_cell = int((x_lim[MAX] - x_lim[MIN])/cell_size_x)
Num_of_y_cell = int((y_lim[MAX] - y_lim[MIN])/cell_size_y)
origin_x = Num_of_x_cell//2
origin_y = Num_of_y_cell//2

# Define a JAX accelerated Laplacian
# This has an error term of delta_x^2
# T is the 2D jnp array of temperature in Kelvins
# ds is default to cell_size_x, the cell size in the x direction
@jit
def jax_laplacian(T, ds=cell_size_x):
  return (-4*T
  +jnp.roll(T, 1, axis=0) + jnp.roll(T, -1, axis=0)
  +jnp.roll(T, 1, axis=1) + jnp.roll(T, -1, axis=1))/ (ds*ds)

@jit
def step(T, dt = time_step):
  dT = dt * (alpha * jax_laplacian(T) + g_dot/pho_cp)
  return T.at[1:-1, 1:-1].set(T[1:-1, 1:-1] + dT[1:-1, 1:-1])


xs = np.linspace(x_lim[MIN], x_lim[MAX], Num_of_x_cell)
ys = np.linspace(y_lim[MIN], y_lim[MAX], Num_of_y_cell)
ys, xs = np.meshgrid(ys, xs)

# Initial condition in K
T0 = 298
T_jnp = jnp.ones((Num_of_x_cell, Num_of_y_cell)) * T0


print(f'T_jnp = {T_jnp}')

# Boundary Conditions, the edge is held at constant temperature
#T0 = 298

## Constant inner temperature of 1000K
T_jnp = T_jnp.at[Num_of_x_cell//2, Num_of_y_cell//2].set(1000)
T_jnp = T_jnp.at[Num_of_x_cell//2-1, Num_of_y_cell//2].set(1000)
T_jnp = T_jnp.at[Num_of_x_cell//2-1, Num_of_y_cell//2-1].set(1000)
T_jnp = T_jnp.at[Num_of_x_cell//2, Num_of_y_cell//2-1].set(1000)



# Using for loop to look for convergence
for i in range(Num_of_iterations):
  Num_of_iteration += 1
  time = Num_of_iteration*time_step
  T_jnp = step(T_jnp)
  
def plot_result():
  fig, axes = plt.subplots(1, 2, figsize=(Window_Width, Window_Height))
  ax=axes[0] # Left subplot
  line, = ax.plot(xs[:,origin_y], T_jnp[:,origin_y], label=f"Numerical at time {time} sec")
  ax_color=axes[1]  # Right subplot
  cmap_plot = ax_color.pcolormesh(xs, ys, T_jnp, shading='gouraud', cmap='jet')
  cbar = plt.colorbar(cmap_plot)  # Add colorbar

  # Add title and labels
  ax.set_title(f"Temperature Distribution (time {time} sec)")
  ax.set_xlabel("X")
  ax.set_ylabel("Temperature")

  # Add legend
  ax.legend()
  
  plt.show()
  
plot_result()
