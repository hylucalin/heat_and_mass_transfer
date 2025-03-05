# Question 2(d) of Examples Paper 1
# This is only for my personal simulation
# Observation: if def c = CFL = (alpha*time_step)/(cell_size_x*cell_size_y), 1/4 seems to be a critical number for numerical stability (e.g. alpha = 10.09 * 1E-5, dx=dy=0.01 and dt=0.24785, CFL = 0.25008065

import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit

# Constants
MAX = 1
MIN = 0
alpha = 10.09 * 1E-5    # m2/s
pho_cp = 8960 * 410     # kg/m3 * J/kgK = J/m3K
g_dot = 0*1E5           # W/m3

# Recrtangular window setup
Window_Width = 16
Window_Height = 7

# Simulation boundaries
## Note: Object should be bounded in x and y
x_lim = [0, 1]
y_lim = [0, 1]

# Simulation parameters
cell_size_x = 0.01
cell_size_y = 0.01
time_step = 0.24785
sim_time = 10000
Num_of_iterations = int(sim_time/time_step)
Num_of_iteration = 0
time = 0

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
def step(T, mask, dt = time_step):
  dT = dt * (alpha * jax_laplacian(T) + g_dot/pho_cp)
  T = jnp.where(mask, T + dT, T)
  T = T.at[0, :].set(T[1, :])
  return T

# Temperature grid initialisation
Num_of_x_cell = int((x_lim[MAX] - x_lim[MIN])/cell_size_x)
Num_of_y_cell = int((y_lim[MAX] - y_lim[MIN])/cell_size_y)
origin_x = Num_of_x_cell//2
origin_y = Num_of_y_cell//2

xs = jnp.linspace(x_lim[MIN], x_lim[MAX], Num_of_x_cell)
ys = jnp.linspace(y_lim[MIN], y_lim[MAX], Num_of_y_cell)
xs, ys = jnp.meshgrid(xs, ys) # xs has same entries in each column whereas ys has the same entries in each row
print(xs, '\n', ys)

# Initial condition in K
T0 = 298
T_jnp = jnp.ones((Num_of_y_cell, Num_of_x_cell)) * T0   # columns x rows

# Boundary Conditions, the edge is held at constant temperature
T1 = 350
T_jnp = T_jnp.at[:,0].set(T1)   # all row, first column



# Returns a mask indicating the update region
def mask_unit_circle(X, Y):
  return (((X**2 + Y**2) <1) & (X>0) & (Y>0))

quater_circle_mask = mask_unit_circle(xs, ys)
print(quater_circle_mask)

# Using for loop to look for convergence
for i in range(Num_of_iterations):
  Num_of_iteration += 1
  time = Num_of_iteration*time_step
  T_jnp = step(T_jnp, mask = quater_circle_mask)
  
#  line.set_ydata(T_jnp[:,origin_y])
#  cmap_plot.set_array(T_jnp.ravel())
#  ax.set_title(f"Temperature Distribution (Iteration {Num_of_iteration})")  # Update title
#  ax.legend([line], [f"Numerical at iteration {Num_of_iteration}"])  # Update legend
#  plt.draw()  # Redraw plot
#  plt.pause(0.000001)  # Pause to show the update


fig, axes = plt.subplots(1, 2, figsize=(Window_Width, Window_Height))
ax=axes[0] # Left subplot
line, = ax.plot(xs[origin_y,:], T_jnp[origin_y,:], label=f"Numerical at centre y={origin_y*cell_size_y}")
line2, = ax.plot(xs[0, :], T_jnp[0, :], label="y=0")
ax_color=axes[1]  # Right subplot
cmap_plot = ax_color.pcolormesh(xs, ys, T_jnp, shading='gouraud', cmap='jet')
cbar = plt.colorbar(cmap_plot)  # Add colorbar

# Add title and labels
ax.set_title(f"Temperature Distribution (Time {time})")
ax.set_xlabel("X")
ax.set_ylabel("Temperature")

# Add legend
ax.legend()

plt.show()



