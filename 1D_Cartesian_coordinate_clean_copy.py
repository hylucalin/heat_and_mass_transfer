import numpy as np
import matplotlib.pyplot as plt

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
cell_size_x = 0.05
cell_size_y = 0.05
time_step = 0.1
sim_time = 150
Num_of_iterations = int(sim_time/time_step)
MAX_percentage_change = 0.15
MIN_percentage_change = 0.01
MAX_time_step = 0.5
Num_of_iteration = 0
time = 0

# Temperature grid initialisation
Num_of_x_cell = int((x_lim[MAX] - x_lim[MIN])/cell_size_x)
Num_of_y_cell = int((y_lim[MAX] - y_lim[MIN])/cell_size_y)
origin_x = Num_of_x_cell//2
origin_y = Num_of_y_cell//2

print(f'Num_of_x_cell = {Num_of_x_cell}, Num_of_y_cell = {Num_of_y_cell}')

xs = np.linspace(x_lim[MIN], x_lim[MAX], Num_of_x_cell)
ys = np.linspace(y_lim[MIN], y_lim[MAX], Num_of_y_cell)
ys, xs = np.meshgrid(ys, xs)

# Initial condition in K
T = np.full_like(xs, fill_value=298)
T0 = 298

print(f'T = {T}')

# Boundary Conditions, the edge is held at constant temperature
#T0 = 298

## Constant inner temperature of 1000K
T[origin_x-1:origin_x+1,origin_y-1:origin_y+1] = 1000

# Using for loop to look for convergence
for i in range(Num_of_iterations):
  Num_of_iteration += 1
  max_percentage_change_reached = 0
  T_new =T
#  print(f'At time {time} sec.')
  for x_index in range(1, Num_of_x_cell-1):
    for y_index in range(1, Num_of_y_cell-1):
      x, y = x_index * cell_size_x+0.5*cell_size_x, y_index * cell_size_y+0.5*cell_size_y
      # Normal case, wouldn't need to be treated differently. It hasn't reached the boundary
      dT_dx = (T[x_index, y_index]-T[x_index-1, y_index])/cell_size_x
      dT_dx_next = (T[x_index+1, y_index]-T[x_index, y_index])/cell_size_x
      d2T_dx2 = (dT_dx_next - dT_dx)/cell_size_x

      dT_dy = (T[x_index, y_index] - T[x_index, y_index-1])/cell_size_y
      dT_dy_next = (T[x_index, y_index+1] - T[x_index, y_index])/cell_size_y
      d2T_dy2 = (dT_dy_next - dT_dy)/cell_size_y

      T_Lagrangian = d2T_dx2 + d2T_dy2
      dT_dt = alpha*T_Lagrangian + g_dot/pho_cp
    
      T_new[x_index, y_index]+=time_step*(dT_dt)
  T = T_new
  
def plot_result():
  fig, axes = plt.subplots(1, 2, figsize=(Window_Width, Window_Height))
  ax=axes[0] # Left subplot
  line, = ax.plot(xs[:,origin_y], T[:,origin_y], label=f"Numerical at iteration {Num_of_iteration}")
  ax_color=axes[1]  # Right subplot
  cmap_plot = ax_color.pcolormesh(xs, ys, T, shading='gouraud', cmap='jet')
  cbar = plt.colorbar(cmap_plot)  # Add colorbar

  # Add title and labels
  ax.set_title(f"Temperature Distribution (Iteration {Num_of_iteration})")
  ax.set_xlabel("X")
  ax.set_ylabel("Temperature")

  # Add legend
  ax.legend()

  plt.show()

plot_result()

