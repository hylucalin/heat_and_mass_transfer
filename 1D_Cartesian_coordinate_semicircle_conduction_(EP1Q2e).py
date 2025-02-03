# Question 2(d) of Examples Paper 1
# This is only for my personal simulation

import numpy as np
import matplotlib.pyplot as plt

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
time_step = 0.01
sim_time = 100
Num_of_iterations = int(sim_time/time_step)
MAX_percentage_change = 0.5
MIN_percentage_change = 0.01
MAX_time_step = 0.05
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

# Boundary Conditions, the edge is held at constant temperature
T0 = 298
T1 = 350
T[0,:]=T1

print(f'T = {T}\nxs={xs}\nys={ys}')

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

# Returns a list of points on the unit circle
# the points are tuples
def find_unit_circle():
  unit_circle_list = []  # Use a list to collect tuples

  for x_index in range(Num_of_x_cell):
    x = x_index * cell_size_x
    y = np.sqrt(1 - x**2)
    y = y - y % cell_size_y  # Snap to grid
    unit_circle_list.append((x, y))  # Append as tuple

  # Convert to NumPy array after the loop (faster)
  unit_circle_list = np.array(unit_circle_list)
  print(unit_circle_list)

find_unit_circle()

# Using for loop to look for convergence
for i in range(Num_of_iterations):
  Num_of_iteration += 1
  time += time_step
  max_percentage_change_reached = 0
  T_new =T
#  print(f'At time {time} sec.')


  for x_index in range(1, Num_of_x_cell-1):
    for y_index in range(0, Num_of_y_cell-1):
      x, y = x_index * cell_size_x, y_index * cell_size_y
      r = np.sqrt(x**2+y**2)
      if (r>=1):
        # Don't update outside the unit circle
#        print(f'Skipping x={x}, y={y}, because r={r}>=1...')
        break
#      else:
#        print(f'Continuing with x={x}, y={y}, r={r}!')
      
      dT_dx = (T[x_index, y_index]-T[x_index-1, y_index])/cell_size_x
      dT_dx_next = (T[x_index+1, y_index]-T[x_index, y_index])/cell_size_x
      d2T_dx2 = (dT_dx_next - dT_dx)/cell_size_x
      
      dT_dy = (T[x_index, y_index] - T[x_index, (y_index-1)%Num_of_y_cell])/cell_size_y
      dT_dy_next = (T[x_index, y_index+1] - T[x_index, y_index])/cell_size_y
      d2T_dy2 = (dT_dy_next - dT_dy)/cell_size_y
      
      if (y_index==0):
        d2T_dy2 = 0

      T_Lagrangian = d2T_dx2 + d2T_dy2
      dT_dt = alpha*T_Lagrangian + g_dot/pho_cp
      
      percentage_change = abs(time_step*(dT_dt)/T[x_index, y_index])
      if (percentage_change > max_percentage_change_reached):
        max_percentage_change_reached = percentage_change
      
#      print(f'At time={time}sec, dt={time_step}, x={x}, y={y/np.pi}PI,\ndT_dx={dT_dx}, dT_dx_next={dT_dx_next}, d2T_dx2={d2T_dx2}\ndT_dy={dT_dy}, dT_dy_next={dT_dy_next}, d2T_dy2={d2T_dy2}\ndT_dt={dT_dt}, T was {T[x_index, y_index]}\npercentage_change={percentage_change}')
      
      if (percentage_change >= MAX_percentage_change):
        # Iterating too fast
        time_step = time_step/(percentage_change/MAX_percentage_change + 1)**2
        print(f'new reduced time step = {time_step}')
      elif (percentage_change <= MIN_percentage_change):
        # Iterating too slowly
        if (time_step < MAX_time_step):
          time_step = time_step+10*abs(percentage_change/MIN_percentage_change)
          print(f'new increased time step = {time_step}')
#          T[x_index, y_index]+=time_step*(dT_dt)
#          print(f'T is now {T[x_index, y_index]}')
      T_new[x_index, y_index]+=time_step*(dT_dt)
#      print(f'T is now {T_new[x_index, y_index]}')
      if(time<0 or np.max(T[:,0])>4000):
        break
    if(time<0 or np.max(T[:,0])>4000):
      break
  T = T_new
  if(time<0 or np.max(T[:,0])>4000):
    break
  
  print(f'At time t={time} sec, maximum percentage change reached is {100*max_percentage_change_reached}%, current time step={time_step}, Tmax={np.max(T[:,0])}')
  
  
  line.set_ydata(T[:,origin_y])
  cmap_plot.set_array(T.ravel())
  ax.set_title(f"Temperature Distribution (Iteration {Num_of_iteration})")  # Update title
  ax.legend([line], [f"Numerical at iteration {Num_of_iteration}"])  # Update legend
  plt.draw()  # Redraw plot
  plt.pause(0.000001)  # Pause to show the update

plt.show()


