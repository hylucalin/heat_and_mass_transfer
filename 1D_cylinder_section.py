import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
MAX = 1
MIN = 0
alpha = 10.09 * 1E-5    # m2/s
pho_cp = 8960 * 410     # kg/m3 * J/kgK = J/m3K
g_dot = 0*1E6           # W/m3

# Recrtangular window setup
Window_Width = 10
Window_Height = 7

# Simulation boundaries
## Note: Object should be bounded in r [0, 5]
r_lim = [0, 5]
theta_lim = [0, 2*np.pi]

# Simulation parameters
cell_size_r = 0.1
cell_size_theta = np.pi/24
time_step = 1
sim_time = 15000
Num_of_iterations = int(sim_time/time_step)
MAX_percentage_change = 0.15
MIN_percentage_change = 0.01
MAX_time_step = 50
Num_of_iteration = 0
time = 0

# Temperature grid initialisation
Num_of_r_cell = int((r_lim[MAX] - r_lim[MIN])/cell_size_r)
Num_of_theta_cell = int((theta_lim[MAX] - theta_lim[MIN])/cell_size_theta)

print(f'Num_of_r_cell = {Num_of_r_cell}, Num_of_theta_cell = {Num_of_theta_cell}')

rs = np.linspace(r_lim[MIN], r_lim[MAX], Num_of_r_cell)
thetas = np.linspace(theta_lim[MIN], theta_lim[MAX], Num_of_theta_cell)
thetas, rs = np.meshgrid(thetas, rs)

# Initial condition in K
T = np.full_like(thetas, fill_value=298)
T0 = 298
T = (T0-400)/(r_lim[MAX]-r_lim[MIN])*rs + 400

print(f'T = {T}')

# Boundary Conditions, the edge is held at constant temperature
#T0 = 298
## Constant outer temperature of T0
T[-1,:] = T0

## Constant inner temperature of 400K
T[0,:] = 400

# Theoretical solution
rs_theoretical = np.linspace(r_lim[MIN], r_lim[MAX], Num_of_r_cell)
Ts_theoretical = ((T0-400)/np.log((r_lim[MAX]-cell_size_r/2)/(r_lim[MIN]+cell_size_r/2))) * np.log(rs_theoretical+cell_size_r/2) + (400*np.log(r_lim[MAX]-cell_size_r/2)-T0*np.log(r_lim[MIN]+cell_size_r/2))/np.log((r_lim[MAX]-cell_size_r/2)/(r_lim[MIN]+cell_size_r/2))

fig, ax = plt.subplots()
line, = ax.plot(rs[:,0], T[:,0], label=f"Numerical at iteration {Num_of_iteration}")
line_theoretical, = ax.plot(rs_theoretical, Ts_theoretical, 'r', label="Analytical")

# Add title and labels
ax.set_title(f"Temperature Distribution (Iteration {Num_of_iteration})")
ax.set_xlabel("Radius")
ax.set_ylabel("Temperature")

# Add legend
ax.legend()

# Using for loop to look for convergence
for i in range(Num_of_iterations):
  Num_of_iteration += 1
  max_percentage_change_reached = 0
  T_new =T
#  print(f'At time {time} sec.')
  for r_index in range(1, Num_of_r_cell):
    for theta_index in range(Num_of_theta_cell):
      r, theta = r_index * cell_size_r+0.5*cell_size_r, theta_index * cell_size_theta+cell_size_theta
      if (r_index < Num_of_r_cell-1):
        # Normal case, wouldn't need to be treated differently. It hasn't reached the boundary
        dT_dr = (T[r_index, theta_index]-T[r_index-1, theta_index])/cell_size_r
        dT_dr_next = (T[r_index+1, theta_index]-T[r_index, theta_index])/cell_size_r
        d2T_dr2 = (dT_dr_next - dT_dr)/cell_size_r

        dT_dtheta = (T[r_index, (theta_index+1)%Num_of_theta_cell] - T[r_index, theta_index])/cell_size_theta
        dT_dtheta_next = (T[r_index, (theta_index+2)%Num_of_theta_cell] - T[r_index, (theta_index+1)%Num_of_theta_cell])/cell_size_theta
        d2T_dtheta2 = (dT_dtheta_next - dT_dtheta)/cell_size_theta

        T_Lagrangian = (1/r) * dT_dr + d2T_dr2 + (1/r**2)*d2T_dtheta2
        dT_dt = alpha*T_Lagrangian + g_dot/pho_cp
        
        percentage_change = abs(time_step*(dT_dt)/T[r_index, theta_index])
        if (percentage_change > max_percentage_change_reached):
          max_percentage_change_reached = percentage_change
        
        print(f'At time={time}sec, dt={time_step}, r={r}, theta={theta/np.pi}PI,\ndT_dr={dT_dr}, dT_dr_next={dT_dr_next}, d2T_dr2={d2T_dr2}\ndT_dtheta={dT_dtheta}, dT_dtheta_next={dT_dtheta_next}, d2T_dtheta2={d2T_dtheta2}\ndT_dt={dT_dt}, T was {T[r_index, theta_index]}\npercentage_change={percentage_change}')
        
        if (percentage_change >= MAX_percentage_change):
          # Iterating too fast
          time_step = time_step/(percentage_change/MAX_percentage_change + 1)**2
          print(f'new reduced time step = {time_step}')
        elif (percentage_change <= MIN_percentage_change):
          # Iterating too slowly
          if (time_step < MAX_time_step):
            time_step = time_step+0.1*abs(percentage_change/MIN_percentage_change)
            print(f'new increased time step = {time_step}')
        else:
          time += time_step
#          T[r_index, theta_index]+=time_step*(dT_dt)
#          print(f'T is now {T[r_index, theta_index]}')
        T_new[r_index, theta_index]+=time_step*(dT_dt)
        print(f'T is now {T_new[r_index, theta_index]}')
      if(time<0 or np.max(T[:,0])>4000):
        break
    if(time<0 or np.max(T[:,0])>4000):
      break
  T = T_new
  if(time<0 or np.max(T[:,0])>4000):
    break
  
  # Averaging in the theta direction to get rid of the unrealistic stripes
  for r_index in range(1, Num_of_r_cell-1):
    for theta_index in range(Num_of_theta_cell):
      r, theta = r_index * cell_size_r+0.5*cell_size_r, theta_index * cell_size_theta+cell_size_theta
      T_new[r_index, theta_index] = (10*T[r_index, theta_index]+8*T[r_index, (theta_index+1)%Num_of_theta_cell]+8*T[r_index, (theta_index-1)%Num_of_theta_cell]+5*T[r_index, (theta_index+2)%Num_of_theta_cell]+5*T[r_index, (theta_index-2)%Num_of_theta_cell]+3*T[r_index, (theta_index+3)%Num_of_theta_cell]+3*T[r_index, (theta_index-3)%Num_of_theta_cell]+2*T[r_index, (theta_index+4)%Num_of_theta_cell]+2*T[r_index, (theta_index-4)%Num_of_theta_cell]+T[r_index, (theta_index+5)%Num_of_theta_cell]+T[r_index, (theta_index-5)%Num_of_theta_cell])/(10+2*(8+5+3+2+1))
  T = T_new
  print(f'At time t={time} sec, maximum percentage change reached is {100*max_percentage_change_reached}%, current time step={time_step}, Tmax={np.max(T[:,0])}')
  line.set_ydata(T[:,0])
  ax.set_title(f"Temperature Distribution (Iteration {Num_of_iteration})")  # Update title
  ax.legend([line, line_theoretical], [f"Numerical at iteration {Num_of_iteration}", "Analytical"])  # Update legend
  plt.draw()  # Redraw plot
  plt.pause(0.000001)  # Pause to show the update

plt.show()

# Polar plot generation
fig, ax = plt.subplots(figsize=(Window_Width, Window_Height), subplot_kw={'projection':'polar'})
#colormap = ax.pcolormesh(thetas, rs, T, shading='auto', cmap='viridis')
colormap = ax.pcolormesh(thetas, rs, T, shading='gouraud', cmap='viridis')#inferno

# Add color bar
plt.colorbar(colormap, ax=ax)


# Update function of T for animation
#def update(frame):
#    # frame is sequential and starts at 0
#    print(f'frame = {frame}')
##    time = time+time_step
#    
#    for r_index in range(2, Num_of_r_cell):
#      for theta_index in range(Num_of_theta_cell):
#        r, theta = r_index * cell_size_r+0.5*cell_size_r, theta_index * cell_size_theta+cell_size_theta
#        if (r_index < Num_of_r_cell-2):
#          # Normal case, wouldn't need to be treated differently. It hasn't reached the boundary
#          dT_dr = (T[r_index, theta_index]-T[r_index-1, theta_index])/cell_size_r
#          dT_dr_next = (T[r_index+1, theta_index]-T[r_index, theta_index])/cell_size_r
#          d2T_dr2 = (dT_dr_next - dT_dr)/cell_size_r
#          print(f'At time={frame*time_step}sec, dT_dr={dT_dr}, dT_dr_next={dT_dr_next}, d2T_dr2={d2T_dr2}')
#          
#          dT_dtheta = (T[r_index, (theta_index+1)%Num_of_theta_cell] - T[r_index, theta_index])/cell_size_theta
#          dT_dtheta_next = (T[r_index, (theta_index+2)%Num_of_theta_cell] - T[r_index, (theta_index+1)%Num_of_theta_cell])/cell_size_theta
#          d2T_dtheta2 = (dT_dtheta_next - dT_dtheta)/cell_size_theta
#          print(f'At time={frame*time_step}sec, dT_dtheta={dT_dtheta}, dT_dtheta_next={dT_dtheta_next}, d2T_dtheta2={d2T_dtheta2}')
#    
#          T_Lagrangian = (1/r) * dT_dr + d2T_dr2 + (1/r**2)*d2T_dtheta2
#          dT_dt = alpha*T_Lagrangian + g_dot/pho_cp
#          print(f'At time={frame*time_step}sec, dT_dt={dT_dt}, T was {T[r_index, theta_index]}')
#          T[r_index, theta_index]+=time_step*(dT_dt)
#          print(f'T is now {T[r_index, theta_index]}')
#    print(T)
#    
#
#    # Update the data in the plot
#    colormap.set_array(T.ravel())  # Update the color values
#
#    return colormap,  # Return the updated plot object
#
## Create the animation
#ani = FuncAnimation(fig, update, frames=int(1/time_step), interval=1, blit=True)

# Show the animation
plt.show()


