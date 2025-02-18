from lab6_robotics import *  # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans
import numpy as np

# Robot model
d = np.zeros(3)  # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.1])  # rotation around Z-axis
alpha = np.zeros(3)  # rotation around X-axis
a = np.array([0.75, 0.5, 0.3])  # displacement along X-axis
revolute = [True, True, True]  # flags specifying the type of joints
robot = MobileManipulator(d, theta, a, alpha, revolute)

# Task definition
tasks = [
        Configuration2D("End-effector configuration", np.array([1.0, 0.0, 0.0]).reshape(3, 1), 5)
        ]

# Simulation params
dt = 0.1

# Drawing preparation for robot simulation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
line, = ax.plot([], [], 'o-', lw=2)  # Robot structure
path, = ax.plot([], [], 'c-', lw=1)  # End-effector path
point, = ax.plot([], [], 'rx')  # Target
PPx = []
PPy = []

# initialize a memory to store the error evolution and velocity output
error_evol = [[], []]  # initialize 2 lists to store position and orientation

# Weighting matrix
# W = np.diag([4.0, 6.0, 3.0, 2.0, 1.0])

# Define a list of desired configuration
desired_config = [np.array([0.0, 0.0, 0.0]).reshape(3, 1),
                  np.array([0.0, 0.0, 0.0]).reshape(3, 1),
                  np.array([0.0, 0.0, 0.0]).reshape(3, 1), # the animation somehow skips the first 3
                  np.array([1.0, 0.0, 0.0]).reshape(3, 1),
                  np.array([-1.0, 1.0, 0.0]).reshape(3, 1),
                  np.array([0.8, -1.0, 0.0]).reshape(3, 1),
                  np.array([1.5, 1.2, 0.0]).reshape(3, 1),
                  np.array([-0.2, 1.0, 0.0]).reshape(3, 1)]
desired_config_counter = 0

# Initialize a memory to store the evolution of mobile base position and EE position
base_pos_evol = [[],[]]
ee_pos_evol = [[],[]]

# Simulation initialization
def init():
    global tasks
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])

    # initialize desired EE position and orientation from the list
    global desired_config_counter
    tasks[0].setDesired(desired_config[desired_config_counter])
    desired_config_counter += 1

    return line, path, point

# Simulation loop
def simulate(t):
    global tasks 
    global robot
    global PPx
    global PPy
    global error_evol
    global base_pos_evol,ee_pos_evol

    # print("Simulate.")
    # Recursive Task-Priority algorithm

    # Initialize null-space projector
    Pi_1 = np.eye(robot.getDOF())

    # Initialize output vector (joint velocity)
    dqi_1 = np.zeros((robot.getDOF(), 1))

    # Loop over tasks
    for i in range(len(tasks)):
        task = tasks[i]

        # Update task state
        task.update(robot)
        Ji = task.getJacobian()
        erri = task.getError()

        # store the error evolution
        error_evol[0].append(np.linalg.norm(erri[:2]))
        error_evol[1].append(erri[2, 0])

        # Skip if the task is not active
        if not task.isActive(robot):
            continue

        # Get feedforward velocity and gain matrix
        feedforward_velocity = task.getFeedforwardVelocity()
        K = task.getGainMatrix()

        xdoti = feedforward_velocity + K @ erri

        # Compute augmented Jacobian
        Ji_bar = Ji @ Pi_1

        # Compute task velocity with feedforward term and K matrix
        dqi = DLS(Ji_bar,0.1) @ (xdoti - Ji @ dqi_1)
        # dqi = WeightedDLS(Ji_bar, 0.1, W) @ (xdoti - Ji @ dqi_1)
        dq = dqi_1 + dqi
        
        # Update null-space projector
        Pi = Pi_1 - np.linalg.pinv(Ji_bar) @ Ji_bar

        # Store the current P and dq for the next iteration
        Pi_1 = Pi
        dqi_1 = dqi


    # Update robot
    robot.update(dq, dt, 3)

    # Store mobile base and EE position
    base_pos = robot.getBasePose()
    base_pos_evol[0].append(base_pos[0,0])
    base_pos_evol[1].append(base_pos[1,0])

    ee_pos = robot.getEETransform()[:2,-1]
    ee_pos_evol[0].append(ee_pos[0])
    ee_pos_evol[1].append(ee_pos[1])

    # Update drawing for robot simulation
    PP = robot.drawing()
    line.set_data(PP[0, :], PP[1, :])
    PPx.append(PP[0, -1])
    PPy.append(PP[1, -1])
    path.set_data(PPx, PPy)
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])

    # -- Mobile base
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2, 0]) + trans.Affine2D().translate(eta[0, 0], eta[1, 0]) + ax.transData)

    return line, veh, path, point


# Function to generate all result plots
def result_plot(error_evol,dt):
    # define simulation time (after repeats)
    tfinal = len(error_evol[0]) * dt
    tt = np.arange(0, tfinal, dt)

    ## Plotting error evolution
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, xlim=(0, tfinal))

    # Plotting position error evolution
    ax1.plot(tt, error_evol[0], label='Position Error')

    # Plotting orientation error evolution
    ax1.plot(tt, error_evol[1], label='Orientation Error')

    ax1.set_title('End-effector Configuration Task Error Evolution')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Error')
    ax1.legend()
    ax1.grid()

    plt.show()



# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt),
                                interval=10, blit=True, init_func=init, repeat=True)

plt.show()

# Show the plot of error evolution
result_plot(error_evol,dt)

# Save the error evolution for future plotting
np.save("method3_base",base_pos_evol)
np.save("method3_ee",ee_pos_evol)
