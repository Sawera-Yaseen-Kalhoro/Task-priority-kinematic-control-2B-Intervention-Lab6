import numpy as np
import matplotlib.pyplot as plt

# Load the position evolutions from all 3 methods
method1_base = np.load("method1_base.npy")
method1_ee = np.load("method1_ee.npy")

method2_base = np.load("method2_base.npy")
method2_ee = np.load("method2_ee.npy")

method3_base = np.load("method3_base.npy")
method3_ee = np.load("method3_ee.npy")

# Plot the position evolutions
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Position Evolutions')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.grid()

ax.plot(method1_base[0],method1_base[1], 'r-', linewidth=0.8, label= "Method 1 - Base")
ax.plot(method1_ee[0],method1_ee[1], 'r--', linewidth=0.8, label="Method 1 - EE")

ax.plot(method2_base[0],method2_base[1], 'b-', linewidth=0.8, label= "Method 2 - Base")
ax.plot(method2_ee[0],method2_ee[1], 'b--', linewidth=0.8, label="Method 2 - EE")

ax.plot(method3_base[0],method3_base[1], 'g-', linewidth=0.8, label= "Method 3 - Base")
ax.plot(method3_ee[0],method3_ee[1], 'g--', linewidth=0.8, label="Method 3 - EE")

ax.legend()

plt.show()
