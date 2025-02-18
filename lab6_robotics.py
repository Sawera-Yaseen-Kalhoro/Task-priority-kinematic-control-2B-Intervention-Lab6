from lab5_robotics import *

def translation(x, y):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def rotation_z(angle):
    return np.array([[np.cos(angle),-np.sin(angle),0,0],
                     [np.sin(angle),np.cos(angle),0,0],
                     [0,0,1,0],
                     [0,0,0,1]])

class MobileManipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.revoluteExt = [True,False] + self.revolute # List of joint types extended with base joints
        self.r = 0            # Distance from robot centre to manipulator base
        self.dof = len(self.revoluteExt) # Number of DOF of the system
        self.q = np.zeros((len(self.revolute),1)) # Vector of joint positions (manipulator)
        self.eta = np.zeros((3,1)) # Vector of base pose (position & orientation)
        self.update(np.zeros((self.dof,1)), 0.0) # Initialise robot state

    '''
        Method that updates the state of the robot.

        Arguments:
        dQ (Numpy array): a column vector of quasi velocities
        dt (double): sampling time
    '''
    
    def update(self, dQ, dt, method=1):
    ## dQ = np.array([[angular vel of mobile base],
    #       [linear vel of mobile base],
    #       [manipulator joint velocities]])
    # Note: the order for the mobile base follow the definition self.revoluteExt
    
        # Update manipulator
        self.q += dQ[2:, 0].reshape(-1,1) * dt

        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]

        # Update mobile base pose
        if method == 1:
            # method 1: move forward, then rotate
            self.eta[0,0] += dQ[1,0] * np.cos(self.eta[2,0]) * dt
            self.eta[1,0] += dQ[1,0] * np.sin(self.eta[2,0]) * dt
            self.eta[2,0] += dQ[0,0] * dt
        elif method == 2:
            # method 2: rotate, then move forward
            self.eta[2,0] += dQ[0,0] * dt
            self.eta[0,0] += dQ[1,0] * np.cos(self.eta[2,0]) * dt
            self.eta[1,0] += dQ[1,0] * np.sin(self.eta[2,0]) * dt
        elif method == 3:
            # method 3: move forward and rotate at the same time
            v = dQ[1,0]
            w = dQ[0,0]

            dtheta = w * dt # change of orientation

            R = v/w # instantaneous cornering radius
            theta_1 = self.eta[2,0]
            dx = R * (sin(theta_1 + dtheta) - sin(theta_1)) # displacement in x
            dy = R * (cos(theta_1) - cos(theta_1 + dtheta)) # displacement in y

            self.eta[0,0] += dx
            self.eta[1,0] += dy
            self.eta[2,0] += dtheta
        
        # Base kinematics
        # Transformation of the mobile base
        Tb = translation(self.eta[0, 0], self.eta[1, 0]) @ rotation_z(self.eta[2, 0])


        # Modify the theta of the base joint, to account for an additional Z rotation
        self.theta[0] -= np.pi/2

        # Combined system kinematics (DH parameters extended with base DOF)
        dExt = np.concatenate([np.array([0, self.r]), self.d])
        thetaExt = np.concatenate([np.array([np.pi/2, 0]), self.theta])
        aExt = np.concatenate([np.array([0, 0]), self.a])
        alphaExt = np.concatenate([np.array([np.pi/2, -np.pi/2]), self.alpha])

        self.T = kinematics(dExt, thetaExt, aExt, alphaExt, Tb)

        return self.T


    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revoluteExt)

    '''
        Method that returns the end-effector transformation.
    '''
    def getEETransform(self):
        return self.T[-1]

    '''
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    '''
    def getJointPos(self, joint):
        return self.q[joint-2]


    def getBasePose(self):
        return self.eta

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof

    ###
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revoluteExt, link)

    def getLinkTransform(self, link):
        return self.T[link]

