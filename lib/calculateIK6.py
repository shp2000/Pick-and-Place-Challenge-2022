import numpy as np
from math import pi, sin, cos
from scipy.linalg import null_space

from lib.calculateFK import FK
from lib.calcJacobian import calcJacobian
from lib.IK_velocity import IK_velocity
import time


class IK:
    lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    
    center = lower + (upper - lower) / 2
    fk = FK()

    def __init__(self, max_steps=200, min_step_size=1e-3):

        self.max_steps = max_steps
        self.min_step_size = min_step_size

    @staticmethod
    def displacement_and_axis(target, current):
        disp = target[0:3, -1] - current[0:3, -1]
        temp = current[0:3, 0:3]
        R = np.linalg.inv(temp) @ target[0:3, 0:3]
        Skew = (R - np.transpose(R)) / 2
        a = np.array([Skew[2, 1], Skew[0, 2], Skew[1, 0]])
        axis = current[0:3, 0:3] @ a

        return disp, axis

    @staticmethod
    def distance_and_angle(G, H):
        dist_temp = G[0:3, -1] - H[0:3, -1]
        distance = np.linalg.norm(dist_temp)
        G_sliced = G[0:3, 0:3]
        H_sliced = H[0:3, 0:3]
        R = np.linalg.inv(G_sliced) @ H_sliced

        magnitude = (np.trace(R) - 1) / 2
        if magnitude > 1:
            magnitude = 1
        elif magnitude < -1:
            magnitude = -1
        angle = np.arccos(magnitude)

        return distance, angle

    @staticmethod
    def end_effector_task(q, target):
        joints, current = FK().forward(q)
        v, omega = IK().displacement_and_axis(target, current)
        dq = IK_velocity(q, v, omega)

        return dq

    @staticmethod
    def joint_centering_task(q, rate=5e-1):
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset

        return dq

    def inverse(self, target, seed):
        s_time = time.time()
        q = seed
        count = 0

        while True:
            dq_ik = self.end_effector_task(q, target)

            # Cost func2 Center Joints
            dq_center = self.joint_centering_task(q)
            J = calcJacobian(q)
            n = null_space(J).flatten()
            project_dq_center = np.dot(dq_center, n) * (n / np.square(np.linalg.norm(n)))
            dq = 0.5 * (dq_ik + 2 * project_dq_center)
            count += 1
            if count > self.max_steps or np.linalg.norm(dq) <= self.min_step_size:
                break
            q = q + dq
        e_time = time.time()
        print(e_time-s_time)

        return q

    
if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=5)

    ik = IK()

    # matches figure in the handout
    seed = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])

    target = np.array([
        [0, -1, 0, 0.3],
        [-1, 0, 0, 0],
        [0, 0, -1, .5],
        [0, 0, 0, 1],
    ])

    q = ik.inverse(target, seed)
    
    print("Solution: ", q)