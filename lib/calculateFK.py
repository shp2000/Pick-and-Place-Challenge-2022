import numpy as np
from math import pi
class FK():

    def __init__(self):

        pass



    def forward(self, q):



        A_list = []
        jointPositions = np.zeros((8, 3))
        T0e = np.identity(4)

        # DH parameters:
        theta = [0, q[0], q[1], q[2], q[3] - pi, q[4], pi-q[5],q[6]-pi/4]
        alpha = [0, -pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, -pi / 2, 0]
        r = [0, 0, 0, 0.0825, 0.0825, 0, 0.088,0]
        d = [0.141, 0.192, 0, 0.316, 0, 0.384, 0, 0.210]


        # loop to calculate list of transformations

        for i in range(len(theta)):

            A = np.array([[np.cos(theta[i]), -1 * (np.sin(theta[i]) * np.cos(alpha[i])), np.sin(theta[i]) * np.sin(alpha[i]),

                           r[i] * np.cos(theta[i])],

                          [np.sin(theta[i]), np.cos(theta[i]) * np.cos(alpha[i]), -1 * (np.cos(theta[i]) * np.sin(alpha[i])),

                           r[i] * np.sin(theta[i])],

                          [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],

                          [0, 0, 0, 1]])



            A_list.append(A)



        T0e = A_list[0] @ A_list[1] @ A_list[2] @ A_list[3] @ A_list[4] @ A_list[5] @ A_list[6] @ A_list[7]  # finding H0e
        T0e = np.round(T0e,2)
        k = A_list[0]
        jointPositions[0, :] = A_list[0][:3, 3]

        # finding joint coordinates9

        for i in range(1, len(A_list)):
            dot = k @ A_list[i]
            jointPositions[i, :] = dot[:3, 3]
            k = dot

        jointPositions = np.round(jointPositions, 2)

        # re assigning intermediate frames to original positions
        joint3_position = np.array([[0], [0], [0.195], [1]])
        jointPositions[2, :] = np.transpose((A_list[0] @ A_list[1] @ A_list[2] @ joint3_position)[:-1, -1:])
        joint5_position = np.array([[0], [0], [0.125], [1]])
        jointPositions[4, :] = np.transpose((A_list[0] @ A_list[1] @ A_list[2] @ A_list[3] @ A_list[4] @ joint5_position)[:-1, -1:])
        joint6_position = np.array([[0], [0], [0.015], [1]])
        jointPositions[5, :] = np.transpose((A_list[0] @ A_list[1] @ A_list[2] @ A_list[3] @ A_list[4] @ A_list[5] @ joint6_position)[:-1, -1:])
        joint7_position = np.array([[0], [0], [0.051], [1]])
        jointPositions[6, :] = np.transpose((A_list[0] @ A_list[1] @ A_list[2] @ A_list[3] @ A_list[4] @ A_list[5] @ A_list[6] @ joint7_position)[:-1,-1:])

        return jointPositions, T0e