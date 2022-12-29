import numpy as np
from math import pi
from lib.calculateFK import FK
import time

def calcJacobian(q_in):


    s_time = time.time()
    J = np.zeros((6, 7))


    A_list = []

    # DH parameters:

    alpha = [0, -pi / 2, pi / 2, pi / 2, pi / 2, pi / 2, -pi / 2, 0]
    d = [0.141, 0.192, 0, 0.316, 0, 0.384, 0, 0.210]
    r = [0, 0, 0, 0.0825, 0.0825, 0, 0.088, 0]
    q = q_in
    theta = [0, q[0], q[1], q[2], q[3] - pi, q[4], pi-q[5],q[6]-pi/4]

    # loop to calculate list of transformations
    for i in range(len(theta)):
      A = np.array([[np.cos(theta[i]), -1 * (np.sin(theta[i]) * np.cos(alpha[i])), np.sin(theta[i]) * np.sin(alpha[i]),
                           r[i] * np.cos(theta[i])],
                          [np.sin(theta[i]), np.cos(theta[i]) * np.cos(alpha[i]), -1 * (np.cos(theta[i]) * np.sin(alpha[i])),
                           r[i] * np.sin(theta[i])],
                          [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
                          [0, 0, 0, 1]])

      A_list.append(A)


    joint_t1 = A_list[0]
    joint_t2 = joint_t1 @ A_list[1]
    joint_t3 = joint_t2 @ A_list[2]
    joint_t4 = joint_t3 @ A_list[3]
    joint_t5 = joint_t4 @ A_list[4]
    joint_t6 = joint_t5 @ A_list[5]
    joint_t7 = joint_t6 @ A_list[6]
    joint_te = joint_t7 @ A_list[7]

    joint_transformations =         [joint_t1,joint_t2,joint_t3,joint_t4,joint_t5,joint_t6,joint_t7,joint_te]

    RM = []

    for data in joint_transformations:
      rotational_matrix = data[:-1, :-1]
      RM.append(rotational_matrix)

    RM.insert(0,np.eye(3))

    R = []

    for data in RM:
      R.append(data[:,2].reshape(3,1))

    TM_1 = joint_t1[:-1, -1:]
    TM_2 = joint_t2[:-1, -1:]
    TM_4 = joint_t4[:-1, -1:]
    TM_e = joint_te[:-1, -1:]


    joint_pos_3 = np.array([[0], [0], [0.195], [1]])
    thirdjoint_pos = joint_t3 @joint_pos_3
    TM_3 = thirdjoint_pos[:-1, -1:]


    joint_pos_5 = np.array([[0], [0], [0.125], [1]])
    fifthjoint_pos = joint_t5 @ joint_pos_5
    TM_5 = fifthjoint_pos[:-1, -1:]


    joint_pos_6 = np.array([[0], [0], [0.015], [1]])
    sixthjoint_pos = joint_t6 @ joint_pos_6
    TM_6 = sixthjoint_pos[:-1, -1:]


    joint_pos_7 = np.array([[0], [0], [0.051], [1]])
    seventhjoint_pos = joint_t7 @ joint_pos_7
    TM_7 = seventhjoint_pos[:-1, -1:]

    TM = [0,TM_1,TM_2,TM_3,TM_4,TM_5,TM_6,TM_7,TM_e]
    O0 = np.array([0,0,0]).reshape(-1,1)

    Linear_Jacobian = []

    for i in range(1,8):
      Linear_Jacobian.append(np.cross(R[i].T, (TM_e - TM[i] ).T).T)

    Angular_Jacobian = []

    for i in range(1,8):
      Angular_Jacobian.append(R[i])



    Linear_Jacobian_Matrix = np.hstack([ Linear_Jacobian[0], Linear_Jacobian[1], Linear_Jacobian[2], Linear_Jacobian[3], Linear_Jacobian[4], -1* Linear_Jacobian[5],Linear_Jacobian[6]])

    Angular_Jacobian_Matrix =  np.hstack([ Angular_Jacobian[0], Angular_Jacobian[1], Angular_Jacobian[2], Angular_Jacobian[3], Angular_Jacobian[4],-1* Angular_Jacobian[5], Angular_Jacobian[6]])


    J = np.vstack([Linear_Jacobian_Matrix,         Angular_Jacobian_Matrix])

    e_time = time.time()
    #print(e_time-s_time)

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
