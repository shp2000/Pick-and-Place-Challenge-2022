import numpy as np


from lib.calcJacobian import calcJacobian


def IK_velocity(q_in, v_in, omega_in):

    J = calcJacobian(q_in)
    zeta = np.append(v_in, omega_in)
    k = np.isnan(zeta)
    nan_filtered_values = np.logical_not(k)
    J_new = J[nan_filtered_values]
    zeta_new = zeta[nan_filtered_values]

    q_dot = np.linalg.lstsq(J_new, zeta_new, rcond=None)[0]

    return q_dot