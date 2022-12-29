import sys
import numpy as np
from copy import deepcopy
from math import pi
import time

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

from lib.calculateFK import FK
from lib.calculateIK6 import IK
ik = IK()
fk = FK()

# Rotates the gribber to align with the block
def align_gripper_block_rotation(H_block):
    print("Aligning Gripper")
    R = H_block[0:3,0:3].T
    new_H_block = H_block
    i = 0
    while i < 3:
        if(np.absolute(R[i,2])>=0.95):
            new_H_block[:, 2] = np.array([0, 0, 1, 0])
            if (i != 1):
                new_H_block[:, 1] = H_block[:, 1]
            else:
                new_H_block[:, 1] = H_block[:, 0]
            new_H_block[:, 0][0:3] = np.cross(new_H_block[:, 1][0:3], new_H_block[:, 2][0:3])

            break
        i = i + 1
    return new_H_block

# Finds the closest block given a detector and current end effector position
def find_block(H_current, detector):
    # Detect some blocks...
    # pose is the list of detections of blocks in cameras frame
    distance_best_block = 1000
    name_best_block = ""
    H_best_block = []
    if (detector.get_detections() == []):
        print("No block was found")
        return "No block", np.identity(4)

    for (name, pose) in detector.get_detections():
         distance = np.linalg.norm(pose[:,3]-H_current[:,3])
         if (distance < distance_best_block):
             distance_best_block = distance
             name_best_block = name
             H_best_block = pose

    return name_best_block, H_best_block

# Tries to grab a dynamic block
def dynamic_grab(arm, detector, dynamic_position):
    print("Moving to dynamic position")
    arm.safe_move_to_position(dynamic_position)

    H_block = np.identity(4)
    found_block = False
    while(found_block == False):
        if (detector.get_detections() != []):
            found_block = True
            for (name, pose) in detector.get_detections():
                H_block = pose
                break

    [_, H_current] = fk.forward(dynamic_position)
    H_ee_camera = detector.get_H_ee_camera()
    H_block = align_gripper_block_rotation(H_block)
    H_new = H_current@H_ee_camera@H_block

    current_position = ik.inverse(H_new, dynamic_position)
    arm.safe_move_to_position(np.array([current_position[0]-pi/32, detect_position[1], current_position[2], current_position[3], current_position[4], current_position[5], current_position[6]]))
    arm.safe_move_to_position(current_position)

    arm.exec_gripper_cmd(0.048, 50)
    print("Successfully grabbed dynamic block!!")

    return current_position

def grab_block(arm, detector, detect_position, seed):
    print("Attempting to grab the block!")
    print("Moving to the fetch the block pose!")

    # Move to the detect position
    arm.safe_move_to_position(detect_position)
    current_position = detect_position
    [_, H_current] = fk.forward(current_position)

    H_ee_camera = detector.get_H_ee_camera()

    [name_best_block, H_best_block] = find_block(H_current, detector)
    H_best_block = align_gripper_block_rotation(H_best_block)

    H_new = H_current@H_ee_camera@H_best_block
    print("Computing Inverse Kinematics! Be Patient!")
    current_position = ik.inverse(H_new, seed)
    print("Block Joint angles Solution: ", current_position)
    # Rotate 1st, 3rd, 7th joint first
    arm.safe_move_to_position(np.array([current_position[0], detect_position[1], current_position[2], current_position[3], current_position[4], current_position[5], current_position[6]]))
    arm.safe_move_to_position(current_position)
    arm.exec_gripper_cmd(0.048, 50)
    print("Successfully grabbed static block!!")

    arm.safe_move_to_position(detect_position)

    return current_position

def place_block(arm, place_position, H_place, seed):
    print("Attempting to place the block!")

    arm.safe_move_to_position(place_position)

    target_position = ik.inverse(H_place, seed)
    print("Block Joint angles Solution: ", target_position)
    arm.safe_move_to_position(target_position)
    arm.exec_gripper_cmd(0.1, 50)
    arm.safe_move_to_position(place_position)

    print("Successfully Place Block")

    return target_position

if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    arm.exec_gripper_cmd(0.08, 50)
    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    neutral_position = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
        detect_position = np.array([pi/12, 0, 0, 0, 0, 0, 0]) + neutral_position
        place_position = np.array([-pi/12, 0, 0, 0, 0, 0, 0]) + neutral_position
        dynamic_position = np.array([-pi/2, 0*pi/64, 0, -2*pi/4, 0, 18*pi/32, pi/4])
        H_place = np.array([[0, -1, 0, 0.6], [-1, 0, 0, -0.145], [0, 0, -1, 0.23], [0, 0, 0, 1]])

        q_place_1 = np.array([-pi/12, 14*pi/128, 0, -76*pi/128, 0, 86*pi/128, pi/4])
        q_place_2 = np.array([-pi/12, 10.5*pi/128, 0, -76*pi/128, 0, 84.5*pi/128, pi/4])
        q_place_3 = np.array([-pi/12, 7*pi/128, 0, -76*pi/128, 0, 84*pi/128, pi/4])
        q_place_4 = np.array([-pi/12, 3.5*pi/128, 0, -76*pi/128, 0, 84*pi/128, pi/4])
        q_place_5 = np.array([-pi/12, 2.5*pi/128, 0, -72*pi/128, 0, 79*pi/128, pi/4])

        dynamic_position_1 = np.array([0.5-pi, 0, 0.52569093, -1.14401406, 0.8859193, 1.30507887, -1.30833])
        dynamic_position_2 = np.array([0.5-pi, 1.07043798, 0.52569093, -1.14401406, 0.8859193, 1.30507887, -1.30833])
        dynamic_position_3 = np.array([1-pi, 1.07043798, 0.52569093, -1.14401406,0.75, 1.30507887, -1.30833])
    else:
        print("**  RED TEAM  **")
        detect_position = np.array([-pi/12, 0, 0, 0, 0, 0, 0]) + neutral_position
        place_position = np.array([pi/12, 0, 0, 0, 0, 0, 0]) + neutral_position
        dynamic_position = np.array([pi/2, 0*pi/64, 0, -2*pi/4, 0, 18*pi/32, pi/4])
        H_place = np.array([[0, -1, 0, 0.6], [-1, 0, 0, 0.145], [0, 0, -1, 0.23], [0, 0, 0, 1]])

        q_place_1 = np.array([pi/12, 14*pi/128, 0, -76*pi/128, 0, 86*pi/128, pi/4])
        q_place_2 = np.array([pi/12, 10.5*pi/128, 0, -76*pi/128, 0, 84.5*pi/128, pi/4])
        q_place_3 = np.array([pi/12, 7*pi/128, 0, -76*pi/128, 0, 84*pi/128, pi/4])
        q_place_4 = np.array([pi/12, 3.5*pi/128, 0, -76*pi/128, 0, 84*pi/128, pi/4])
        q_place_5 = np.array([pi/12, 2.5*pi/128, 0, -72*pi/128, 0, 79*pi/128, pi/4])
        q_place_6 = np.array([pi/12, 5*pi/128, 0, -63*pi/128, 0, 70*pi/128, pi/4])

        dynamic_position_1 = np.array([0.75, 1.07043798, 0.52569093, -1.14401406, pi/2-0.35, 1.30507887+pi/4, -1.30833-0.05])
        dynamic_position_2 = np.array([0.75, 1.07043798, 0.52569093, -1.14401406, pi/2-0.35, 1.30507887+pi/4, -1.30833-0.05])
        dynamic_position_3 = np.array([1.2, 1.07043798, 0.52569093, -1.14401406,pi/2-0.35, 1.30507887+pi/4, -1.30833-0.05])
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    # STUDENT CODE HERE

    #arm.safe_move_to_position(np.array(q_place_6))

    #arm.exec_gripper_cmd(0.1, 50)
    #arm.safe_move_to_position(dynamic_position_1)
    #arm.safe_move_to_position(dynamic_position_2)
    #arm.safe_move_to_position(dynamic_position_3)
    #time.sleep(10)
    #arm.exec_gripper_cmd(0.048, 50)

    # Place Block
    #arm.safe_move_to_position(dynamic_position)
    #arm.safe_move_to_position(place_position)
    #arm.safe_move_to_position(np.array(q_place_6))
    #arm.exec_gripper_cmd(0.1, 50)
    #arm.safe_move_to_position(place_position)


    #arm.exec_gripper_cmd(0.1, 50)
    #arm.safe_move_to_position(dynamic_position_1)
    #arm.safe_move_to_position(dynamic_position_2)
    #arm.safe_move_to_position(dynamic_position_3)
    #time.sleep(10)
    #arm.exec_gripper_cmd(0.048, 50)

    # Place Block
    #arm.safe_move_to_position(dynamic_position)
    #arm.safe_move_to_position(place_position)
    #arm.safe_move_to_position(np.array(q_place_5))
    #arm.exec_gripper_cmd(0.1, 50)
    #arm.safe_move_to_position(place_position)

    ### BLOCK ONE ###
    print("Opening Gripper")
    arm.exec_gripper_cmd(0.1, 50)

    seed_grab = detect_position
    grab_block(arm, detector, detect_position, seed_grab)

    arm.safe_move_to_position(place_position)
    arm.safe_move_to_position(np.array(q_place_1))
    arm.exec_gripper_cmd(0.1, 50)
    arm.safe_move_to_position(place_position)

    ### BLOCK TWO ###

    # Grab Block
    grab_block(arm, detector, detect_position, seed_grab)

    # Place Block
    arm.safe_move_to_position(place_position)
    arm.safe_move_to_position(np.array(q_place_2))
    arm.exec_gripper_cmd(0.1, 50)
    arm.safe_move_to_position(place_position)


    ### BLOCK THREE ###

    # Grab Block
    grab_block(arm, detector, detect_position, seed_grab)

    # Place Block
    arm.safe_move_to_position(place_position)
    arm.safe_move_to_position(np.array(q_place_3))
    arm.exec_gripper_cmd(0.1, 50)
    arm.safe_move_to_position(place_position)

    ### BLOCK FOUR ###

    # Grab Block
    grab_block(arm, detector, detect_position, seed_grab)

    # Place Block
    arm.safe_move_to_position(place_position)
    arm.safe_move_to_position(np.array(q_place_4))
    arm.exec_gripper_cmd(0.1, 50)
    arm.safe_move_to_position(place_position)

    ### BLOCK FIVE ###

    # Grab Block
    arm.exec_gripper_cmd(0.1, 50)
    arm.safe_move_to_position(dynamic_position_1)
    arm.safe_move_to_position(dynamic_position_2)
    arm.safe_move_to_position(dynamic_position_3)
    time.sleep(10)
    arm.exec_gripper_cmd(0.048, 50)

    # Place Block
    arm.safe_move_to_position(dynamic_position)
    arm.safe_move_to_position(place_position)
    arm.safe_move_to_position(np.array(q_place_5))
    arm.exec_gripper_cmd(0.1, 50)
    arm.safe_move_to_position(place_position)

    ### BLOCK SIX ###

    # Grab Block
    arm.exec_gripper_cmd(0.1, 50)
    arm.safe_move_to_position(dynamic_position_1)
    arm.safe_move_to_position(dynamic_position_2)
    arm.safe_move_to_position(dynamic_position_3)
    time.sleep(10)
    arm.exec_gripper_cmd(0.048, 50)

    # Place Block
    arm.safe_move_to_position(dynamic_position)
    arm.safe_move_to_position(place_position)
    arm.safe_move_to_position(np.array(q_place_6))
    arm.exec_gripper_cmd(0.1, 50)
    arm.safe_move_to_position(place_position)

    # END STUDENT CODE