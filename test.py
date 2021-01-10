import numpy as np
import quaternion as quat

def euler2dcm(euler):
    phi   = euler[2] # Z axis Yaw
    theta = euler[1] # Y axis Pitch
    psi   = euler[0] # X axis Roll
    rotx = np.array([[1, 0, 0],
                    [0, np.cos(psi), np.sin(psi)],
                    [0, -np.sin(psi), np.cos(psi)]])
    roty = np.array([[np.cos(theta), 0, -np.sin(theta)],
                    [0, 1, 0],
                    [np.sin(theta), 0, np.cos(theta)]])
    rotz = np.array([[np.cos(phi), np.sin(phi), 0],
                    [-np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    dcm = rotx @ roty @ rotz
    return dcm

def dcm2quaternion(dcm):
    # calculate quaternion from DCM
    q = np.zeros(4, dtype=float)
    
    # q = [q1,q2,q3,q4] version
    # q[3] = 0.5 * np.sqrt(1 + dcm[0,0] + dcm[1,1] + dcm[2,2])
    # q[0] = 0.25 * (dcm[1,2] - dcm[2,1]) / q[3]
    # q[1] = 0.25 * (dcm[2,0] - dcm[0,2]) / q[3]
    # q[2] = 0.25 * (dcm[0,1] - dcm[1,0]) / q[3]

    #q = [q0,q1,q2,q3] version
    C0 = np.trace(dcm)
    C = [C0,dcm[0,0],dcm[1,1],dcm[2,2]]
    Cj = max(C)
    j = C.index(Cj)
    q[j] = 0.5 * np.sqrt(1+2*Cj-C0)

    if j == 0:
        q[1] =(dcm[1,2] - dcm[2,1]) / (4*q[0])
        q[2] =(dcm[2,0] - dcm[0,2]) / (4*q[0])
        q[3] =(dcm[0,1] - dcm[1,0]) / (4*q[0])
    elif j==1:# %ε(1)が最大の場合
        q[0]=(dcm[1,2]-dcm[2,1])/(4*q[1])
        q[3]=(dcm[2,0]+dcm[0,2])/(4*q[1])
        q[2]=(dcm[0,1]+dcm[1,0])/(4*q[1])
    elif j==2: # %ε(2)が最大の場合
        q[0]=(dcm[2,0]-dcm[0,2])/(4*q[2])
        q[3]=(dcm[1,2]+dcm[2,1])/(4*q[2])
        q[1]=(dcm[0,1]+dcm[1,0])/(4*q[2])
    elif j==3: # %ε(3)が最大の場合
        q[0]=(dcm[1,2]-dcm[2,1])/(4*q[3])
        q[2]=(dcm[0,1]+dcm[2,0])/(4*q[3])
        q[1]=(dcm[2,0]+dcm[0,2])/(4*q[3])
    return q


def quaternion2dcm(q):
    dcm = np.zeros((3,3), dtype=float)
    dcm[0,0] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]
    dcm[0,1] = 2 * (q[0]*q[1] + q[2]*q[3])
    dcm[0,2] = 2 * (q[0]*q[2] - q[1]*q[3])
    dcm[1,0] = 2 * (q[0]*q[1] - q[2]*q[3])
    dcm[1,1] = - q[0]*q[0] + q[1]*q[1] - q[2]*q[2] + q[3]*q[3]
    dcm[1,2] = 2 * (q[1]*q[2] + q[0]*q[3])
    dcm[2,0] = 2 * (q[0]*q[2] + q[1]*q[3])
    dcm[2,1] = 2 * (q[1]*q[2] - q[0]*q[3])
    dcm[2,2] = - q[0]*q[0] - q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    return dcm

def dcm2euler(dcm):
    # calculate 321 Euler angles [rad] from DCM
    sin_theta = - dcm[0,2]
    if sin_theta == 1 or sin_theta == -1:
        theta = np.arcsin(sin_theta)
        psi = 0
        sin_phi = -dcm(2,1)
        phi = np.arcsin(sin_phi)
    else:
        theta = np.arcsin(sin_theta)
        phi = np.arctan2(dcm[1,2], dcm[2,2])
        psi = np.arctan2(dcm[0,1], dcm[0,0])
        
    euler = np.array([psi, theta, phi])
    return euler

# 初期状態
startEuler = np.deg2rad(np.array([-44.34233084,41.69147074,-22.07080979]))
dcm = euler2dcm(startEuler)
startQuate = dcm2quaternion(dcm)
# startQuate = quat.from_euler_angles(startEuler, beta=None, gamma=None)
# DCM = quat.as_rotation_matrix(startQuate)
print(startQuate)
print(dcm)

quate = np.array([-0.0338,0.391,-0.283,0.875])
tmp = quaternion2dcm(quate)
euler = dcm2euler(tmp)
print(tmp)
print(np.rad2deg(euler))

from gym import spaces
max_torque = 0.5
action_bound = np.array([max_torque, max_torque, max_torque],dtype=np.float32)
a = spaces.Box(-action_bound, action_bound)
# print(a)

# a = 3
# if a > 1:
#     print('condition 1')
# elif a > 2:
#     print("condition 2")
# else:
#     pass