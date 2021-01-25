import os

import math
import matplotlib.pyplot as plt
import torch
from gym import wrappers 
import numpy as np


from network import DDPGAgent
from utils import *


def train(batch_size=128, critic_lr=1e-3, actor_lr=1e-4, max_episodes=1000, max_steps=300, gamma=0.99, tau=1e-3,
          buffer_maxlen=100000):
    # simulation of the agent solving the spacecraft attitude control problem
    env = make("SatelliteContinuous")

    max_episodes = max_episodes
    max_steps = max_steps
    batch_size = batch_size

    gamma = gamma
    tau = tau
    buffer_maxlen = buffer_maxlen
    critic_lr = critic_lr
    actor_lr = actor_lr

    #agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr, True, max_episodes * max_steps)
    curr_dir = os.path.abspath(os.getcwd())
    agent = torch.load(curr_dir + "/models/spacecraft_control_ddpg.pkl")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)

    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
	# plt.show()
    plt.savefig(curr_dir + "/results/plot_reward_hist.png")

    curr_dir = os.path.abspath(os.getcwd())
    if not os.path.isdir("models"):
        os.mkdir("models")
    torch.save(agent, curr_dir + "/models/spacecraft_control_ddpg.pkl")


def evaluate():
    # simulation of the agent solving the cartpole swing-up problem
    env = make("SatelliteContinuous")
    # uncomment for recording a video of simulation
    # env = wrappers.Monitor(env, './video', force=True)

    curr_dir = os.path.abspath(os.getcwd())

    agent = torch.load(curr_dir + "/models/spacecraft_control_ddpg.pkl")
    agent.train = False

    state = env.reset()
    r = 0
    qe = np.empty((0,4))
    q = np.empty((0,4))
    w = np.empty((0,3))
    actions = np.empty((0,3))

    dt = 0.1
    simutime = 30
    simulation_iterations = int(simutime/dt) -1 # dt is 0.01

    for i in range(1, simulation_iterations):
        action = agent.get_action(state)
        # action = np.squeeze(action)
        next_error_state, reward, done, next_state, _ = env.step(action)
        # env.render()
        q=np.append(q,next_state[0:4].reshape(1,-1),axis=0)
        qe=np.append(qe,next_error_state[0:4].reshape(1,-1),axis=0)
        w=np.append(w,next_error_state[8:11].reshape(1,-1),axis=0)
        r += reward
        actions = np.append(actions, action.reshape(1,-1),axis=0)

        state = next_error_state

    env.close()
    #-------------------------------結果のプロット----------------------------------
    #show the total reward
    print("Total Reward is : " + str(r))
    # データの形の整理
    q = q.reshape([-1,4])
    qe = qe.reshape([-1,4])
    w = w.reshape([-1,3])
    # angle = [e for i in]

    # plot the angle and action curve
    #-------------------plot settings------------------------------
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 15 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 15 # 軸だけ変更されます 
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams['axes.grid'] = True # make grid
    #--------------------------------------------------------------  
    curr_dir = os.path.abspath(os.getcwd())
    if not os.path.isdir("results"):
        os.mkdir("results")
    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(np.arange(simulation_iterations-1)*dt, q[:,0],label =r"$q_{0}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, q[:,1],label =r"$q_{1}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, q[:,2],label =r"$q_{2}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, q[:,3],label =r"$q_{3}$")
    plt.title('Quaternion')
    plt.ylabel('quaternion value')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    plt.savefig(curr_dir + "/results/plot_quaternion.png")

    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(np.arange(simulation_iterations-1)*dt, qe[:,0],label =r"$q_{0}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, qe[:,1],label =r"$q_{1}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, qe[:,2],label =r"$q_{2}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, qe[:,3],label =r"$q_{3}$")
    plt.title('Quaternion Error')
    plt.ylabel('quaternion value')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    plt.savefig(curr_dir + "/results/plot_error_quaternion.png")

    angle = np.array([np.rad2deg(env.dcm2euler(env.quaternion2dcm(q[i,:]))).tolist() for i in range(simulation_iterations-1)])
    angle = angle.reshape([-1,3])
    plt.figure(figsize=(yoko,tate),dpi=100)
    plt.plot(np.arange(simulation_iterations-1)*dt, angle[:,0],label = r"$\phi$")
    plt.plot(np.arange(simulation_iterations-1)*dt, angle[:,1],label = r"$\theta$")
    plt.plot(np.arange(simulation_iterations-1)*dt, angle[:,2],label = r"$\psi$")
    # plt.title('Action')
    plt.ylabel('angle [deg]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    plt.savefig(curr_dir + "/results/plot_angle.png")


    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(np.arange(simulation_iterations-1)*dt, w[:,0],label =r"$\omega_{x}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, w[:,1],label =r"$\omega_{y}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, w[:,2],label =r"$\omega_{z}$")
    plt.title('Angular velocity')
    plt.ylabel('angular velocity [rad/s]')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    plt.savefig(curr_dir + "/results/plot_ang_vel.png")

    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(np.arange(simulation_iterations-1)*dt, actions[:,0],label = r"$\tau_{x}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, actions[:,1],label = r"$\tau_{x}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, actions[:,2],label = r"$\tau_{x}$")
    plt.title('Action')
    plt.ylabel('Input torque [Nm]')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    plt.savefig(curr_dir + "/results/plot_torque.png")

    plt.show()
    # -------------------------結果プロット終わり--------------------------------
def env_test():

    # simulation of the agent solving the cartpole swing-up problem
    env = make("SatelliteContinuous")
    # uncomment for recording a video of simulation
    # env = wrappers.Monitor(env, './video', force=True)

    curr_dir = os.path.abspath(os.getcwd())
    env.reset()
    r = 0
    qe = np.empty((0,4))
    q = np.empty((0,4))
    w = np.empty((0,3))
    actions = np.empty((0,3))

    kp = 0.7
    kd = 1.9
    Kp = np.array([[0,kp,0,0],
                  [0,0,kp,0],
                  [0,0,0,kp]])
    Kd = np.array([[kd,0,0],
                  [0,kd,0],
                  [0,0,kd]])
    action = np.array([0,0,0]).reshape(1,3)
    actions = np.append(actions, action,axis=0)

    dt = 0.01
    simulation_iterations = int(50/0.01) -1 # dt is 0.01

    for i in range(1, simulation_iterations):
        action = np.squeeze(action)
        next_error_state, reward, done, next_state, _ = env.step(action)
        # env.render()
        # q=np.append(q,next_state[0].reshape(1,-1),axis=0)
        # qe=np.append(qe,next_error_state[0].reshape(1,-1),axis=0)
        # w=np.append(w,next_error_state[2].reshape(1,-1),axis=0)
        q=np.append(q,next_state[:4].reshape(1,-1),axis=0)
        qe=np.append(qe,next_error_state[:4].reshape(1,-1),axis=0)
        w=np.append(w,next_error_state[-3:].reshape(1,-1),axis=0)
        r += reward
        # state = next_state
        #----------------control law (PID controller)-----------------------
        action = -Kp@next_error_state[:4].reshape(-1,1)-Kd@next_error_state[-3:].reshape(-1,1)
        actions = np.append(actions, action.reshape(1,-1),axis=0)
        #--------------------------------------------------------------------

    # env.close()
    #show the total reward
    print("Total Reward is : " + str(r))
    # データの形の整理
    q = q.reshape([-1,4])
    qe = qe.reshape([-1,4])
    w = w.reshape([-1,3])
    # angle = [e for i in]

    # plot the angle and action curve
    curr_dir = os.path.abspath(os.getcwd())
    if not os.path.isdir("results"):
        os.mkdir("results")
    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(np.arange(simulation_iterations-1)*dt, q[:,0],label =r"$q_{0}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, q[:,1],label =r"$q_{1}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, q[:,2],label =r"$q_{2}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, q[:,3],label =r"$q_{3}$")
    plt.title('Quaternion')
    plt.ylabel('quaternion value')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    # plt.savefig(curr_dir + "/results/plot_angle.png")

    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(np.arange(simulation_iterations-1)*dt, qe[:,0],label =r"$q_{0}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, qe[:,1],label =r"$q_{1}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, qe[:,2],label =r"$q_{2}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, qe[:,3],label =r"$q_{3}$")
    plt.title('Quaternion Error')
    plt.ylabel('quaternion value')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    # plt.savefig(curr_dir + "/results/plot_angle.png")

    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(np.arange(simulation_iterations-1)*dt, w[:,0],label =r"$\omega_{x}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, w[:,1],label =r"$\omega_{y}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, w[:,2],label =r"$\omega_{z}$")
    plt.title('Angular velocity')
    plt.ylabel('angular velocity [rad/s]')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    # plt.savefig(curr_dir + "/results/plot_angle.png")

    plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.plot(np.arange(simulation_iterations)*dt, actions[:,0],label = r"$\tau_{x}$")
    plt.plot(np.arange(simulation_iterations)*dt, actions[:,1],label = r"$\tau_{x}$")
    plt.plot(np.arange(simulation_iterations)*dt, actions[:,2],label = r"$\tau_{x}$")
    plt.title('Action')
    plt.ylabel('Input torque [Nm]')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)

    # plt.savefig(curr_dir + "/results/plot_action.png")
    plt.show()


if __name__ == '__main__':
    plt.close()
    val = input('Enter the number 1:train 2:evaluate 3:env_test  > ')
    if val == '1':
        train()
    elif val == '2':
        evaluate()
    elif val == '3':
        env_test()
    else:
        print("You entered the wrong number, run again and choose from 1 or 2 or 3.")
