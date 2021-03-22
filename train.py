import os

import math
import matplotlib.pyplot as plt
import torch
from gym import wrappers 
import numpy as np
import wandb


from network import TD3Agent
from utils import *


def train():
    # simulation of the agent solving the spacecraft attitude control problem
    env = make("SatelliteContinuous")
    #logger
    wandb.init(project='Satellite-continuous',
        config={
        "batch_size": 128,
        "critic_lr": 1e-3,
        "actor_lr": 1e-4,
        "max_episodes": 1000,
        "max_steps": 300,
        "gamma": 0.99,
        "tau" : 1e-3,
        "buffer_maxlen": 100000,
        "policy_noise": 0.2,
        "policy_freq": 2,
        "noise_clip": 0.5,
        "prioritized_on": False,
        "State": 'angle:4, ang_rate:4, ang_vel:3',}
    )
    config = wandb.config

    max_episodes = config.max_episodes
    max_steps = config.max_steps
    batch_size = config.batch_size

    policy_noise = config.policy_noise
    policy_freq = config.policy_freq
    noise_clip = config.noise_clip

    gamma = config.gamma
    buffer_maxlen = config.buffer_maxlen
    tau = config.tau
    critic_lr = config.critic_lr
    actor_lr = config.actor_lr

    agent = TD3Agent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr, True, max_episodes * max_steps,
                    policy_freq, policy_noise, noise_clip)
    # wandb.watch([agent.critic,agent.actor], log="all")
    # curr_dir = os.path.abspath(os.getcwd())
    # agent = torch.load(curr_dir + "/models/spacecraft_control_ddpg.pkl")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)

    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
	# plt.show()
    # plt.savefig(curr_dir + "/results/plot_reward_hist.png")

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

    agent = torch.load(curr_dir + "/models/spacecraft_control_ddpg.pkl",map_location='cpu')
    agent.device = torch.device('cpu')
    agent.train = False

    state = env.reset()
    print('The goal angle :'+ str(env.goalEuler) + " the target multi:" + str(env.multi))
    r = 0
    qe = np.empty((0,4))
    q = np.empty((0,4))
    w = np.empty((0,3))
    actions = np.empty((0,3))
    r_hist = np.empty((0,3))    

    dt = 0.1
    simutime = 30
    max_steps = int(simutime/dt) # dt is 0.1

    for i in range(max_steps):
        action = agent.get_action(state)
        # action = np.squeeze(action)
        next_error_state, reward, done, next_state, _ = env.step(action)
        # env.render()
        q=np.append(q,next_state[0:4].reshape(1,-1),axis=0)
        qe=np.append(qe,next_error_state[0:4].reshape(1,-1),axis=0)
        w=np.append(w,next_error_state[8:11].reshape(1,-1),axis=0)
        r += reward
        actions = np.append(actions, action.reshape(1,-1),axis=0)
        # r_hist = np.append(r_hist, np.array([-env.r1,-env.r2,-env.r3]).reshape(1,-1),axis=0)

        state = next_error_state

    env.close()
    #-------------------------------結果のプロット----------------------------------
    #region
    #show the total reward
    print("Total Reward is : " + str(r))
    # データの形の整理
    q = q.reshape([-1,4])
    qe = qe.reshape([-1,4])
    w = w.reshape([-1,3])
    
    # plot the angle and action curve
    #-------------------plot settings------------------------------
    plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    plt.rcParams["font.size"] = 10 # 全体のフォントサイズが変更されます。
    plt.rcParams['xtick.labelsize'] = 10 # 軸だけ変更されます。
    plt.rcParams['ytick.labelsize'] = 10 # 軸だけ変更されます 
    plt.rcParams['xtick.direction'] = 'in' # x axis in
    plt.rcParams['ytick.direction'] = 'in' # y axis in 
    plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    plt.rcParams['axes.grid'] = True # make grid
    plt.rcParams["legend.loc"] = "best"         # 凡例の位置、"best"でいい感じのところ
    plt.rcParams["legend.frameon"] = True       # 凡例を囲うかどうか、Trueで囲う、Falseで囲わない
    plt.rcParams["legend.framealpha"] = 1.0     # 透過度、0.0から1.0の値を入れる
    plt.rcParams["legend.facecolor"] = "white"  # 背景色
    # plt.rcParams["legend.edgecolor"] = "black"  # 囲いの色
    plt.rcParams["legend.fancybox"] = True     # Trueにすると囲いの四隅が丸くなる
    #--------------------------------------------------------------  
    curr_dir = os.path.abspath(os.getcwd())
    if not os.path.isdir("results"):
        os.mkdir("results")
    
    plt.figure(figsize=(12,5),dpi=100)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.subplot(231)
    plt.plot(np.arange(max_steps)*dt, q[:,0],label =r"$q_{0}$")
    plt.plot(np.arange(max_steps)*dt, q[:,1],label =r"$q_{1}$")
    plt.plot(np.arange(max_steps)*dt, q[:,2],label =r"$q_{2}$")
    plt.plot(np.arange(max_steps)*dt, q[:,3],label =r"$q_{3}$")
    plt.title('Quaternion')
    plt.ylabel('quaternion value')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.tight_layout()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    # plt.savefig(curr_dir + "/results/plot_quaternion.png")

    # plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.subplot(232)
    plt.plot(np.arange(max_steps)*dt, qe[:,0],label =r"$q_{0}$")
    plt.plot(np.arange(max_steps)*dt, qe[:,1],label =r"$q_{1}$")
    plt.plot(np.arange(max_steps)*dt, qe[:,2],label =r"$q_{2}$")
    plt.plot(np.arange(max_steps)*dt, qe[:,3],label =r"$q_{3}$")
    plt.title('Quaternion Error')
    plt.ylabel('quaternion value')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.tight_layout()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    # plt.savefig(curr_dir + "/results/plot_error_quaternion.png")

    angle = np.array([np.rad2deg(env.dcm2euler(env.quaternion2dcm(q[i,:]))).tolist() for i in range(max_steps)])
    angle = angle.reshape([-1,3])
    print(angle[-1,:])
    # plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.subplot(233)
    plt.plot(np.arange(max_steps)*dt, angle[:,0],label = r"$\phi$")
    plt.plot(np.arange(max_steps)*dt, angle[:,1],label = r"$\theta$")
    plt.plot(np.arange(max_steps)*dt, angle[:,2],label = r"$\psi$")
    # plt.title('Action')
    plt.ylabel('angle [deg]')
    plt.xlabel(r'time [s]')
    plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.05), ncol=3)
    plt.tight_layout()
    # plt.ylim(-20, 20)
    plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.savefig(curr_dir + "/results/plot_angle.png")

    # plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.subplot(234)
    plt.plot(np.arange(max_steps)*dt, w[:,0],label =r"$\omega_{x}$")
    plt.plot(np.arange(max_steps)*dt, w[:,1],label =r"$\omega_{y}$")
    plt.plot(np.arange(max_steps)*dt, w[:,2],label =r"$\omega_{z}$")
    plt.title('Angular velocity')
    plt.ylabel('angular velocity [rad/s]')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.tight_layout()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    # plt.savefig(curr_dir + "/results/plot_ang_vel.png")

    # plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.subplot(235)
    plt.plot(np.arange(max_steps)*dt, actions[:,0],label = r"$\tau_{x}$")
    plt.plot(np.arange(max_steps)*dt, actions[:,1],label = r"$\tau_{x}$")
    plt.plot(np.arange(max_steps)*dt, actions[:,2],label = r"$\tau_{x}$")
    plt.title('Action')
    plt.ylabel('Input torque [Nm]')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.tight_layout()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    # plt.savefig(curr_dir + "/results/plot_torque.png")
    plt.savefig(curr_dir + "/results/total_results.png")

    # plt.figure(figsize=(8,4),dpi=100)
    # plt.plot(np.arange(max_steps)*dt, r_hist[:,0],label = r"$q$ pnlty")
    # plt.plot(np.arange(max_steps)*dt, r_hist[:,1],label = r"$\omega$ pnlty")
    # plt.plot(np.arange(max_steps)*dt, r_hist[:,2],label = r"$\tau$ pnlty")
    # plt.plot(np.arange(max_steps)*dt, r_hist[:,0]+r_hist[:,1]+r_hist[:,2],label = r"$toal$",linestyle='dotted')
    # # plt.title('Action')
    # plt.ylabel('reward')
    # plt.xlabel(r'time [s]')
    # plt.tight_layout()
    # plt.legend()
    # # plt.ylim(-20, 20)
    # plt.grid(True, color='k', linestyle='dotted', linewidth=0.8)
    # plt.savefig(curr_dir + "/results/reward_compo.png")

    plt.show()
    #endregion
    # -------------------------結果プロット終わり--------------------------------
def env_test():

    # simulation of the agent solving the cartpole swing-up problem
    env = make("SatelliteContinuous")
    curr_dir = os.path.abspath(os.getcwd())
    env.reset()
    print('The goal angle :'+ str(env.goalEuler) + " the target multi:" + str(env.multi))
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

    dt = 0.1
    simutime = 50
    simulation_iterations = int(simutime/dt) -1 # dt is 0.01

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
    plt.figure(figsize=(12,5),dpi=100)
    # plt.figure(figsize=(5.0,3.5),dpi=100
    plt.subplot(231)
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

    # plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.subplot(232)
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

    angle = np.array([np.rad2deg(env.dcm2euler(env.quaternion2dcm(q[i,:]))).tolist() for i in range(simulation_iterations-1)])
    angle = angle.reshape([-1,3])
    # plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.subplot(233)
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

    # plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.subplot(234)
    plt.plot(np.arange(simulation_iterations-1)*dt, w[:,0],label =r"$\omega_{x}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, w[:,1],label =r"$\omega_{y}$")
    plt.plot(np.arange(simulation_iterations-1)*dt, w[:,2],label =r"$\omega_{z}$")
    plt.title('Angular velocity')
    plt.ylabel('angular velocity [rad/s]')
    plt.xlabel(r'time [s]')
    plt.legend()
    plt.grid(color='k', linestyle='dotted', linewidth=0.6)
    # plt.savefig(curr_dir + "/results/plot_angle.png")

    # plt.figure(figsize=(5.0,3.5),dpi=100)
    plt.subplot(235)
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
    val = input('Enter the number 1:train 2:evaluate 3:env_test > ')
    if val == '1':
        train()
    elif val == '2':
        evaluate()
    elif val == '3':
        env_test()
    else:
        print("You entered the wrong number, run again and choose from 1 or 2 or 3.")
