from environment import SatelliteContinuousEnv
from gym import make as gym_make
from tqdm import tqdm
from collections import OrderedDict
import wandb


def make(env_name, *make_args, **make_kwargs):
    if env_name == "SatelliteContinuous":
        return SatelliteContinuousEnv()
    else:
        return gym_make(env_name, *make_args, **make_kwargs)


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    counter = 0
    try:
        # with tqdm(range(max_episodes),leave=False) as pbar:
        #     for episode, ch in enumerate(pbar):
        #         pbar.set_description("[Train] Episode %d" % episode)
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0    
            for step in range(max_steps):
                # pbar.set_postfix(OrderedDict(goal= env.goalEuler, steps = step))#OrderedDict(loss=1-episode/5, acc=episode/10))
                action = agent.get_action(state, (episode + 1) * (step + 1))
                next_error_state, reward, done, next_state, _ = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_error_state, done)
                episode_reward += reward

                # update the agent if enough transitions are stored in replay buffer
                if len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)

                if done or step == max_steps - 1:
                    episode_rewards.append(episode_reward)
                    wandb.log({ "episode reward": episode_reward,
                                "critic_loss": agent.critic_loss_for_log,
                                "actor_loss": agent.actor_loss_for_log})
                    # Count number of consecutive games with cumulative rewards >-55 for early stopping
                    print("\nEpisode " + str(episode) + " total reward : " + str(episode_reward))
                    break

                state = next_error_state
                # Early stopping, if cumulative rewards of 10 consecutive games were >-55
                # if counter == 10:
                    # break
    except KeyboardInterrupt:
        print('Training stopped manually!!!')
        pass

    return episode_rewards


def mini_batch_train_pd(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    counter = 0
    try:
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0    
            input = np.array([0,0,0]).reshape(1,3)
            for step in range(max_steps):
                action = agent.get_action(state, (episode + 1) * (step + 1))
                next_error_state, reward, done, next_state, _ = env.step(input)
                agent.replay_buffer.push(state, action, reward, next_error_state, done)
                action = (action + 1)/2
                kp = action[0]
                kd = action[1]
                Kp = np.array([[0,kp,0,0],
                                [0,0,kp,0],
                                [0,0,0,kp]])
                Kd = np.array([[kd,0,0],
                                [0,kd,0],
                                [0,0,kd]])
                input = -Kp@next_error_state[:4].reshape(-1,1)-Kd@next_error_state[-3:].reshape(-1,1)
                episode_reward += reward

                # update the agent if enough transitions are stored in replay buffer
                if len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)

                if done or step == max_steps - 1:
                    episode_rewards.append(episode_reward)
                    wandb.log({ "episode reward": episode_reward,
                                "critic_loss": agent.critic_loss_for_log,
                                "actor_loss": agent.actor_loss_for_log})
                    # Count number of consecutive games with cumulative rewards >-55 for early stopping
                    print("\nEpisode " + str(episode) + " total reward : " + str(episode_reward))
                    break

                state = next_error_state
    except KeyboardInterrupt:
        print('Training stopped manually!!!')
        pass

    return episode_rewards