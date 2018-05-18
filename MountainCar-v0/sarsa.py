# coding:utf-8
import gym
import numpy as np
from matplotlib import pyplot

#連続値から離散値への変更
def get_status(_observation):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high-env_low) / 50
    position = int((_observation[0] - env_low[0]) / env_dx[0])
    velocity = int((_observation[1] - env_low[1]) / env_dx[1])
    return position, velocity

 #Q値の更新
def update_q_table(_env, _q_table, _action, _observation, _next_observation, _reward, _episode):
     alpha = 0.2
     gamma = 0.99

     # 行動後の状態で得られる最大行動価値
     next_position, next_velocity = get_status(_next_observation)
     #next_max_q_value = max(_q_table[next_position][next_velocity])
     next_q_value = _q_table[next_position][next_velocity][get_action(_env, _q_table, _next_observation, _episode)]

     # 行動前の状態の行動価値
     position, velocity = get_status(_observation)
     q_value = _q_table[position][velocity][_action]

     _q_table[position][velocity][_action] = q_value + alpha * (_reward + gamma * next_q_value - q_value)

     return _q_table

  # ε-greedy
def get_action(_env, _q_table, _observation, _episode):
      epsilon = 0.002
      if np.random.uniform(0,1) > epsilon:
          position, velocity = get_status(observation)
          _action = np.argmax(_q_table[position][velocity])
      else:
          _action = np.random.choice([0,1,2])
      return _action

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.render()
    observation = env.reset()
    action = 0
    rewards = []
    q_table = np.zeros((50, 50, 3)) # 位置の離散化数×速度の離散化数×行動数
    for i_episode in range(5001):
        total_reward = 0
        observation = env.reset()
        for t in range(200):
            if(i_episode % 1000 == 0):
                env.render()
            #env.render()
            #print(observation)
            action = get_action(env, q_table, observation, i_episode)
            #行動選択
            next_observation, reward, done, info = env.step(action)
            #価値更新
            q_table = update_q_table(env, q_table, action, observation, next_observation, reward, i_episode)
            total_reward += reward

            observation = next_observation

            if done:
                if i_episode % 100 == 0:
                        print("episode: {}, total_reward: {}".format(i_episode, total_reward))
                rewards.append(total_reward)
                break

    x = np.linspace(0, len(rewards), len(rewards))
    y = rewards
    pyplot.plot(x,y)
    pyplot.show()
