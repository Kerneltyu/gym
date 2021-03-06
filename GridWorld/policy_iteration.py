import gridworld
import numpy as np
from numpy.random import *

THETA = 0.9
GAMMA = 0.05
PROB = 0
NEXT_STATE = 1
REWARD = 2
IS_DONE = 3

env = gridworld.GridworldEnv()
actions = [gridworld.UP, gridworld.RIGHT, gridworld.DOWN, gridworld.LEFT]
num_S = np.prod(env.shape)

def init_policy():
    #ランダムに初期化
    #pi = randint(len(actions),size=num_S)
    pi = [0 for i in range(num_S)]
    return pi

def init_value():
    #v_list = rand(num_S)
    v_list = np.zeros(num_S)
    return v_list

def init_trans_table():
    trans_table = np.zeros((num_S, len(actions), num_S))
    for s in range(num_S):
        for j in range(len(actions)):
            for s_prime in range(num_S):
                if(s_prime == s+1 and j==1 and s_prime%4 != 0):
                    trans_table[s][j][s_prime] = 1
                elif(s_prime==s-1 and j==3 and s_prime%4 != 3):
                    trans_table[s][j][s_prime] = 1
                elif(s_prime%4 == s%4 and s_prime//4 == s//4 + 1 and j==2):
                    trans_table[s][j][s_prime] = 1
                elif(s_prime%4 == s%4 and s_prime//4 == s//4 - 1 and j==0):
                    trans_table[s][j][s_prime] = 1
    return trans_table

def init_reward_func():
    reward_func = np.zeros((num_S, len(actions), num_S))
    for i in range(num_S):
        for j in range(len(actions)):
            for k in range(num_S):
                if(k == 0 or k == num_S-1):
                    reward_func[i][j][k] = 1
    return reward_func

def evaluate_policy(pi, v_list, trans_table, reward_func):
    '''
    args:
        pi: 現在の方策
        v_list: 状態価値関数
        trans_table: 状態遷移表
        reward_func: 報酬関数
    '''
    #traisitional probはgridworldのPで与えられる．ここでは決定的
    gamma = GAMMA
    delta = np.float('inf')
    counter = 0
    while delta > THETA:
        counter += 1
        delta = 0
        for s in range(1, num_S-1):
            v = v_list[s]
            v_list[s] = sum([trans_table[s][pi[s]][s_prime] * (reward_func[s][pi[s]][s_prime] + gamma * v_list[s_prime]) for s_prime in range(num_S)])
        #print("{}:{}".format(counter, delta))
    #print(v_list)
    return v_list

def improve_policy(pi, v_list, trans_table, reward_func):
    gamma = GAMMA
    policy_stable = True
    for s in range(1, num_S-1):
        b = pi[s]
        #print([ sum([trans_table[s][a][s_prime] * (reward_func[s][a][s_prime] + gamma * v_list[s_prime]) for s_prime in range(num_S)]) for a in range(len(actions))])
        pi[s] = np.argmax([ sum([trans_table[s][a][s_prime] * (reward_func[s][a][s_prime] + gamma * v_list[s_prime]) for s_prime in range(num_S)]) for a in range(len(actions))])
        #pi[s] = np.identity(len(actions))[]
        if b != pi[s]: policy_stable = False
    return policy_stable, pi

def convert_arrow(pi):
    arrow_pi = []
    for s, act in enumerate(pi):
        if(s==0 or s==len(pi)-1):
            arrow_pi.append('G')
        elif(act == 0):
            arrow_pi.append('↑')
        elif(act == 1):
            arrow_pi.append('→')
        elif(act == 2):
            arrow_pi.append('↓')
        else:
            arrow_pi.append('←')
    return arrow_pi
def main():
    s, obs = env.render()
    trans_info = env.P
    # {state:{action:[(prob, next_state, reward, is_done)]}}
    pi = init_policy()
    v_list = init_value()
    trans_table = init_trans_table()
    print(pi)
    print(v_list)
    reward_func = init_reward_func()
    policy_stable = False

    while not policy_stable:
        v_list = evaluate_policy(pi, v_list, trans_table, reward_func)
        policy_stable, pi = improve_policy(pi, v_list, trans_table, reward_func)
        #print(policy_stable)
    print(np.array(v_list).reshape(4,4))
    print(np.array(convert_arrow(pi)).reshape(4,4))

if __name__ == '__main__':
    main()
