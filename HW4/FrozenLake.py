import numpy as np
import time
import tools
import matplotlib.pyplot as plt
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from hiive.mdptoolbox import example, mdp

def transitionReward(environment):
    nA = environment.env.nA
    nS = environment.env.nS
    T = np.zeros([nA, nS, nS])
    R = np.zeros([nS, nA])
    for s in range(nS):
        for a in range(nA):
            P = environment.P[s][a]
            for p, s1, r, _ in P:
                T[a, s, s1] += p
                R[s, a] = r
            T[a, s, :] = T[a, s, :]/np.sum(T[a, s, :])
    return T, R

def runGymEnv(env, p, n_episodes, max_iter):
    steps = []
    fail_count = 0
    success_count = 0
    R = []
    for episode in range(n_episodes):
        s = env.reset()
        total_reward = 0
        i = 0
        while True:
            a = p[s]
            s, r, done, info = env.step(a)
            total_reward += r
            i += 1
            if i == max_iter:
                steps.append(i)
            if done and r == 1:
                steps.append(i)
                success_count += 1
                break
            elif done and r == 0:
                steps.append(i)
                fail_count += 1
                break
            elif i > max_iter:
                fail_count += 1
                break
        R.append(total_reward)
    return steps, R, success_count, fail_count

def vIter(env, T, R, gamma, n_episodes, max_iter, size):
    avg_R = []
    iter = []
    mean_v = []
    max_v = []
    reward = []
    error = []
    time = []
    success_rate = []
    fail_rate = []
    for g in gamma:
        vi = mdp.ValueIteration(T, R, g)
        vi.run()
        vi_steps, vi_R, vi_success_count, vi_fail_count = runGymEnv(env, vi.policy, n_episodes, max_iter)
        avg_R.append(sum(vi_R) / n_episodes)
        success_rate.append(vi_success_count / n_episodes)
        fail_rate.append(vi_fail_count / n_episodes)
        '''
        print('gamma: ', g)
        print('vi_steps: ', vi_steps)
        print('vi_R: ', vi_R)
        print('vi_success: ', vi_success_count)
        print('vi_fail: ', vi_fail_count)
        print('vi_run_stats: ', vi.run_stats)
        '''
        iter.append(list(range(1, len(vi.run_stats) + 1)))
        mean_v.append([i['Mean V'] for i in vi.run_stats])
        max_v.append([i['Max V'] for i in vi.run_stats])
        reward.append([i['Reward'] for i in vi.run_stats])
        error.append([i['Error'] for i in vi.run_stats])
        time.append([i['Time'] for i in vi.run_stats])
    title, xlabel, ylabel = ['Value Iteration', 'Iterations', 'Mean V']
    savefile = 'plots/FrozenLake/V_Iter_MeanV_' + str(size) + '.png'
    plotCurves(iter, mean_v, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Value Iteration', 'Iterations', 'Max V']
    savefile = 'plots/FrozenLake/V_Iter_MaxV_' + str(size) + '.png'
    plotCurves(iter, max_v, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Value Iteration', 'Iterations', 'Reward']
    savefile = 'plots/FrozenLake/V_Iter_Reward_' + str(size) + '.png'
    plotCurves(iter, reward, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Value Iteration', 'Iterations', 'Error']
    savefile = 'plots/FrozenLake/V_Iter_Error_' + str(size) + '.png'
    plotCurves(iter, error, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Value Iteration', 'Iterations', 'Time']
    savefile = 'plots/FrozenLake/V_Iter_Time_' + str(size) + '.png'
    plotCurves(iter, time, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Value Iteration', 'Gamma', 'Average Reward']
    savefile = 'plots/FrozenLake/V_Iter_Gamma_Avg_Reward_' + str(size) + '.png'
    plotIndCurves(gamma, avg_R, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Value Iteration', 'Gamma', 'Success Rate']
    savefile = 'plots/FrozenLake/V_Iter_Gamma_Success_Rate_' + str(size) + '.png'
    plotIndCurves(gamma, success_rate, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Value Iteration', 'Gamma', 'Failure Rate']
    savefile = 'plots/FrozenLake/V_Iter_Gamma_Fail_Rate_' + str(size) + '.png'
    plotIndCurves(gamma, fail_rate, title, xlabel, ylabel, savefile)

def pIter(env, T, R, gamma, n_episodes, max_iter, size):
    iter = []
    mean_v = []
    max_v = []
    reward = []
    error = []
    time = []
    avg_R = []
    success_rate = []
    fail_rate = []
    for g in gamma:
        pi = mdp.ValueIteration(T, R, g)
        pi.run()
        pi_steps, pi_R, pi_success_count, pi_fail_count = runGymEnv(env, pi.policy, n_episodes, max_iter)
        avg_R.append(sum(pi_R) / n_episodes)
        success_rate.append(pi_success_count / n_episodes)
        fail_rate.append(pi_fail_count / n_episodes)
        iter.append(list(range(1, len(pi.run_stats) + 1)))
        mean_v.append([i['Mean V'] for i in pi.run_stats])
        max_v.append([i['Max V'] for i in pi.run_stats])
        reward.append([i['Reward'] for i in pi.run_stats])
        error.append([i['Error'] for i in pi.run_stats])
        time.append([i['Time'] for i in pi.run_stats])
    title, xlabel, ylabel = ['Policy Iteration', 'Iterations', 'Mean V']
    savefile = 'plots/FrozenLake/P_Iter_MeanV_' + str(size) + '.png'
    plotCurves(iter, mean_v, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Policy Iteration', 'Iterations', 'Max V']
    savefile = 'plots/FrozenLake/P_Iter_MaxV_' + str(size) + '.png'
    plotCurves(iter, max_v, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Policy Iteration', 'Iterations', 'Reward']
    savefile = 'plots/FrozenLake/P_Iter_Reward_' + str(size) + '.png'
    plotCurves(iter, reward, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Policy Iteration', 'Iterations', 'Error']
    savefile = 'plots/FrozenLake/P_Iter_Error_' + str(size) + '.png'
    plotCurves(iter, error, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Policy Iteration', 'Iterations', 'Time']
    savefile = 'plots/FrozenLake/P_Iter_Time_' + str(size) + '.png'
    plotCurves(iter, time, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Policy Iteration', 'Gamma', 'Average Reward']
    savefile = 'plots/FrozenLake/P_Iter_Gamma_Avg_Reward_' + str(size) + '.png'
    plotIndCurves(gamma, avg_R, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Policy Iteration', 'Gamma', 'Success Rate']
    savefile = 'plots/FrozenLake/P_Iter_Gamma_Success_Rate_' + str(size) + '.png'
    plotIndCurves(gamma, success_rate, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Policy Iteration', 'Gamma', 'Failure Rate']
    savefile = 'plots/FrozenLake/P_Iter_Gamma_Fail_Rate_' + str(size) + '.png'
    plotIndCurves(gamma, fail_rate, title, xlabel, ylabel, savefile)

def qLearningGamma(T, R, gamma, size):
    iter = []
    mean_v = []
    max_v = []
    reward = []
    error = []
    time = []
    for g in gamma:
        q = mdp.QLearning(T, R, gamma=g, alpha=0.1, alpha_decay=0.99999, epsilon=0.95, epsilon_decay=0.9999, n_iter=1000000)
        q.run()
        iter.append(list(range(1, len(q.run_stats) + 1)))
        mean_v.append([i['Mean V'] for i in q.run_stats])
        max_v.append([i['Max V'] for i in q.run_stats])
        reward.append([i['Reward'] for i in q.run_stats])
        error.append([i['Error'] for i in q.run_stats])
        time.append([i['Time'] for i in q.run_stats])
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Mean V']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_MeanV_Gamma_' + str(size) + '.png'
    plotCurves(iter, mean_v, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Max V']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_MaxV_Gamma_' + str(size) + '.png'
    plotCurves(iter, max_v, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Reward']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_Reward_Gamma_' + str(size) + '.png'
    plotCurves(iter, reward, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Error']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_Error_Gamma_' + str(size) + '.png'
    plotCurves(iter, error, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Time']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_Time_Gamma_' + str(size) + '.png'
    plotCurves(iter, time, gamma, title, xlabel, ylabel, savefile)

def qLearningAlpha(T, R, gamma, alpha, epsilon, size):
    iter = []
    mean_v = []
    max_v = []
    reward = []
    error = []
    time = []
    for a in alpha:
        q = mdp.QLearning(T, R, gamma=gamma, alpha=a, alpha_decay=0.99999, epsilon=epsilon, epsilon_decay=0.9999, n_iter=1000000)
        q.run()
        iter.append(list(range(1, len(q.run_stats) + 1)))
        mean_v.append([i['Mean V'] for i in q.run_stats])
        max_v.append([i['Max V'] for i in q.run_stats])
        reward.append([i['Reward'] for i in q.run_stats])
        error.append([i['Error'] for i in q.run_stats])
        time.append([i['Time'] for i in q.run_stats])
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Mean V']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_MeanV_Alpha_' + str(size) + '.png'
    plotCurves(iter, mean_v, alpha, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Max V']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_MaxV_Alpha_' + str(size) + '.png'
    plotCurves(iter, max_v, alpha, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Reward']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_Reward_Alpha_' + str(size) + '.png'
    plotCurves(iter, reward, alpha, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Error']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_Error_Alpha_' + str(size) + '.png'
    plotCurves(iter, error, alpha, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Time']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_Time_Alpha_' + str(size) + '.png'
    plotCurves(iter, time, alpha, title, xlabel, ylabel, savefile)

def qLearningEpsilon(T, R, gamma, alpha, epsilon, size):
    iter = []
    mean_v = []
    max_v = []
    reward = []
    error = []
    time = []
    for e in epsilon:
        q = mdp.QLearning(T, R, gamma=gamma, alpha=alpha, alpha_decay=0.99999, epsilon=e, epsilon_decay=0.9999, n_iter=1000000)
        q.run()
        iter.append(list(range(1, len(q.run_stats) + 1)))
        mean_v.append([i['Mean V'] for i in q.run_stats])
        max_v.append([i['Max V'] for i in q.run_stats])
        reward.append([i['Reward'] for i in q.run_stats])
        error.append([i['Error'] for i in q.run_stats])
        time.append([i['Time'] for i in q.run_stats])
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Mean V']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_MeanV_Epsilon_' + str(size) + '.png'
    plotCurves(iter, mean_v, epsilon, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Max V']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_MaxV_Epsilon_' + str(size) + '.png'
    plotCurves(iter, max_v, epsilon, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Reward']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_Reward_Epsilon_' + str(size) + '.png'
    plotCurves(iter, reward, epsilon, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Error']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_Error_Epsilon_' + str(size) + '.png'
    plotCurves(iter, error, epsilon, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Q Learning', 'Iterations', 'Time']
    savefile = 'plots/FrozenLake/Q_Learning_Iter_Time_Epsilon_' + str(size) + '.png'
    plotCurves(iter, time, epsilon, title, xlabel, ylabel, savefile)

def plotCurves(x, y, gamma, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for i in range(len(y)):
        plt.plot(x[i], y[i], label=gamma[i])
    plt.legend(loc='best')
    plt.grid()
    plt.savefig(savefile)

def plotIndCurves(x, y, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y, 'o-')
    plt.grid()
    plt.savefig(savefile)

if __name__ == '__main__':
    size = 8
    random_map = generate_random_map(size=size, p=0.8)
    # random_map = generate_random_map(size=8, p=0.3)
    environment = gym.make('FrozenLake-v1', desc=random_map)
    n_episodes = 1000
    max_iter = 3000
    environment.reset()
    environment.render()
    T, R = transitionReward(environment)
    # gamma = [0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
    # gamma = [0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    gamma = [0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.99]
    print('Started Value Iteration')
    vIter(environment, T, R, gamma, n_episodes, max_iter, size)
    print('Ended Value Iteration')
    print('Started Policy Iteration')
    pIter(environment, T, R, gamma, n_episodes, max_iter, size)
    print('Started Policy Iteration')
    # gamma = [0.8, 0.99]
    print('Started Q Learning Gamma')
    qLearningGamma(T, R, gamma, size)
    print('Ended Q Learning Gamma')
    print('Started Q Learning Alpha')
    alpha = [0.001, 0.01, 0.1, 1]
    gamma = 0.95
    epsilon = 0.95
    qLearningAlpha(T, R, gamma, alpha, epsilon, size)
    print('Ended Q Learning Alpha')
    print('Started Q Learning Epsilon')
    alpha = 0.1
    gamma = 0.95
    epsilon = [0.1, 0.5, 0.9, 0.95, 0.99, 1]
    qLearningEpsilon(T, R, gamma, alpha, epsilon, size)
    print('Ended Q Learning Epsilon')
    print('Started deriving optimal numbers for VI:')
    vi = mdp.ValueIteration(T, R, gamma=0.99)
    vi.run()
    print('Ended deriving optimal numbers for VI:')
    print('Started deriving optimal numbers for PI:')
    pi = mdp.ValueIteration(T, R, gamma=0.99)
    pi.run()
    print('Ended deriving optimal numbers for PI:')
    print('Started deriving optimal numbers for Q Learning:')
    q_95 = mdp.QLearning(T, R, gamma=0.95, alpha=0.1, alpha_decay=0.99999, epsilon=0.95, epsilon_decay=0.9999,
                      n_iter=1000000)
    q_95.run()
    q_99 = mdp.QLearning(T, R, gamma=0.99, alpha=0.1, alpha_decay=0.99999, epsilon=0.95, epsilon_decay=0.9999,
                         n_iter=1000000)
    q_99.run()
    print('Ended deriving optimal numbers for Q Learning:')
    print('Value Iteration Policy:', vi.policy)
    print('Policy Iteration Policy:', pi.policy)
    print('Q Learning Policy Gamma=0.95:', q_95.policy)
    print('Q Learning Policy Gamma=0.99:', q_99.policy)

'''
gamma = 0.99 Value Iteration Policy: 
(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 3, 3, 3, 3, 1, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 3, 3, 1, 3, 1, 1, 1, 0, 0, 2, 0, 0, 2, 2, 3, 0, 0, 0, 1, 1, 2, 0, 0, 2, 1, 0)
gamma = 0.99 Policy Iteration Policy: 
(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 3, 3, 3, 3, 1, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 3, 3, 1, 3, 1, 1, 1, 0, 0, 2, 0, 0, 2, 2, 3, 0, 0, 0, 1, 1, 2, 0, 0, 2, 1, 0)
gamma = 0.95 Q Learning Policy: 
(1, 0, 2, 0, 3, 1, 2, 0, 0, 1, 1, 1, 1, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 3, 2, 1, 0, 3, 1, 0, 0, 0, 2, 1, 1, 1, 3, 0, 0, 0, 3, 0, 2, 2, 2, 0, 0, 0, 2, 3, 3, 0, 0, 2, 1, 0)
'''