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
    iter = []
    R = []
    for i in range(n_episodes):
        done = False
        total_reward = 0
        i = 0
        s = env.reset()
        while not done:
            a = p[s]
            s, r, done, info = env.step(a)
            total_reward += r
            i += 1
            if i > max_iter:
                break
        iter.append(i)
        R.append(total_reward)
    return iter, R

def vIter(env, T, R, gamma, n_episodes):
    avg_R = []
    iter = []
    mean_v = []
    max_v = []
    reward = []
    error = []
    max_iter = 1000
    for g in gamma:
        vi = mdp.ValueIteration(T, R, g)
        vi.run()
        vi_iter, vi_R = runGymEnv(env, vi.policy, n_episodes, max_iter)
        avg_R.append(sum(vi_R) / n_episodes)
        iter.append(list(range(1, len(vi.run_stats) +1 )))
        mean_v.append([i['Mean V'] for i in vi.run_stats])
        max_v.append([i['Max V'] for i in vi.run_stats])
        reward.append([i['Reward'] for i in vi.run_stats])
        error.append([i['Error'] for i in vi.run_stats])
        '''
        ax[0, 0].plot(range(1, vi.iter + 1), [i['Mean V'] for i in vi.run_stats], label=g)
        ax[0, 0].set_xlabel('Iteration')
        ax[0, 0].set_ylabel('Mean V')
        ax[0, 0].grid()
        ax[0, 0].legend(loc='best')
        ax[0, 1].plot(range(1, vi.iter + 1), [i['Reward'] for i in vi.run_stats], label=g)
        ax[0, 1].set_xlabel('Iteration')
        ax[0, 1].set_ylabel('Reward')
        ax[0, 1].grid()
        ax[0, 1].legend(loc='best')
        ax[1, 0].plot(range(1, vi.iter + 1), [i['Max V'] for i in vi.run_stats], label=g)
        ax[1, 0].set_xlabel('Iteration')
        ax[1, 0].set_ylabel('Max V')
        ax[1, 0].grid()
        ax[1, 0].legend(loc='best')
        ax[1, 1].plot(range(1, vi.iter + 1), [i['Error'] for i in vi.run_stats], label=g)
        ax[1, 1].set_xlabel('Iteration')
        ax[1, 1].set_ylabel('Error')
        ax[1, 1].grid()
        ax[1, 1].legend(loc='best')
    
    ax[1, 1].plot(gamma, avg_R)
    ax[1, 1].set_xlabel('Gamma')
    ax[1, 1].set_ylabel('Average Reward')
    ax[1, 1].grid()
    
    savefile = 'plots/V_Iter.png'
    plt.savefig(savefile)
    '''
    title, xlabel, ylabel = ['Value Iteration', 'Iterations', 'Mean V']
    savefile = 'plots/V_Iter_MeanV.png'
    plotCurves(iter, mean_v, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Value Iteration', 'Iterations', 'Max V']
    savefile = 'plots/V_Iter_MaxV.png'
    plotCurves(iter, max_v, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Value Iteration', 'Iterations', 'Reward']
    savefile = 'plots/V_Iter_Reward.png'
    plotCurves(iter, reward, gamma, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['Value Iteration', 'Iterations', 'Error']
    savefile = 'plots/V_Iter_Error.png'
    plotCurves(iter, error, gamma, title, xlabel, ylabel, savefile)

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

def pIter(env, T, R, gamma, n_episodes):
    avg_R = []
    max_iter = 1000
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for g in gamma:
        pi = mdp.ValueIteration(T, R, g)
        pi.run()
        pi_iter, pi_R = runGymEnv(env, pi.policy, n_episodes, max_iter)
        avg_R.append(sum(pi_R)/n_episodes)


if __name__ == '__main__':
    random_map = generate_random_map(size=20, p=0.8)
    environment = gym.make('FrozenLake-v1', desc=random_map)
    n_episodes = 500
    environment.reset()
    environment.render()
    T, R = transitionReward(environment)
    # gamma = [0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
    gamma = [0.8, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99]
    vIter(environment, T, R, gamma, n_episodes)
    pIter(environment, T, R, gamma, n_episodes)



