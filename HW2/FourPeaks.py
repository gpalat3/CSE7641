import numpy as np
import time
import matplotlib.pyplot as plt
import mlrose_hiive

def randomHillClimb(fitness_fn, problem_size):
    fitness = []
    times = []
    for length in problem_size:
        problem = mlrose_hiive.DiscreteOpt(length=length, fitness_fn=fitness_fn, maximize=True, max_val=2)
        start_tm = time.time()
        best_state, best_fitness, _ = mlrose_hiive.random_hill_climb(problem=problem, max_iters=1000, max_attempts=100, random_state=99)
        end_tm = time.time()
        fit_time = end_tm - start_tm
        fitness.append(best_fitness)
        times.append(fit_time)
    fitness = np.array(fitness)
    times = np.array(times)
    problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness_fn, maximize=True, max_val=2)
    rhc = mlrose_hiive.RHCRunner(problem=problem, iteration_list=[1000], max_attempts=100, seed=99, restart_list=[0],
                                 experiment_name='RandomizedHillClimbing', output_directory='output/FourPeaks/')
    rhc_stats, rhc_curves = rhc.run()
    return fitness, times, rhc_stats, rhc_curves

def simulatedAnnealing(fitness_fn, problem_size):
    fitness = []
    times = []
    for length in problem_size:
        problem = mlrose_hiive.DiscreteOpt(length=length, fitness_fn=fitness_fn, maximize=True, max_val=2)
        start_tm = time.time()
        best_state, best_fitness, _ = mlrose_hiive.simulated_annealing(problem=problem, max_iters=1000, max_attempts=100, random_state=99)
        end_tm = time.time()
        fit_time = end_tm - start_tm
        fitness.append(best_fitness)
        times.append(fit_time)
    fitness = np.array(fitness)
    times = np.array(times)
    problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness_fn, maximize=True, max_val=2)
    sa = mlrose_hiive.SARunner(problem=problem, iteration_list=[1000], max_attempts=100, temperature_list=[250],
                               decay_list=[mlrose_hiive.ExpDecay], seed=99, experiment_name='SimulatedAnnealing',
                               output_directory='output/FourPeaks/')
    sa_stats, sa_curves = sa.run()
    return fitness, times, sa_stats, sa_curves

def geneticAlg(fitness_fn, problem_size):
    fitness = []
    times = []
    for length in problem_size:
        problem = mlrose_hiive.DiscreteOpt(length=length, fitness_fn=fitness_fn, maximize=True, max_val=2)
        start_tm = time.time()
        best_state, best_fitness, _ = mlrose_hiive.genetic_alg(problem=problem, pop_size=2*length, mutation_prob=0.2, max_iters=1000, max_attempts=100, random_state=99)
        end_tm = time.time()
        fit_time = end_tm - start_tm
        fitness.append(best_fitness)
        times.append(fit_time)
    fitness = np.array(fitness)
    times = np.array(times)
    problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness_fn, maximize=True, max_val=2)
    ga = mlrose_hiive.GARunner(problem=problem, iteration_list=[1000], max_attempts=100, population_sizes=[200],
                               mutation_rates=[0.2], seed=99, experiment_name='GeneticAlgorithm',
                               output_directory='output/FourPeaks/')
    ga_stats, ga_curves = ga.run()
    return fitness, times, ga_stats, ga_curves

def mimic(fitness_fn, problem_size):
    fitness = []
    times = []
    for length in problem_size:
        problem = mlrose_hiive.DiscreteOpt(length=length, fitness_fn=fitness_fn, maximize=True, max_val=2)
        start_tm = time.time()
        best_state, best_fitness, _ = mlrose_hiive.mimic(problem=problem, pop_size=2*length, keep_pct=0.25, max_iters=1000, max_attempts=100, random_state=99)
        end_tm = time.time()
        fit_time = end_tm - start_tm
        fitness.append(best_fitness)
        times.append(fit_time)
    fitness = np.array(fitness)
    times = np.array(times)
    problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=fitness_fn, maximize=True, max_val=2)
    mimic = mlrose_hiive.MIMICRunner(problem=problem, iteration_list=[1000], max_attempts=100, population_sizes=[200],
                                     keep_percent_list=[0.25], use_fast_mimic=True, seed=99, experiment_name='MIMIC',
                                     output_directory='output/FourPeaks/')
    mimic_stats, mimic_curves = mimic.run()
    return fitness, times, mimic_stats, mimic_curves

def plotCurves(x1, y1, y2, y3, y4, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x1, y1, 'o-', label='RHC')
    plt.plot(x1, y2, 'o-', label='SA')
    plt.plot(x1, y3, 'o-', label='GA')
    plt.plot(x1, y4, 'o-', label='MIMIC')
    plt.legend(loc='best')
    plt.savefig(savefile)

def plotRunnerCurves(x1, y1, x2, y2, x3, y3, x4, y4, title, xlabel, ylabel, savefile):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x1, y1, label='RHC')
    plt.plot(x2, y2, label='SA')
    plt.plot(x3, y3, label='GA')
    plt.plot(x4, y4, label='MIMIC')
    plt.legend(loc='best')
    plt.savefig(savefile)

if __name__ == '__main__':
    fitness_fn = mlrose_hiive.FourPeaks(t_pct=0.1)
    problem_size = range(10, 101, 10)
    fitness_rhc, times_rhc, stats_rhc, curves_rhc = randomHillClimb(fitness_fn, problem_size)
    fitness_sa, times_sa, stats_sa, curves_sa = simulatedAnnealing(fitness_fn, problem_size)
    fitness_ga, times_ga, stats_ga, curves_ga = geneticAlg(fitness_fn, problem_size)
    fitness_mimic, times_mimic, stats_mimic, curves_mimic = mimic(fitness_fn, problem_size)
    title, xlabel, ylabel = ['FourPeaks - Fitness', 'Problem Size', 'Fitness']
    savefile = 'plots/FourPeaks_Fitness_Problem_Size.png'
    plotCurves(problem_size, fitness_rhc, fitness_sa, fitness_ga, fitness_mimic, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['FourPeaks - Time', 'Problem Size', 'Time']
    savefile = 'plots/FourPeaks_Time_Problem_Size.png'
    plotCurves(problem_size, times_rhc, times_sa, times_ga, times_mimic, title, xlabel, ylabel, savefile)
    title, xlabel, ylabel = ['FourPeaks - FEvals', 'Iteration', 'FEvals']
    savefile = 'plots/FourPeaks_Iterations_FEvals.png'
    plotRunnerCurves(curves_rhc[xlabel], curves_rhc[ylabel], curves_sa[xlabel], curves_sa[ylabel], curves_ga[xlabel],
               curves_ga[ylabel], curves_mimic[xlabel], curves_mimic[ylabel], title, xlabel, ylabel, savefile)



