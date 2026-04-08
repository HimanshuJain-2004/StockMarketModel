import csv

from sklearn import svm
import numpy as np
from numpy.random import random as rand
import random
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vmdpy import VMD
from scipy.fftpack import hilbert, fft, ifft
from math import log
import pandas as pd
import copy
# from tensorflow.keras.losses import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_squared_error
datasets = 'DJIA'
data_addr = 'C:/Users/Gursimranjeet Singh/REGCN/data/data/'+datasets+'.npy'

tau = 0.  # noise-tolerance (no strict fidelity enforcement)
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7

def coarse_grid_search(data1):
    K_list = [2, 3, 4, 5]
    alpha_list = [500, 1000, 2000, 3000]

    best_score = np.inf
    best_params = None

    for K in K_list:
        for alpha in alpha_list:
            s = 0
            for i in range(data1.shape[1]):
                u, _, _ = VMD(
                    data1[:, i],
                    alpha,
                    tau,
                    K,
                    DC,
                    init,
                    tol
                )
                u1 = np.sum(u, axis=0)
                residual = data1[:, i] - u1

                df = pd.DataFrame(
                    np.vstack([data1[:, i], residual]).T
                )
                s += df.corr().iloc[0, 1]

            s /= data1.shape[1]

            print(f"[GRID] K={K}, alpha={alpha}, score={s:.4f}")

            if s < best_score:
                best_score = s
                best_params = (K, alpha)

    print("\n[GRID BEST– MIN CORR]", best_params, "score:", best_score)
    return best_params


def Fun(x, data1, low, ub):
    K = int(x[0])
    alpha = int(x[1])

    if K <= low[0]:
        K = low[0]
    if K >= ub[0]:
        K = ub[0]

    if alpha <= low[1]:
        alpha = low[1]
    if alpha >= ub[1]:
        alpha = ub[1]
    s = 0
    for i in range(data1.shape[1]):
        u, u_hat, omega = VMD(data1[:,i], alpha, tau, K, DC, init, tol)
        u1 = np.sum(u, axis=0)
        u2 = list(map(lambda x: x[0] - x[1], zip(data1[:,i], u1)))

        u3 = list(map(list, zip(*[data1[:,i],u2])))
        df = pd.DataFrame(u3)
        s += df.corr()[0][1]
    s /= data1.shape[1]

    return s

class GAIndividual:

    def __init__(self, vardim, bound,x1):
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0
        self.x1 = x1

    def generate(self):
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in range(0, len):
            self.chrom[i] = self.bound[0][i] + \
                            (self.bound[1][i] - self.bound[0][i]) * rnd[i]

    def calculateFitness(self):
        self.fitness = Fun( self.chrom,self.x1)

class GeneticAlgorithm:

    def __init__(self, sizepop, vardim, bound, MAXGEN, params,x1):
        self.sizepop = sizepop
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.fitness = np.zeros((self.sizepop, 1))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params
        self.x1 = x1
        # self.i = i
    def initialize(self):

        for i in range(0, self.sizepop):
            ind = GAIndividual(self.vardim, self.bound,self.x1)
            ind.generate()
            self.population.append(ind)

    def evaluate(self):
        for i in range(self.sizepop):
            self.population[i].fitness = Fun(
                self.population[i].chrom,
                self.population[i].x1,
                low,
                ub
            )
            self.fitness[i] = self.population[i].fitness


    def solve(self):

        self.t = 0  
        self.initialize()  
        self.evaluate()  
        best = np.min(self.fitness)  
        bestIndex = np.argmin(self.fitness)  
        self.best = copy.deepcopy(self.population[bestIndex])
        self.avefitness = np.mean(self.fitness)  
        while (self.t < self.MAXGEN):
            self.t += 1
            self.selectionOperation()  
            self.crossoverOperation()  
            self.mutationOperation()  
            self.evaluate()  
            best = np.min(self.fitness)
            bestIndex = np.argmin(self.fitness)
            if best < self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.fitness)
            self.BEST.append(self.best)

        print("Optimal solution is:", int(self.best.chrom[0]), int(self.best.chrom[1]))
        result = [int(self.best.chrom[0]), int(self.best.chrom[1])]
        with open('C:/Users/Gursimranjeet Singh/REGCN/result/' + datasets + '_GA.csv', 'a',newline='', encoding='UTF8',) as f:
            d = csv.writer(f)
            d.writerow(result)

    def selectionOperation(self):
        '''
        selection operation for Genetic Algorithm
        '''
        newpop = []
        totalFitness = np.sum(self.fitness)
        accuFitness = np.zeros((self.sizepop, 1))

        
        sum1 = 0.
        for i in range(0, self.sizepop):
            accuFitness[i] = sum1 + self.fitness[i] / totalFitness
            sum1 = accuFitness[i]

    
        for i in range(0, self.sizepop):
            r = random.random()
            idx = 0
            for j in range(0, self.sizepop - 1):
                if j == 0 and r < accuFitness[j]:
                    idx = 0
                    break
                elif r >= accuFitness[j] and r < accuFitness[j + 1]:
                    idx = j + 1
                    break
            newpop.append(self.population[idx])
        self.population = newpop

    def crossoverOperation(self):
        '''
        crossover operation for genetic algorithm
        '''
        newpop = []

        for i in range(0, self.sizepop, 2):
            idx1 = random.randint(0, self.sizepop - 1)
            idx2 = random.randint(0, self.sizepop - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, self.sizepop - 1)
            newpop.append(copy.deepcopy(self.population[idx1]))
            newpop.append(copy.deepcopy(self.population[idx2]))
            r = random.random()

            if r < self.params[0]:
                crossPos = random.randint(1, self.vardim - 1)
                for j in range(crossPos, self.vardim):
                    newpop[i].chrom[j] = newpop[i].chrom[j] * self.params[2] + \
                                         (1 - self.params[2]) * newpop[i + 1].chrom[j]

                    newpop[i + 1].chrom[j] = newpop[i + 1].chrom[j] * self.params[2] + \
                                             (1 - self.params[2]) * newpop[i].chrom[j]
        self.population = newpop

    def mutationOperation(self):
        '''
        mutation operation for genetic algorithm
        '''
        newpop = []
        for i in range(0, self.sizepop):
            newpop.append(copy.deepcopy(self.population[i]))
            r = random.random()
            if r < self.params[1]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:

                    newpop[i].chrom[mutatePos] = newpop[i].chrom[mutatePos] - \
                                                 (newpop[i].chrom[mutatePos] - self.bound[0][mutatePos]) * \
                                                 (1 - random.random() ** (1 - self.t / self.MAXGEN))
                else:

                    newpop[i].chrom[mutatePos] = newpop[i].chrom[mutatePos] + \
                                                 (self.bound[1][mutatePos] - newpop[i].chrom[mutatePos]) * \
                                                 (1 - random.random() ** (1 - self.t / self.MAXGEN))
        self.population = newpop

if __name__ == "__main__":
    data = np.load(data_addr, allow_pickle=True)
    data = data.astype(float)
    print(data.shape)
    for i in range (data.shape[0]):
        tdata = data[i]
        train_size = int(tdata.shape[0] * 0.8)
        train_data = tdata[0:train_size]
        x1 = train_data
        best_K, best_alpha = coarse_grid_search(x1)

        low = [
            max(2, best_K - 1),
            max(100, best_alpha - 300)
        ]

        ub = [
            min(5, best_K + 1),
            best_alpha + 300
        ]

        bound = [low, ub]
        ga = GeneticAlgorithm(
            sizepop=10,    
            vardim=2,
            bound=bound,
            MAXGEN=20,    
            params=[0.9, 0.1, 0.5],
            x1=x1
        )

        ga.solve()



