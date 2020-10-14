#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *

import random


class PlayerControllerHMM(PlayerControllerHMMAbstract):

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.Obs = []
        self.ObsT = []
        numhiddenstates = 2
        self.pi = [[1 / numhiddenstates for i in range(numhiddenstates)] for z in range(N_FISH)]
        self.A = [[[1 / numhiddenstates for i in range(numhiddenstates)] for j in range(numhiddenstates)]  for z in range(N_FISH)]
        self.B = [[[1 / N_EMISSIONS for i in range(N_EMISSIONS)] for j in range(numhiddenstates)] for z in range(N_FISH)]
        self.modelspecies = [None for i in range(N_SPECIES)]
        self.modelfish = [None for i in range(N_FISH)]
        self.currentfish = -1
        pass

    def transpose(self, matrix):
        return list(map(list, zip(*matrix)))

    def pickupfish(self):
        self.currentfish = self.currentfish + 1

    def bestguess(self):
        best = 0
        bestindex = 0
        for model, i in zip(self.modelspecies, range(N_SPECIES)):
            if model != None:
                f = HMM1()
                prob = f.forward_algorithm(model[0], model[1], model[2], self.ObsT[self.currentfish])
                if prob > best:
                    best = prob
                    bestindex = i

        return bestindex

    def isModelsempy(self):
        dif = True
        for model in self.modelspecies:
            if model != None:
                dif = False
        return dif

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        # This code would make a random guess on each step:
        # return (step % N_FISH, random.randint(0, N_SPECIES - 1))
        T = 90
        if (step <= T):
            self.Obs.append(observations)
        else:
            self.Obs.append(observations)
            self.ObsT = self.transpose(self.Obs)

            self.pickupfish()
            if(self.isModelsempy()==False):
                bestspecies = self.bestguess()
                return (self.currentfish,bestspecies)
            else:
                #Random guess
                return (self.currentfish,0)
        return None

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """

        if(correct == False):
            model = HMM3()
            newA, newB, newpi = model.run_learningalgorithm(self.A[fish_id], self.B[fish_id], self.pi[fish_id], self.ObsT[fish_id])
            self.modelspecies[true_type] = [newA,newB,newpi]
        pass

class HMM3():


    def __init__(self):
        self.xa = []

    def ln(self,x):
        n = 1000.0
        return n * ((x ** (1 / n)) - 1)

    def forward_algorithm(self,A,B,pi,Obs):
        c = [None for i in range(len(Obs))]
        c[0]  = 0
        alphas = [[None for i in range(len(A))] for j in range(len(Obs))]

        for i in range(len(A)):
            alphas[0][i] = pi[i]*B[i][Obs[0]]
            c[0] = c[0] + alphas[0][i]
        c[0] = 1/c[0]
        for i in range(len(A)):
            alphas[0][i] = c[0]*alphas[0][i]
        for t in range (1,len(Obs)):
            c[t]= 0
            for i in range(len(A)):
                alphas[t][i] = 0
                for j in range(len(A)):
                    alphas[t][i] = alphas[t][i] + alphas[t-1][j]*A[j][i]
                alphas[t][i] = alphas[t][i]*B[i][Obs[t]]

                c[t] = c[t] + alphas[t][i]
            c[t] = 1/c[t]
            for i in range(len(A)):
                alphas[t][i] = c[t]*alphas[t][i]
        return alphas, c

    def backward_algorithm(self,A,B,pi,Obs, c):
        betas = [[None for i in range(len(A))] for j in range(len(Obs))]
        for i in range(len(A)):
            betas[-1][i] = c[-1]
        for t in range(len(Obs)-2,0,-1):
            for i in range(len(A)):
                betas[t][i] = 0
                for j in range(len(A)):
                    betas[t][i] = betas[t][i] + A[i][j]*B[j][Obs[t+1]]*betas[t+1][j]
                betas[t][i] = c[t]*betas[t][i]
        return betas

    def gammas(self, A, B, alphas, betas, Obs):
        di_gammas = [[[None for i in range(len(A))] for j in range(len(A))] for t in range(len(Obs))]
        gammas = [[None for i in range(len(A))] for t in range(len(Obs))]
        for t in range(len(Obs)-1):
            for i in range(len(A)):
                gammas[t][i] = 0
                for j in range(len(A)):
                    di_gammas[t][i][j] = alphas[t][i]*A[i][j]*B[j][Obs[t+1]]*betas[t+1][j]
                    gammas[t][i] = gammas[t][i] + di_gammas[t][i][j]
        for i in range(len(A)):
            gammas[-1][i] = alphas[-1][i]
        return gammas, di_gammas


    def restimate_pi(self,gamma,A):
        new_pi =  [None for i in range(len(A))]
        for i in range(len(A)):
            new_pi[i] = gamma[0][i]
        return new_pi

    def restimate_A(self, A,Obs,gammas, digammas):

        for i in range(len(A)):
            denom = 0
            for t in range(len(Obs)-1):
                denom = denom + gammas[t][i]
            for j in range(len(A)):
                numer = 0
                for t in range(len(Obs)-1):
                    numer = numer + digammas[t][i][j]
                A[i][j] = numer/denom
        return A

    def restimate_B(self,A, B, Obs, gammas):
        for i in range(len(A)):
            denom = 0
            for t in range(len(Obs)):
                denom = denom + gammas[t][i]
            for j in range(len(B[0])):
                numer = 0
                for t in range(len(Obs)):
                    if Obs[t] == j:
                        numer = numer + gammas[t][i]
                B[i][j] = numer/denom
        return B


    def compute_log(self, c):
        logProb = 0
        for i in range(len(c)):
            logProb  = logProb + self.ln(c[i])
        logProb = -logProb
        return logProb

    def algorithm(self,A,B,pi, Obs):
        alphas, c = self.forward_algorithm(A, B, pi, Obs)
        betas = self.backward_algorithm(A, B, pi, Obs, c)
        gammas, digammas = self.gammas(A, B, alphas, betas, Obs)
        new_pi = self.restimate_pi(gammas,A)
        new_A = self.restimate_A(A, Obs, gammas, digammas)
        new_B = self.restimate_B(A, B, Obs, gammas)
        logProb = self.compute_log(c)
        return new_A,new_B,new_pi,logProb


    def create_output(self, A, B):
        outputA = []
        outputA.append(str(len(A)))
        outputA.append(str(len(A[0])))
        for i in range(len(A)):
            for j in range(len(A)):
                outputA.append(str(A[i][j]))
        seperator = " "
        h = seperator.join(map(str, outputA))
        print(h)
        outputB = []
        outputB.append(str(len(B)))
        outputB.append(str(len(B[0])))
        for i in range(len(B)):
            for j in range(len(B[0])):
                outputB.append(str(B[i][j]))
        seperator = " "
        bsep = seperator.join(map(str, outputB))
        print(bsep)

    def run_learningalgorithm(self, A, B, pi, Obs):
        maxIters = 30
        iters = 0
        oldLogProb = float('-inf')
        while (iters < maxIters):
            A, B, pi, logProb = self.algorithm(A, B, pi, Obs)
            iters = iters + 1
            if (iters < maxIters and logProb > oldLogProb):
                oldLogProb = logProb
            else:
                return A,B,pi

class HMM1():


    def __init__(self):
        self.xxxx = []


    def creation_matrix(self, rowA,columnA,line):
        #matrix does not have the dimensions. Clean matrix
        matrix = []

        for j in range(int(rowA)):
            row = []
            for i in range(int(columnA)):
                row.append(line[j * columnA + i])
            matrix.append(row)
        return matrix

    def transpose(self, matrix):
        return list(map(list, zip(*matrix)))

    def element_multiplication(self, matrix1, matrix2):
        return [a * b for a, b in zip(matrix1, matrix2)]
    def multiplication(self,matrix1,matrix2):
        result = self.empty_result(matrix1,matrix2)

        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                total = 0
                for ii in range(len(matrix1[0])):
                    total += matrix1[i][ii] * matrix2[ii][j]
                result[i][j] = total
        return result

    def empty_result(self,matrix1,matrix2):
        rowMatrix1 = len(matrix1)
        columnMatrix2 = len(matrix2[0])
        result = [0]*rowMatrix1*columnMatrix2
        matrixresult = self.creation_matrix(rowMatrix1, columnMatrix2, result)
        return matrixresult

    def forward_algorithm(self,A,B,pi,Obs):
        Btransposed = self.transpose(B)
        alpha1 = [self.element_multiplication(pi, Btransposed[Obs[0]])]
        current_alpha = alpha1
        for concrete_observation in Obs[1:]:
            new_pi = self.multiplication(current_alpha, A)
            current_alpha = [self.element_multiplication(new_pi[0], Btransposed[concrete_observation])]
        sum = 0
        for elem in current_alpha[0]:
            sum += elem
        return sum

