"""
Question 7: We know the algorithm has converged when the logProb it's not higher than the previous one. We want it to be
higher, because we want that P(Observations|model) increases in every iteration. We want the model that makes the sequence
of observations the most likely. It never converges, it only gets better and better (that means that the next probability
that we will find will be higher than the previous one


"""

import sys
import matplotlib.pyplot as plt
import math

class HMM():


    def __init__(self):
        self.xxxx = []

    def parse_file(self):
        vector = []
        for line in sys.stdin:
            vector.append(line)
        return vector

    def parse_lines(self, vector):
        lines = []

        for x in vector:
            current_line = []
            current_word = ""
            for char in x:
                if (char == " " or char == '\n') and current_word !="":
                    current_line.append(float(current_word))
                    current_word = ""
                else:
                    current_word += char

            if current_word != "" and current_word!="\n":
                current_line.append(float(current_word))

            lines.append(current_line)
        return lines
    def parse_index(self,lines):
        rowA = int(lines[0][0])
        columnA = int(lines[0][1])
        rowB = int(lines[1][0])
        columnB = int(lines[1][1])
        rowPi = int(lines[2][0])
        columnPi = int(lines[2][1])
        return [[rowA,columnA],[rowB,columnB],[rowPi,columnPi]]
    def creation_matrix(self, rowA,columnA,line):
        matrix = []

        for j in range(int(rowA)):
            row = []
            for i in range(int(columnA)):
                row.append(line[j * columnA + i])
            matrix.append(row)
        return matrix

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
                    if A[i][j] != 0:
                        print("A",A[i][j] )
                        print("B",B[j][Obs[t+1]] )
                        print("ALPHAS",alphas[t][i] )
                        print("BETAS", betas[t+1][j])
                    di_gammas[t][i][j] = alphas[t][i]*A[i][j]*B[j][Obs[t+1]]*betas[t+1][j]
                    gammas[t][i] = gammas[t][i] + di_gammas[t][i][j]
        for i in range(len(A)):
            gammas[-1][i] = alphas[-1][i]

        return gammas, di_gammas



    def restimate_pi(self,gamma):
        new_pi =  [None for i in range(len(A))]
        for i in range(len(A)):
            new_pi[i] = gamma[0][i]
        return new_pi

    def restimate_A(self, A,Obs,gammas, digammas):
        #print("DIGAMMAS", digammas)
        print("GAMMAS", gammas)
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
            logProb  = logProb + Hmm3.ln(c[i])
        logProb = -logProb
        return logProb

    def algorithm(self,A,B,pi, Obs):
        alphas, c = Hmm3.forward_algorithm(A, B, pi, Obs)
        betas = Hmm3.backward_algorithm(A, B, pi, Obs, c)
        gammas, digammas = Hmm3.gammas(A, B, alphas, betas, Obs)
        new_pi = Hmm3.restimate_pi(gammas)
        new_A = Hmm3.restimate_A(A, Obs, gammas, digammas)
        new_B = Hmm3.restimate_B(A, B, Obs, gammas)
        logProb = Hmm3.compute_log(c)
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
    def element_multiplication(self, matrix1, matrix2):
        return [a * b for a, b in zip(matrix1, matrix2)]


if __name__ == "__main__":
    Hmm3 = HMM()
    vector = Hmm3.parse_file()
    lines = Hmm3.parse_lines(vector)
    index = Hmm3.parse_index(lines)
    A = Hmm3.creation_matrix(index[0][0], index[0][1], lines[0][2:])
    B = Hmm3.creation_matrix(index[1][0], index[1][1], lines[1][2:])
    pi = Hmm3.creation_matrix(index[2][0], index[2][1], lines[2][2:])
    pi = pi[0]
    Obs = lines[3][1:]
    for i in range(len(Obs)):
        Obs[i] = int(Obs[i])
    maxIters = 3000
    iters = 0
    oldLogProb = float('-inf')
    log = []
    while(iters<maxIters):
        A,B,pi,logProb = Hmm3.algorithm(A,B,pi,Obs)
        log.append(logProb)
        iters = iters + 1
        if(iters<maxIters and logProb>oldLogProb):
            oldLogProb = logProb
        else:
            Hmm3.create_output(A,B)
            plt.plot(log)
            plt.ylabel('log')
            plt.xlabel('iterations')
            plt.title("Convergence with the sequence of 1000 observations")
            plt.show()
            break
