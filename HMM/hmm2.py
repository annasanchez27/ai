import sys


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

    def transpose(self, matrix):
        return list(map(list, zip(*matrix)))


    def element_multiplication(self, matrix1, matrix2):
        return [a * b for a, b in zip(matrix1, matrix2)]

    def find_probabilities(self,A,B,pi,Obs):
        for i in range(len(Obs)):
            Obs[i] = int(Obs[i])
        Btransposed = Hmm2.transpose(B)
        Atransposed = Hmm2.transpose(A)
        delta1 = Hmm2.element_multiplication(pi[0], Btransposed[Obs[0]])
        new_delta = Hmm2.creation_matrix(index[0][0], index[0][1], [0] * index[0][0] * index[0][1])
        max_delta = [0] * index[0][0]
        final = []
        finalindex = []
        final.append(delta1)
        for current_obs in Obs[1:]:
            indexlist = []
            for j in range(index[0][0]):
                for i in range(index[0][1]):
                    new_delta[j][i] = delta1[i] * Atransposed[j][i] * Btransposed[current_obs][j]
            for i in range(index[0][0]):
                max_delta[i] = max(new_delta[i])
                indexlist.append(new_delta[i].index(max_delta[i]))

            delta1 = max_delta
            final.append(delta1[:])
            finalindex.append(indexlist)
        return final,finalindex

    def get_state(self, position, liststates, accu):
        newstate = liststates[-1][position]
        liststates.pop()
        accu.append(newstate)
        if len(liststates)==0:
            return accu
        else:
            self.get_state(newstate,liststates,accu)
            return accu




if __name__ == "__main__":
    Hmm2 = HMM()
    vector = Hmm2.parse_file()
    lines = Hmm2.parse_lines(vector)
    index = Hmm2.parse_index(lines)
    A = Hmm2.creation_matrix(index[0][0], index[0][1], lines[0][2:])
    B = Hmm2.creation_matrix(index[1][0], index[1][1], lines[1][2:])
    pi = Hmm2.creation_matrix(index[2][0], index[2][1], lines[2][2:])
    Obs = lines[3][1:]
    probabilities, stateslist = Hmm2.find_probabilities(A,B,pi,Obs)
    #Get the state with the maximum probability
    a = max(probabilities[-1])
    accu = []
    laststate = probabilities[-1].index(a)
    accu.append(laststate)
    #Need to find the state associated to this probability
    position = stateslist[-1][laststate]
    accu.append(position)
    stateslist.pop()
    list = Hmm2.get_state(position, stateslist, accu)
    list.reverse()
    seperator = " "
    h = seperator.join(map(str, list))
    print(h)

