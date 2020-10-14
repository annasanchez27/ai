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
        for i in range(len(Obs)):
            Obs[i] = int(Obs[i])
        Btransposed = Hmm1.transpose(B)
        alpha1 = [Hmm1.element_multiplication(pi[0], Btransposed[Obs[0]])]
        current_alpha = alpha1
        for concrete_observation in Obs[1:]:
            new_pi = Hmm1.multiplication(current_alpha, A)
            current_alpha = [Hmm1.element_multiplication(new_pi[0], Btransposed[concrete_observation])]
        sum = 0
        for elem in current_alpha[0]:
            sum += elem
        print(sum)
if __name__ == "__main__":
    Hmm1 = HMM()
    vector = Hmm1.parse_file()

    lines = Hmm1.parse_lines(vector)
    index = Hmm1.parse_index(lines)
    A = Hmm1.creation_matrix(index[0][0], index[0][1], lines[0][2:])
    B = Hmm1.creation_matrix(index[1][0], index[1][1], lines[1][2:])
    pi = Hmm1.creation_matrix(index[2][0], index[2][1], lines[2][2:])
    Obs = lines[3][1:]

    Hmm1.forward_algorithm(A,B,pi,Obs)

