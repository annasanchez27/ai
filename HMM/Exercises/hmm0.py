import sys


class HMM():


    def __init__(self):
        self.xxxx = []

    def creation_matrix(self, rowA,columnA,line):
        matrix = []
        for j in range(int(rowA)):
            row = []
            for i in range(int(columnA)):
                row.append(line[j * columnA + i])
            matrix.append(row)
        return matrix

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

    def create_output(self,result):
        row = len(result)
        column = len(result[0])
        output = [str(row), str(column)]
        for i in range(row):
            for j in range(column):
                output.append(str(result[i][j]))
        seperator = " "
        h = seperator.join(map(str, output))
        print(h)

if __name__ == "__main__":
    Hmm0 = HMM()
    vector = Hmm0.parse_file()
    lines = Hmm0.parse_lines(vector)
    index = Hmm0.parse_index(lines)
    A = Hmm0.creation_matrix(index[0][0], index[0][1], lines[0][2:])
    B = Hmm0.creation_matrix(index[1][0], index[1][1], lines[1][2:])
    pi = Hmm0.creation_matrix(index[2][0], index[2][1], lines[2][2:])
    print("PI ", pi)
    pxt = Hmm0.multiplication(pi,A)
    pobservation = Hmm0.multiplication(pxt,B)
    Hmm0.create_output(pobservation)
