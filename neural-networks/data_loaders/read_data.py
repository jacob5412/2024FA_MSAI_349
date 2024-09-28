import numpy as np


def read_mnist(file_path):
    data_set = []
    with open(file_path, "rt") as f:
        for line in f:
            line = line.replace("\n", "")
            tokens = line.split(",")
            attribs = []
            for i in range(785):
                attribs.append(int(tokens[i]))
            data_set.append(np.array(attribs))
    return np.array(data_set)


def read_insurability(file_path):
    count = 0
    data = []
    with open(file_path, "rt") as f:
        for line in f:
            if count > 0:
                line = line.replace("\n", "")
                tokens = line.split(",")
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == "Good":
                        cls = 0
                    elif tokens[3] == "Neutral":
                        cls = 1
                    else:
                        cls = 2
                    data.append(np.array([cls, x1, x2, x3]))
            count = count + 1
    return np.array(data)
