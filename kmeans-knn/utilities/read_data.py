"""
Utilities to read data from a CSV.
"""


def read_data(file_name):
    """
    Read data from a CSV file and return it as a list of label-feature pairs.

    Args:
        file_name (str): The name of the CSV file to read.

    Returns:
        list: A list of label-feature pairs.
    """
    data_set = []
    with open(file_name, "rt") as f:
        for line in f:
            line = line.replace("\n", "")
            tokens = line.split(",")
            label = tokens[0]
            attribs = []
            for i in range(len(tokens) - 2):
                attribs.append(tokens[i + 1])
            data_set.append([label, attribs])
    return data_set


def get_numerical_labels(data_set, data_type=int):
    """
    Extract numerical labels from a list of label-feature pairs.

    Args:
        data_set (list): A list of label-feature pairs.
        data_type (type): The data type to cast the labels to (int or float).

    Returns:
        list: A list of numerical labels.
    """
    labels = [data_type(row[0]) for row in data_set]
    return labels


def get_numerical_features(data_set, data_type=int):
    """
    Extract numerical features from a list of label-feature pairs.

    Args:
        data_set (list): A list of label-feature pairs.
        data_type (type): The data type to cast the labels to (int or float).

    Returns:
        list: A list of numerical features.
    """
    features = [[data_type(datapoint) for datapoint in row[1]] for row in data_set]
    return features
