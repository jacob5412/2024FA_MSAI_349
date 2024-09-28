"""
Unit Tests
"""

import argparse
import math
import random

import ID3
import matplotlib.pyplot as plt
import parse
from random_forest import RandomForest


def plot_random_forest_id3_accuracies(
    training_sizes,
    avg_test_accuracy_id3_pruning,
    avg_test_accuracy_id3_not_pruning,
    avg_test_accuracy_rf,
    avg_train_accuracy_id3_pruning,
    avg_train_accuracy_id3_not_pruning,
    avg_train_accuracy_rf,
    dataset_name,
):
    """
    Plot learning curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        training_sizes,
        avg_test_accuracy_id3_pruning,
        label="[Test] ID3 (With Pruning)",
        color="orange",
        marker="o",
    )
    plt.plot(
        training_sizes,
        avg_test_accuracy_id3_not_pruning,
        label="[Test] ID3 (Without Pruning)",
        color="blue",
        marker="o",
    )
    plt.plot(
        training_sizes,
        avg_test_accuracy_rf,
        label="[Test] Random Forest",
        color="green",
        marker="o",
    )
    plt.plot(
        training_sizes,
        avg_train_accuracy_id3_pruning,
        label="[Train] ID3 (With Pruning)",
        color="orange",
        linestyle="--",
        marker="o",
    )
    plt.plot(
        training_sizes,
        avg_train_accuracy_id3_not_pruning,
        label="[Train] ID3 (Without Pruning)",
        color="blue",
        linestyle="--",
        marker="o",
    )
    plt.plot(
        training_sizes,
        avg_train_accuracy_rf,
        label="[Train] Random Forest",
        color="green",
        linestyle="--",
        marker="o",
    )
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Average Accuracy")
    plt.title(
        f"Learning Curves for {dataset_name} Data - ID3 (With and Without Pruning) vs Random Forest"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(f"images/id3_vs_rf_{dataset_name}.png", bbox_inches="tight")
    print(f"Saved image as images/id3_vs_rf_{dataset_name}.png")


def plot_num_trees_accuracies(accuracies_dict):
    """
    Plot number of trees against the accuracies.
    """
    trees = list(accuracies_dict.keys())
    accuracy = list(accuracies_dict.values())

    plt.figure(figsize=(10, 6))
    plt.plot(trees, accuracy, marker="o", linestyle="-")
    plt.title("Random Forest Accuracy vs. Number of Trees")
    plt.xlabel("Number of Trees")
    plt.ylabel("Average Accuracy on Test Data")
    plt.grid(True)
    plt.savefig("images/rf_num_trees_accuracies.png", bbox_inches="tight")


def get_best_num_trees_for_random_forest(filename, plot=True):
    """
    Find the best number of trees for random forest.
    """
    data = parse.parse(filename)
    num_trees_accuracies = {}

    for num_trees in range(2, 15):
        accuracies = []
        for _ in range(25):
            random.shuffle(data)
            split_index = int(0.8 * len(data))
            train = data[:split_index]
            test = data[split_index:]

            random_forest = RandomForest(num_trees)
            random_forest.fit(train)
            acc = random_forest.test(test)
            accuracies.append(acc)
        num_trees_accuracies[num_trees] = sum(accuracies) / len(accuracies)
    if plot:
        plot_num_trees_accuracies(num_trees_accuracies)
    return max(num_trees_accuracies, key=lambda key: num_trees_accuracies[key])


def learning_curves_random_forest_id3(filename, training_sizes, num_trees, plot=True):
    """
    Plot learning curves of RF and ID3 (with and without pruning)
    """
    data = parse.parse(filename)
    avg_test_accuracy_id3_pruning = []
    avg_test_accuracy_id3_not_pruning = []
    avg_test_accuracy_rf = []
    avg_train_accuracy_id3_pruning = []
    avg_train_accuracy_id3_not_pruning = []
    avg_train_accuracy_rf = []

    for train_size in training_sizes:
        with_pruning_acc_test = []
        without_pruning_acc_test = []
        rf_acc_test = []
        with_pruning_acc_train = []
        without_pruning_acc_train = []
        rf_acc_train = []

        for _ in range(25):
            random.shuffle(data)
            validation_size = max(1, math.ceil((int(train_size / 0.8) - train_size) // 2))
            test_size = validation_size
            train = data[:train_size]
            valid = data[train_size : train_size + validation_size]
            test = data[
                train_size + validation_size : train_size + validation_size + test_size
            ]

            tree = ID3.ID3(train, 0)
            acc = ID3.test(tree, train)
            with_pruning_acc_train.append(acc)
            ID3.prune(tree, valid)
            acc = ID3.test(tree, test)
            with_pruning_acc_test.append(acc)

            tree = ID3.ID3(train + valid, 0)
            acc = ID3.test(tree, train + valid)
            without_pruning_acc_train.append(acc)
            acc = ID3.test(tree, test)
            without_pruning_acc_test.append(acc)

            random_forest = RandomForest(num_trees)
            random_forest.fit(train + valid)
            acc = random_forest.test(train + valid)
            rf_acc_train.append(acc)
            acc = random_forest.test(test)
            rf_acc_test.append(acc)

        avg_test_accuracy_id3_pruning.append(
            sum(with_pruning_acc_test) / len(with_pruning_acc_test)
        )
        avg_test_accuracy_id3_not_pruning.append(
            sum(without_pruning_acc_test) / len(without_pruning_acc_test)
        )
        avg_test_accuracy_rf.append(sum(rf_acc_test) / len(rf_acc_test))
        avg_train_accuracy_id3_pruning.append(
            sum(with_pruning_acc_train) / len(with_pruning_acc_train)
        )
        avg_train_accuracy_id3_not_pruning.append(
            sum(without_pruning_acc_train) / len(without_pruning_acc_train)
        )
        avg_train_accuracy_rf.append(sum(rf_acc_train) / len(rf_acc_train))
    if plot:
        plot_random_forest_id3_accuracies(
            training_sizes,
            avg_test_accuracy_id3_pruning,
            avg_test_accuracy_id3_not_pruning,
            avg_test_accuracy_rf,
            avg_train_accuracy_id3_pruning,
            avg_train_accuracy_id3_not_pruning,
            avg_train_accuracy_rf,
            "Candy",
        )


def compare_accuracies_id3_rf(filename, num_trees):
    """
    Compare the accuracies of the best RF against ID3 (with and
    without pruning).
    """
    data = parse.parse(filename)
    with_pruning_acc = []
    without_pruning = []
    rf_acc = []

    for _ in range(25):
        random.shuffle(data)
        train = data[: len(data) // 2]
        valid = data[len(data) // 2 : 3 * len(data) // 4]
        test = data[3 * len(data) // 4 :]

        tree = ID3.ID3(train, 0)
        ID3.prune(tree, valid)
        acc = ID3.test(tree, test)
        with_pruning_acc.append(acc)

        tree = ID3.ID3(train + valid, 0)
        acc = ID3.test(tree, test)
        without_pruning.append(acc)

        random_forest = RandomForest(num_trees)
        random_forest.fit(train + valid)
        acc = random_forest.test(test)
        rf_acc.append(acc)
    print(
        "Avg accuracy of ID3 with pruning: ",
        sum(with_pruning_acc) / len(with_pruning_acc),
    )
    print(
        "Avg accuracy of ID3 without pruning: ",
        sum(without_pruning) / len(without_pruning),
    )
    print(
        "Avg accuracy of RF: ",
        sum(rf_acc) / len(rf_acc),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves.")
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Set to False to disable plotting",
    )
    args = parser.parse_args()
    if not args.plot:
        print("Plotting is disabled.")

    random.seed(101)
    best_num_trees = get_best_num_trees_for_random_forest("candy.data", args.plot)
    learning_curves_random_forest_id3(
        "candy.data", range(5, 68, 4), best_num_trees, args.plot
    )
    compare_accuracies_id3_rf("candy.data", best_num_trees)
