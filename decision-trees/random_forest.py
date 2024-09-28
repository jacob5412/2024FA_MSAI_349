"""
Implementation of the random forest algorithm.
"""

import random

import ID3


class RandomForest:
    """
    Random Forest - an ensemble learning technique that builds multiple trees.
    Here, we're building trees using bootstrapping (sampling with replacement)

    Attributes:
        random_forest_nodes (list): A list to store the decision tree nodes
                                     in the random forest.
        num_trees (int): The number of decision trees in the random forest.
    """

    def __init__(self, num_trees):
        random.seed(101)
        self.random_forest_nodes = []
        self.num_trees = num_trees

    def fit(self, examples):
        """
        Fits the random forest to a dataset using bootstrapped samples and
        creates decision trees.
        """
        for _ in range(self.num_trees):
            # create a bootstrapped sample by randomly selecting an example n
            # times; with replacement, i.e., can select one example more than
            # once; n = len(examples)
            bootstrap_sample = [random.choice(examples) for _ in range(len(examples) + 1)]
            available_attributes = [
                attribute for attribute in examples[0].keys() if attribute != "Class"
            ]
            subset_attributes = set(
                random.sample(
                    available_attributes,
                    random.randint(2, len(available_attributes)),
                )
            )
            random_forest_node = ID3.ID3_helper(bootstrap_sample, subset_attributes)
            self.random_forest_nodes.append(random_forest_node)

    def test(self, examples):
        """
        Tests the accuracy of the random forest on a dataset.
        """
        num_correct_predictions = sum(
            [self.evaluate(example) == example["Class"] for example in examples]
        )
        return num_correct_predictions / len(examples)

    def evaluate(self, example):
        """
        Evaluates a single example using the random forest's ensemble of
        decision trees. Use majority voting to predict the class.
        """
        predictions = [
            ID3.evaluate(random_forest_node, example)
            for random_forest_node in self.random_forest_nodes
        ]
        return ID3.get_most_common_class(predictions)
