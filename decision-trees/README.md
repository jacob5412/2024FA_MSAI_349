## Instructions

Python Version(s):
1. `anaconda3-2023.07-0` OR
2. `Python3.11`

### File Descriptions

1. `candy.data`: Determine a class of candy based on attributes.
2. `tennis.data`: Will someone play tennis based on the weather?
3. `house_votes_84.data`: House of representative votes (1984).

### Running the Project

1. We then run the `mini_auto_grader.py` file to test out the code we've written; this tests the `ID3.py` file, which runs our implementation of the ID3 algorithm.
2. Then, we run the `unit_tests.py` file to test our code against the house, tennis, and candy datasets. To debug the code, we used `tennis.data` and `candy.data`; these are implemented through additional functions.
3. Correspondingly, we run `plot_learning_curves.py` to plot the learning curves for house, candy, and tennis data. Here, we plot the learning curve for a pruned and non-pruned decision tree based on a cross-validated dataset. These plots are, in turn, stored in the __images__ folder. 
4. For the random_forest problem, we run the `tune_random_forest.py` to figure out the best number of trees for a random forest algorithm. The algorithm is stored in `random_forest.py`, which fits the random forest to a dataset using bootstrapped samples and creates decision trees. The Random Forest classifier runs on the `candy.data` dataset and then compares the results of the single decision tree constructed by the ID3 algorithm.
