## Instructions

In order to run the code, do the following:

* To test each implementation on Iris, run:
  * For K-means, `run_iris_kmeans.py`
  * For KNNs, `run_iris_knns.py`
  * For Soft K-means, `run_iris_soft_kmeans.py`
* If you want to test if our implementation of Euclidean Distance, Cosine Distance, and PCA are correct, run `unit_tests.py`
* Finally, to run the algorithms on MNIST:
  * `run_mnist_kmeans.py --no-tuning` if you don't want to tune the hyperparameters. **Warning:** You may choose to search for the best hyper-parameters by excluding the argument, but it will take a while to run. Hyper-parameters are stored in `hyperparams_files/kmeans_hyperparams.csv` and the results are in `hyperparams_files/kmeans_hyperparams_results.csv`.
  * `run_mnist_knns.py --no-tuning` for KNNs; `hyperparams_files/knns_hyperparams.csv` for hyper-params; `hyperparams_files/knns_hyperparams_results.csv` for results.
  * `run_mnist_soft_kmeans.py --no-tuning` for Soft K-means; `hyperparams_files/soft_kmeans_hyperparams.csv` for hyper-params; `hyperparams_files/soft_kmeans_hyperparams_results.csv` for results.
