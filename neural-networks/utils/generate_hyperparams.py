import itertools


def get_hyperparams_q1():
    num_epochs_range = [150, 500, 1000]
    initial_lr_range = [0.001, 0.005, 0.01]
    lr_decay_factor_range = [0.1, 0.2, 0.3, 1]  # 1 is no decay
    lr_decay_step_range = [100, 250, 350]

    all_combinations = list(
        itertools.product(
            num_epochs_range,
            initial_lr_range,
            lr_decay_factor_range,
            lr_decay_step_range,
        )
    )
    return all_combinations


def get_hyperparams_q2():
    num_epochs_range = [250, 550, 1100]
    initial_lr_range = [0.001, 0.005, 0.01]
    lr_decay_factor_range = [0.1, 0.2, 0.3, 1]
    lr_decay_step_range = [100, 250, 350]

    all_combinations = list(
        itertools.product(
            num_epochs_range,
            initial_lr_range,
            lr_decay_factor_range,
            lr_decay_step_range,
        )
    )
    return all_combinations


def get_hyperparams_q3():
    num_epochs_range = [250, 550, 1100]
    initial_lr_range = [0.001, 0.005, 0.01]
    lr_decay_factor_range = [0.1, 0.2, 0.3, 1]
    lr_decay_step_range = [100, 250, 350]
    regularization_range = [0, 0.001, 0.01, 0.1]  # 0 is dropout

    all_combinations = list(
        itertools.product(
            num_epochs_range,
            initial_lr_range,
            lr_decay_factor_range,
            lr_decay_step_range,
            regularization_range,
        )
    )
    return all_combinations


def get_hyperparams_q4():
    num_epochs_range = [1800, 2400, 3800]
    initial_lr_range = [0.001, 0.005, 0.01]

    all_combinations = list(
        itertools.product(
            num_epochs_range,
            initial_lr_range,
        )
    )
    return all_combinations
