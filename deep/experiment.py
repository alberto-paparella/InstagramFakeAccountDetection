# The main experiment runner for multilayer perceptron experiments
from deep.common import load_model, get_dataset_instafake, get_dataset_IJECE, get_dataset_combined, get_compatible_dataset


def run_experiment(folder, names, mode, n_iter=10):
    """
    Runs an experiment on certain models.
    :param folder: The base folder from which models need to be loaded up
    :param names: The names of the dataset, in the format [default, custom]
    :param mode: The mode of operation. Can be ijece, if, combo-par, combo-full, comp-if, comp-ijece
    :param n_iter: Number of iterations
    :return: results of the experiments.
    """
    if mode == "ijece":
        (default_train, default_test), (custom_train, custom_test) = get_dataset_IJECE()
    elif mode == "if":
        (default_train, default_test), (custom_train, custom_test) = get_dataset_instafake()
    # From this point forward, we don't have "default" models to compare against. Hence, default_test is set as None
    elif mode == "combo-par":
        (custom_train, custom_test) = get_dataset_combined(False)
        default_test = None
    elif mode == "combo-full":
        (custom_train, custom_test) = get_dataset_combined(True)
        default_test = None
    elif mode == "comp-if":
        (custom_train, custom_test) = get_compatible_dataset("if")
        default_test = None
    elif mode == "comp-ijece":
        (custom_train, custom_test) = get_compatible_dataset("ijece")
        default_test = None
    else:
        return
    default_model = None
    # Try to load the default model - if defined
    if mode != "combo-full" and mode != "combo-par" and mode != "comp-if" and mode != "comp-ijece":
        default_model = load_model(folder, names[0])
    custom_model = load_model(folder, names[1])
    # Build up the testing items
    items = [{"model": default_model, "test": default_test, "idx": 0},
             {"model": custom_model, "test": custom_test, "idx": 1}]
    print(f"Running deep learning experiments for {n_iter} times...")
    results = [{"accuracy": 0, "loss": 0, "precision": 0, "recall": 0, "f1": 0},
               {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0, "f1": 0}]
    for item in items:
        if not item["model"]:
            continue
        for i in range(n_iter):
            # Run experiments and compute metrics
            loss, acc, precision, recall = item["model"].evaluate(x=item["test"].iloc[:, :-1],
                                                                  y=item["test"].iloc[:, -1], verbose=0)
            results[item["idx"]]["accuracy"] += acc
            results[item["idx"]]["loss"] += loss
            results[item["idx"]]["precision"] += precision
            results[item["idx"]]["recall"] += recall
            results[item["idx"]]["f1"] += 2 * (precision * recall) / (precision + recall)
    # Compute averages and then return them
    for elem in ["accuracy", "precision", "recall", "f1"]:
        results[1][elem] = results[1][elem] / n_iter
        results[0][elem] = results[0][elem] / n_iter
        print(f"{elem} - Default: {results[0][elem]}; Custom: {results[1][elem]} ")
    print("=============================")
    return {'default': results[0], 'custom': results[1]}


if __name__ == "__main__":
    run_experiment("./deep/InstaFake/checkpoint",
                   ["INSTAFAKE_DEFAULT_1688545747.011179", "INSTAFAKE_CUSTOM_1688545747.011179"], "if")
