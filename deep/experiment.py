from common import load_model, get_dataset_spz, get_dataset_IJECE, get_dataset_combined


def run_experiment(folder, names, mode, n_iter=10):
    if mode == "ijece":
        (default_train, default_validation), (custom_train, custom_validation) = get_dataset_IJECE()
    elif mode == "if":
        (default_train, default_validation), (custom_train, custom_validation) = get_dataset_spz()
    elif mode == "combo":
        (default_train, default_validation) = get_dataset_combined(False)
        (custom_train, custom_validation) = get_dataset_combined(True)
    else:
        return
    default_model = load_model(folder, names[0])
    custom_model = load_model(folder, names[1])
    items = [{"model": default_model, "validation": default_validation, "idx":0},
             {"model": custom_model, "validation": custom_validation, "idx":1}]
    print(f"Evaluating models for {n_iter} times...")
    results = [{"accuracy": 0, "loss": 0, "precision": 0, "recall": 0, "f1":0},
               {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0, "f1":0}]
    for item in items:
        for i in range(n_iter):
            loss, acc, precision, recall = item["model"].evaluate(x=item["validation"].iloc[:, :-1],
                                                                  y=item["validation"].iloc[:, -1], verbose=0)
            results[item["idx"]]["accuracy"] += acc
            results[item["idx"]]["loss"] += loss
            results[item["idx"]]["precision"] += precision
            results[item["idx"]]["recall"] += recall
            results[item["idx"]]["f1"] += 2*(precision*recall)/(precision+recall)

    for elem in ["accuracy", "loss", "precision", "recall", "f1"]:
        print(f"{elem}: Custom {results[1][elem]/n_iter} , Default: {results[0][elem]/n_iter}")


if __name__ == "__main__":
    run_experiment("./deep/spz/checkpoint", ["SPZ_DEFAULT_1688545747.011179","SPZ_CUSTOM_1688545747.011179"], "if")
