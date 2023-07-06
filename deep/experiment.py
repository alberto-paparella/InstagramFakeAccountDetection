from deep.common import load_model, get_dataset_instafake, get_dataset_IJECE, get_dataset_combined


def run_experiment(folder, names, mode, n_iter=10):
    if mode == "ijece":
        (default_train, default_validation), (custom_train, custom_validation) = get_dataset_IJECE()
    elif mode == "if":
        (default_train, default_validation), (custom_train, custom_validation) = get_dataset_instafake()
    elif mode == "combo-par":
        (custom_train, custom_validation) = get_dataset_combined(False)
        default_validation = default_validation = None
    elif mode == "combo-full":
        (custom_train, custom_validation) = get_dataset_combined(True)
        default_validation = default_validation = None
    else:
        return
    default_model = None
    if mode != "combo-full" and mode != "combo-par":
        default_model = load_model(folder, names[0])
    custom_model = load_model(folder, names[1])
    items = [{"model": default_model, "validation": default_validation, "idx": 0},
             {"model": custom_model, "validation": custom_validation, "idx": 1}]
    print(f"Running deep learning experiments for {n_iter} times...")
    results = [{"Accuracy": 0, "Loss": 0, "Precision": 0, "Recall": 0, "F1": 0},
               {"Accuracy": 0, "Loss": 0, "Precision": 0, "Recall": 0, "F1": 0}]
    for item in items:
        if not item["model"]:
            continue
        for i in range(n_iter):
            loss, acc, precision, recall = item["model"].evaluate(x=item["validation"].iloc[:, :-1],
                                                                  y=item["validation"].iloc[:, -1], verbose=0)
            results[item["idx"]]["Accuracy"] += acc
            results[item["idx"]]["Loss"] += loss
            results[item["idx"]]["Precision"] += precision
            results[item["idx"]]["Recall"] += recall
            results[item["idx"]]["F1"] += 2 * (precision * recall) / (precision + recall)

    for elem in ["Accuracy", "Precision", "Recall", "F1"]:
        results[1][elem] = results[1][elem] / n_iter
        results[0][elem] = results[0][elem] / n_iter
        print(f"{elem} - Default: {results[0][elem]}; Custom: {results[1][elem]} ")
    print("=============================")
    return {'default': results[0], 'custom': results[1]}


if __name__ == "__main__":
    run_experiment("./deep/InstaFake/checkpoint",
                   ["INSTAFAKE_DEFAULT_1688545747.011179", "INSTAFAKE_CUSTOM_1688545747.011179"], "if")
