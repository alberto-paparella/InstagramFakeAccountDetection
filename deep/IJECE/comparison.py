from IJECE_custom import run_model as run_custom_model
from IJECE_default import run_model as run_default_model
from deep.common import get_dataset_IJECE, train_save

eval_steps = 10
results = {"custom": {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0},
           "default": {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0}}
(default_train, default_validation), (custom_train, custom_validation) = get_dataset_IJECE()


custom_model = train_save("IJECE_CUSTOM", custom_train, run_custom_model ,"./deep/IJECE/checkpoint")
default_model = train_save("IJECE_DEFAULT", default_train, run_default_model, "./deep/IJECE/checkpoint")

print(f"Now evaluating custom model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc, precision, recall = custom_model.evaluate(x=custom_validation.iloc[:, :-1],
                                                         y=custom_validation.iloc[:, -1], verbose=0)
    results["custom"]["accuracy"] += acc
    results["custom"]["loss"] += loss
    results["custom"]["precision"] += precision
    results["custom"]["recall"] += recall
print(f"Now evaluating default model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc, precision, recall = default_model.evaluate(x=default_validation.iloc[:, :-1],
                                                          y=default_validation.iloc[:, -1], verbose=0)
    results["default"]["accuracy"] += acc
    results["default"]["loss"] += loss
    results["default"]["precision"] += precision
    results["default"]["recall"] += recall
for elem in ["accuracy", "loss", "precision", "recall"]:
    print(
        f"{elem}: Custom {results['custom'][elem] / eval_steps} , Default: {results['default'][elem] / eval_steps}")
