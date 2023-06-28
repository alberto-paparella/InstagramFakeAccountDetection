from IJECE_custom import run_model as run_custom_model
from IJECE_default import run_model as run_default_model
from deep.common import get_dataset_IJECE

eval_steps = 10
results = {"custom": {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0},
           "default": {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0}}
(default_train, default_validation), (custom_train, custom_validation) = get_dataset_IJECE()

custom_model = run_custom_model(custom_train)
default_model = run_default_model(default_train)

print(f"Now evaluating custom model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc, precision, recall = custom_model.evaluate(x=custom_validation.iloc[:, :-1], y=custom_validation.iloc[:, -1], verbose=0)
    results["custom"]["accuracy"] += acc
    results["custom"]["loss"] += loss
    results["custom"]["precision"] += precision
    results["custom"]["recall"] += recall
print(f"Now evaluating default model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc, precision, recall = default_model.evaluate(x=default_validation.iloc[:, :-1], y=default_validation.iloc[:, -1], verbose=0)
    results["default"]["accuracy"] += acc
    results["default"]["loss"] += loss
    results["default"]["precision"] += precision
    results["default"]["recall"] += recall
for elem in ["accuracy", "loss", "precision", "recall"]:
    print(f"{elem}: Custom {results['custom'][elem]/eval_steps} , Default: Custom {results['default'][elem]/eval_steps}")
