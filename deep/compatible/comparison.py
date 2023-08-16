# This file is used only in development, an allows the training and comparison of neural network models.
import datetime

from compatible_ijece import run_model as run_ijece_comp_model
from compatible_if import run_model as run_if_comp_model
from deep.common import get_compatible_dataset, train_save

eval_steps = 10
results = {"custom": {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0},
           "default": {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0}}
(ijece_train, ijece_val) = get_compatible_dataset("ijece")
(if_train, if_val) = get_compatible_dataset("if")
timestamp = datetime.datetime.now().timestamp()

ijece_model = train_save("COMP_IJECE", ijece_train, run_ijece_comp_model, "./deep/compatible/checkpoint",
                         timestamp)
if_model = train_save("COMP_INSTAFAKE", if_train, run_if_comp_model, "./deep/compatible/checkpoint",
                      timestamp)

print(f"Now evaluating ijece model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc, precision, recall = ijece_model.evaluate(x=ijece_val.iloc[:, :-1],
                                                        y=ijece_val.iloc[:, -1], verbose=0)
    results["custom"]["accuracy"] += acc
    results["custom"]["loss"] += loss
    results["custom"]["precision"] += precision
    results["custom"]["recall"] += recall
print(f"Now evaluating instafake model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc, precision, recall = if_model.evaluate(x=if_val.iloc[:, :-1],
                                                     y=if_val.iloc[:, -1], verbose=0)
    results["default"]["accuracy"] += acc
    results["default"]["loss"] += loss
    results["default"]["precision"] += precision
    results["default"]["recall"] += recall
for elem in ["accuracy", "loss", "precision", "recall"]:
    print(
        f"{elem}: IF {results['custom'][elem] / eval_steps} , IJECE: {results['default'][elem] / eval_steps}")
