# This file is used only in development, an allows the training and comparison of neural network models.
import datetime

from instafake_custom import run_model as run_custom_model
from instafake_default import run_model as run_default_model
from deep.common import get_dataset_instafake, train_save

eval_steps = 10
custom_acc = 0
def_acc = 0

results = {"custom": {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0},
           "default": {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0}}

(default_train, default_val), (custom_train, custom_val) = get_dataset_instafake()

timestamp = datetime.datetime.now().timestamp()

custom_model = train_save("INSTAFAKE_CUSTOM", custom_train, run_custom_model, "./deep/InstaFake/checkpoint", timestamp)
default_model = train_save("INSTAFAKE_DEFAULT", default_train, run_default_model, "./deep/InstaFake/checkpoint", timestamp)

print(f"Now evaluating custom model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc, precision, recall = custom_model.evaluate(x=custom_val.iloc[:, :-1],
                                                         y=custom_val.iloc[:, -1], verbose=0)
    results["custom"]["accuracy"] += acc
    results["custom"]["loss"] += loss
    results["custom"]["precision"] += precision
    results["custom"]["recall"] += recall
print(f"Now evaluating default model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc, precision, recall = default_model.evaluate(x=default_val.iloc[:, :-1],
                                                          y=default_val.iloc[:, -1], verbose=0)
    results["default"]["accuracy"] += acc
    results["default"]["loss"] += loss
    results["default"]["precision"] += precision
    results["default"]["recall"] += recall

for elem in ["accuracy", "loss", "precision", "recall"]:
    print(f"{elem}: Custom {results['custom'][elem]/eval_steps} , Default: {results['default'][elem]/eval_steps}")
