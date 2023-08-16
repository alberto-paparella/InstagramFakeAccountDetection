# This file is used only in development, an allows the training and comparison of neural network models.
import datetime

from combined_partial import run_model as run_partial_model
from combined_full import run_model as run_full_model
from deep.common import get_dataset_combined, train_save

eval_steps = 10
results = {"custom": {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0},
           "default": {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0}}
(partial_train, partial_test) = get_dataset_combined(False)
(full_train, full_test) = get_dataset_combined(True)
timestamp = datetime.datetime.now().timestamp()

combined_partial_model = train_save("COMBO_PART", partial_train, run_partial_model, "./deep/combined/checkpoint",
                                    timestamp)
combined_full_model = train_save("COMBO_FULL", full_train, run_full_model, "./deep/combined/checkpoint",
                                 timestamp)

print(f"Now evaluating full model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc, precision, recall = combined_partial_model.evaluate(x=partial_test.iloc[:, :-1],
                                                                   y=partial_test.iloc[:, -1], verbose=0)
    results["custom"]["accuracy"] += acc
    results["custom"]["loss"] += loss
    results["custom"]["precision"] += precision
    results["custom"]["recall"] += recall
print(f"Now evaluating partial model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc, precision, recall = combined_full_model.evaluate(x=full_test.iloc[:, :-1],
                                                                y=full_test.iloc[:, -1], verbose=0)
    results["default"]["accuracy"] += acc
    results["default"]["loss"] += loss
    results["default"]["precision"] += precision
    results["default"]["recall"] += recall
for elem in ["accuracy", "loss", "precision", "recall"]:
    print(
        f"{elem}: Partial {results['custom'][elem] / eval_steps} , Full: {results['default'][elem] / eval_steps}")
