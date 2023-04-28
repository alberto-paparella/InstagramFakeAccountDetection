from spz_custom import run_model as run_custom_model
from spz_default import run_model as run_default_model
from deep.common import get_dataset_spz

eval_steps = 10
custom_acc = 0
def_acc = 0
(default_train, default_validation), (custom_train, custom_validation) = get_dataset_spz()

custom_model = run_custom_model(custom_train)
default_model = run_default_model(default_train)

print(f"Now evaluating custom model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc = custom_model.evaluate(x=custom_validation.iloc[:, :-1], y=custom_validation.iloc[:, -1], verbose=0)
    custom_acc += acc
print(f"Now evaluating default model {eval_steps} times...")
for i in range(eval_steps):
    loss, acc = default_model.evaluate(x=default_validation.iloc[:, :-1], y=default_validation.iloc[:, -1], verbose=0)
    def_acc += acc
print(f"Custom accuracy is {custom_acc/eval_steps}, Default accuracy is {def_acc/eval_steps}")