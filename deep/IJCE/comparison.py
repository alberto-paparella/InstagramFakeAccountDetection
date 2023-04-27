from IJCE_custom import model as custom_model
from IJCE_custom import validation as custom_validation
from IJCE_default import model as default_model
from IJCE_default import validation as default_validation

print(f"Custom model")
custom_model.evaluate(x=custom_validation.iloc[:, :-1], y=custom_validation.iloc[:, -1])
print(f"Default model")
default_model.evaluate(x=default_validation.iloc[:, :-1], y=default_validation.iloc[:, -1])
