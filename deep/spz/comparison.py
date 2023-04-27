from spz_custom import model as custom_model
from spz_custom import validation as custom_validation
from spz_default import model as default_model
from spz_default import validation as default_validation

print(f"Custom model")
custom_model.evaluate(x=custom_validation.iloc[:, :-1], y=custom_validation.iloc[:, -1])
print(f"Default model")
default_model.evaluate(x=default_validation.iloc[:, :-1], y=default_validation.iloc[:, -1])
