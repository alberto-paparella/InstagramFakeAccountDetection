# Caricare il dataset nel formato .csv
# Caricare il dataset nel formato parziale
from dataset.normalizer import csv_importer_full
from sequoia_comparison.utils import find_demarcator, shuffle_and_split, print_classification_report

default_dataset = csv_importer_full("dataset/sources/user_fake_authentic_2class.csv")

idx = find_demarcator(default_dataset)
fake = default_dataset[:idx]
correct = default_dataset[idx:]

train_df, validation_df = shuffle_and_split(fake, correct)

custom_train_df = train_df.drop(["pic", "cl", "cz", "ni", "lt", "ahc", "pr", "fo", "cs"], axis=1)
custom_validation_df = validation_df.drop(["pic", "cl", "cz", "ni", "lt", "ahc", "pr", "fo", "cs"], axis=1)
print(custom_train_df)
print(custom_validation_df)
print(id(custom_train_df))
print(id(train_df))

# Default tree
print_classification_report(train_df.iloc[:, :-2], train_df.iloc[:, -1], validation_df)
# Custom tree
print_classification_report(custom_train_df.iloc[:, :-2], custom_train_df.iloc[:, -1], custom_validation_df)
