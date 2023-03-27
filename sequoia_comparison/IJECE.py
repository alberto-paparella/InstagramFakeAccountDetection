# Caricare il dataset nel formato .csv
# Caricare il dataset nel formato parziale
from dataset.normalizer import csv_importer, csv_importer_full
import csv
import random
import pandas as pd

from sklearn import tree, metrics


def find_demarcator(dataset):
    """
    Restituisce l'indice del primo elemento non fake
    :param dataset: il dataset
    :return: l'indice
    """
    idx = 0
    for elem in dataset:
        if elem['fake'] == 1:
            idx += 1
        else:
            break
    return idx


PERCENT_TRAIN = 70

default_dataset = csv_importer_full("../dataset/sources/user_fake_authentic_2class.csv")
custom_dataset = csv_importer("../dataset/sources/user_fake_authentic_2class.csv")

print(f"Now splitting dataset with ratio {PERCENT_TRAIN}:{100 - PERCENT_TRAIN}")

idx = find_demarcator(default_dataset)

fake = default_dataset[:idx]
correct = default_dataset[idx:]
custom_fake = custom_dataset[:idx]
custom_correct = custom_dataset[idx:]

random.shuffle(fake)
random.shuffle(correct)
random.shuffle(custom_fake)
random.shuffle(custom_correct)

train = fake[:int(len(fake) * (PERCENT_TRAIN / 100))]
train += correct[:int(len(correct) * (PERCENT_TRAIN / 100))]
custom_train = custom_fake[:int(len(custom_fake) * (PERCENT_TRAIN / 100))]
custom_train += custom_correct[:int(len(custom_correct) * (PERCENT_TRAIN / 100))]

validation = fake[int(len(fake) * (PERCENT_TRAIN / 100)):]
validation += correct[int(len(correct) * (PERCENT_TRAIN / 100)):]
custom_validation = custom_fake[int(len(custom_fake) * (PERCENT_TRAIN / 100)):]
custom_validation += custom_correct[int(len(custom_correct) * (PERCENT_TRAIN / 100)):]

random.shuffle(train)
random.shuffle(validation)
random.shuffle(custom_train)
random.shuffle(custom_validation)

print("Loading complete.")

train_df = pd.DataFrame.from_dict(train)
validation_df = pd.DataFrame.from_dict(validation)
print(train_df)
print(validation_df)

custom_train_df = pd.DataFrame.from_dict(custom_train)
custom_validation_df = pd.DataFrame.from_dict(custom_validation)
print(custom_train_df)
print(custom_validation_df)

# Default tree
X, y = train_df.iloc[:, :-2], train_df.iloc[:, -1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
print("Fitting complete.")

X_val, y_val = validation_df.iloc[:, :-2], validation_df.iloc[:, -1]
y_pred = clf.predict(X_val)

#y_compare = y_pred - y_val
#print('accuracy =', 100 - (sum(abs(y_compare)) / len(validation_df.index)) * 100)
print(metrics.classification_report(y_val,y_pred))

# Custom tree
cX, cy = custom_train_df.iloc[:,:-2], custom_train_df.iloc[:,-1]
cclf = tree.DecisionTreeClassifier()
cclf = cclf.fit(cX, cy)
print("Fitting complete.")

cX_val, cy_val = custom_validation_df.iloc[:,:-2], validation_df.iloc[:, -1]
cy_pred = cclf.predict(cX_val)

#cy_compare = cy_pred - cy_val
#print('accuracy =', 100 - (sum(abs(cy_compare)) / len(custom_validation_df.index)) * 100)
print(metrics.classification_report(cy_val,cy_pred))