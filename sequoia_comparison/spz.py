from dataset.normalizer import json_importer, json_importer_full
import random
import pandas as pd
import os

from sklearn import tree, metrics

PERCENT_TRAIN = 70

print(f"Now splitting dataset with ratio {PERCENT_TRAIN}:{100 - PERCENT_TRAIN}")

fake = json_importer_full("../dataset/sources/automatedAccountData.json", True)
correct = json_importer_full("../dataset/sources/nonautomatedAccountData.json", False)

random.shuffle(fake)
random.shuffle(correct)

train = fake[:int(len(fake) * (PERCENT_TRAIN / 100))]
train += correct[:int(len(correct) * (PERCENT_TRAIN / 100))]
validation = fake[int(len(fake) * (PERCENT_TRAIN / 100)):]
validation += correct[int(len(correct) * (PERCENT_TRAIN / 100)):]

random.shuffle(train)
random.shuffle(validation)

print("Loading complete.")

train_df = pd.DataFrame.from_dict(train)
validation_df = pd.DataFrame.from_dict(validation)

custom_train_df = train_df.drop(["mediaLikeNumbers", "mediaCommentNumbers",
                                 "mediaCommentsAreDisabled", "mediaHashtagNumbers", "mediaHasLocationInfo",
                                 "userHasHighlighReels", "usernameLength", "usernameDigitCount"], axis=1)
custom_validation_df = validation_df.drop(["mediaLikeNumbers", "mediaCommentNumbers",
                                           "mediaCommentsAreDisabled", "mediaHashtagNumbers", "mediaHasLocationInfo",
                                           "userHasHighlighReels", "usernameLength", "usernameDigitCount"], axis=1)

print(train_df)
print(validation_df)

print(custom_train_df)
print(custom_validation_df)

# Default tree
X, y = train_df.iloc[:, :-2], train_df.iloc[:, -1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
print("Fitting complete.")

X_val, y_val = validation_df.iloc[:, :-2], validation_df.iloc[:, -1]
y_pred = clf.predict(X_val)
print(metrics.classification_report(y_val, y_pred))

# Custom tree
cX, cy = custom_train_df.iloc[:, :-2], custom_train_df.iloc[:, -1]
cclf = tree.DecisionTreeClassifier()
cclf = cclf.fit(cX, cy)
print("Fitting complete.")

cX_val, cy_val = custom_validation_df.iloc[:, :-2], custom_validation_df.iloc[:, -1]
cy_pred = cclf.predict(cX_val)

print(metrics.classification_report(cy_val, cy_pred))
