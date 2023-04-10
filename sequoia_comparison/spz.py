from dataset.normalizer import json_importer_full
from sequoia_comparison.utils import shuffle_and_split, print_classification_report

PERCENT_TRAIN = 70

print(f"Now splitting dataset with ratio {PERCENT_TRAIN}:{100 - PERCENT_TRAIN}")

fake = json_importer_full("dataset/sources/automatedAccountData.json", True)
correct = json_importer_full("dataset/sources/nonautomatedAccountData.json", False)

train_df, validation_df = shuffle_and_split(fake, correct)

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
print_classification_report(train_df.iloc[:, :-2], train_df.iloc[:, -1], validation_df)
# Custom tree
print_classification_report(custom_train_df.iloc[:, :-2], custom_train_df.iloc[:, -1], custom_validation_df)
