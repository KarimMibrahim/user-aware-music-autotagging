import pandas as pd
DATASET_PATH = "/src_code/repo/GroundTruth/"

# read data
train = pd.read_csv(DATASET_PATH + "train_set.csv")
test = pd.read_csv(DATASET_PATH + "test_set.csv")
valid = pd.read_csv(DATASET_PATH + "validation_set.csv")

# combine and save
all_labels = pd.concat([train,test,valid], ignore_index=True)
all_labels.to_csv(DATASET_PATH + "single_label_all.csv", index = False)

# make multilabel sets
# for training
train.drop("user_id",axis = 1, inplace=True)
train_multilabel = train.groupby("song_id").sum()
train_multilabel[train_multilabel>1] = 1
train_multilabel = train_multilabel.reset_index()
train_multilabel.to_csv(DATASET_PATH+"train_multilabel.csv", index = False)

# for test
test.drop("user_id",axis = 1, inplace=True)
test_multilabel = test.groupby("song_id").sum()
test_multilabel[test_multilabel>1] = 1
test_multilabel = test_multilabel.reset_index()
test_multilabel.to_csv(DATASET_PATH+"test_multilabel.csv", index = False)

# for validation
valid.drop("user_id",axis = 1, inplace=True)
valid_multilabel = valid.groupby("song_id").sum()
valid_multilabel[valid_multilabel>1] = 1
valid_multilabel = valid_multilabel.reset_index()
valid_multilabel.to_csv(DATASET_PATH+"validation_multilabel.csv", index = False)

# combine and save
all_samples = pd.concat([train_multilabel,test_multilabel,valid_multilabel])
all_samples = all_samples.groupby("song_id").sum()
all_samples[all_samples>1] = 1
all_samples = all_samples.reset_index()
all_samples.to_csv(DATASET_PATH+"multilabel_all.csv", index = False)