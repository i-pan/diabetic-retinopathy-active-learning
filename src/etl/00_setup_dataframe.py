import numpy as np
import pandas as pd


train_df = pd.read_csv("../../data/trainLabels.csv")
test_df = pd.read_csv("../../data/retinopathy_solution.csv")

train_df["filename"] = "train/" + train_df.image + ".jpeg"
test_df["filename"] = "test/" + test_df.image + ".jpeg"
del test_df["Usage"]
eyepacs_df = pd.concat([train_df, test_df])
eyepacs_df["label"] = eyepacs_df.level
eyepacs_df["pid"] = eyepacs_df.image.apply(lambda x: int(x.split("_")[0]))

# Sample 5,000 patients for test
np.random.seed(8)
test_pids = np.random.choice(eyepacs_df.pid.unique(), 5000, replace=False)
# Sample 500 patients for validation
not_test_pids = list(set(eyepacs_df.pid.unique()) - set(test_pids))
assert len(not_test_pids) == len(eyepacs_df.pid.unique()) - 5000
valid_pids = np.random.choice(not_test_pids, 500, replace=False)
eyepacs_df["split"] = "train"
eyepacs_df.loc[eyepacs_df.pid.isin(list(test_pids)), "split"] = "test"
eyepacs_df.loc[eyepacs_df.pid.isin(list(valid_pids)), "split"] = "valid"
print(eyepacs_df.split.value_counts())

# Sample 500 patients for the initial labeled set
initial_train_pids = np.random.choice(eyepacs_df[eyepacs_df.split == "train"].pid.unique(),
                                      500,
                                      replace=False)
eyepacs_df["initial"] = 0
eyepacs_df.loc[eyepacs_df.pid.isin(list(initial_train_pids)), "initial"] = 1

print(eyepacs_df.initial.value_counts())
print(eyepacs_df[eyepacs_df.split == "train"].initial.value_counts())

del eyepacs_df["level"]
del eyepacs_df["image"]

aptos_df = pd.read_csv("../../data/aptos/train.csv")
aptos_df["filename"] = "aptos/train_images/" + aptos_df.id_code + ".png"
aptos_df["pid"] = aptos_df.id_code
aptos_df["label"] = aptos_df["diagnosis"]
aptos_df["split"] = "ext_test"
del aptos_df["diagnosis"]
del aptos_df["id_code"]

df = pd.concat([eyepacs_df, aptos_df])
print(df.split.value_counts())
df.to_csv("../../data/active_learning_dataframe.csv", index=False)


