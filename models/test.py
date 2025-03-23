import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold

data = pd.DataFrame([['red', 'strawberry'],  # color, fruit
                     ['red', 'strawberry'],
                     ['red', 'strawberry'],
                     ['red', 'strawberry'],
                     ['red', 'strawberry'],
                     ['yellow', 'banana'],
                     ['yellow', 'banana'],
                     ['yellow', 'banana'],
                     ['yellow', 'banana'],
                     ['yellow', 'banana']])

X = data[0]

# KFold
for train_index, test_index in KFold(n_splits=2, shuffle=True, random_state=1).split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

print()

# RepeatedKFold
for train_index, test_index in RepeatedKFold(n_splits=2, n_repeats=3, random_state=1).split(X):
    print("TRAIN:", train_index, "TEST:", test_index)