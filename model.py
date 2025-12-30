import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split, learning_curve


parser = argparse.ArgumentParser(
    prog='agl_mat_gbt',
)
parser.add_argument('-f', '--features')
args = parser.parse_args()


feat_df = pd.read_csv(args.features)

y = feat_df["Target"]
X = feat_df.drop(["ID", "Target"], axis=1, errors='ignore')

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=10000,
    learning_rate=.01,
    max_depth=7,
    min_child_weight=3,
    colsample_bytree=.7,
    tree_method="hist",
    device = 'cuda',
    n_jobs=1
)

train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X = X,
    y = y,
    cv = 5,
    train_sizes=np.linspace(0.1,1.0,5),
    scoring="neg_root_mean_squared_error",
    n_jobs=1,
    shuffle=True
)

train_mean = -np.mean(train_scores, axis=1)
train_std  =  np.std(train_scores, axis=1)

test_mean  = -np.mean(test_scores, axis=1)
test_std   =  np.std(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label="Training RMSE")
plt.plot(train_sizes, test_mean, label="Validation RMSE")

plt.fill_between(
    train_sizes,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.2
)

plt.fill_between(
    train_sizes,
    test_mean - test_std,
    test_mean + test_std,
    alpha=0.2
)

plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.legend()
plt.tight_layout()

plt.savefig("learning_curve_xgboost.png", dpi=300, bbox_inches="tight")
plt.show()