import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.text import Annotation
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# just clean up real quick
df = pd.read_csv("Life_Expectancy_Data.csv")
df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
df = df.dropna(subset=["Life expectancy"])
df["Status"] = (df["Status"] == "Developed").astype(int)
df[["Income composition of resources", "Schooling"]] = (
    df[["Income composition of resources", "Schooling"]].replace(0, np.nan)
)


groups = df["Country"]
y = df["Life expectancy"]


# drop the outcome-like mortality columns and the near-duplicate thinness column it is literally telling us the death rate which is what we want the why for
outcome_like = ["Adult Mortality", "infant deaths", "under-five deaths"]
X = df.drop(columns=["Life expectancy", "Country", "Year", "thinness 1-19 years"] + outcome_like)


# ---------- split by country ----------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(X, y, groups))
X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
g_tr = groups.iloc[train_idx]

# ---------- tune ----------
param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 8, None],
    "min_samples_leaf": [5, 10, 20, 40, 80],
    "ccp_alpha": [0, 0.01, 0.05, 0.1],
}
search = GridSearchCV(
    DecisionTreeRegressor(random_state=0),
    param_grid,
    cv=GroupKFold(n_splits=5),
    scoring="neg_root_mean_squared_error",
)
search.fit(X_tr, y_tr, groups=g_tr)
print("best params:", search.best_params_)

best = search.best_estimator_
pred = best.predict(X_te)


# NEW: numbers to back up your claims
print(export_text(best, feature_names=list(X.columns), max_depth=3))
imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(10).round(3))



fig, ax = plt.subplots(figsize=(30, 12))
plot_tree(best, feature_names=X.columns, filled=True, rounded=True,
          max_depth=3, fontsize=11, precision=1, impurity=False, ax=ax)

vals = best.tree_.value.flatten()
norm = Normalize(vals.min(), vals.max())
for a in ax.findobj(Annotation):
    hit = re.search(r"value = ([\d.]+)", a.get_text())
    if hit and a.get_bbox_patch() is not None:
        a.get_bbox_patch().set_facecolor(cm.Greens(0.15 + 0.7 * norm(float(hit.group(1)))))


plt.tight_layout()
fig.savefig("life_expectancy_treeRUN2.png", dpi=300, bbox_inches="tight")


# ======================================================
# NO-HIV RUN: same pipeline, HIV/AIDS removed
# ======================================================
cols_noHIV = [c for c in X.columns if c != "HIV/AIDS"]

search2 = GridSearchCV(
    DecisionTreeRegressor(random_state=0),
    param_grid,                                   # same grid as the main run
    cv=GroupKFold(n_splits=5),
    scoring="neg_root_mean_squared_error",
)
search2.fit(X_tr[cols_noHIV], y_tr, groups=g_tr)
best2 = search2.best_estimator_
pred2 = best2.predict(X_te[cols_noHIV])

print("\n===== NO HIV =====")
print("best params:", search2.best_params_)
print("test RMSE:", np.sqrt(mean_squared_error(y_te, pred2)))
print("test R^2:", r2_score(y_te, pred2))
print(export_text(best2, feature_names=cols_noHIV, max_depth=3))
imp2 = pd.Series(best2.feature_importances_, index=cols_noHIV).sort_values(ascending=False)
print(imp2.head(10).round(3))

# plot
fig2, ax2 = plt.subplots(figsize=(30, 12))
plot_tree(best2, feature_names=cols_noHIV, filled=True, rounded=True,
          max_depth=3, fontsize=11, precision=1, impurity=False, ax=ax2)

vals2 = best2.tree_.value.flatten()
norm2 = Normalize(vals2.min(), vals2.max())
for a in ax2.findobj(Annotation):
    hit = re.search(r"value = ([\d.]+)", a.get_text())
    if hit and a.get_bbox_patch() is not None:
        a.get_bbox_patch().set_facecolor(cm.Greens(0.15 + 0.7 * norm2(float(hit.group(1)))))

plt.tight_layout()
fig2.savefig("life_expectancy_tree_noHIV.png", dpi=300, bbox_inches="tight")