import json

new_source = '''import lightgbm as lgb
import xgboost as xgb

to_predict = test_feat[test_feat["ID"].isin(sub_raw["ID"])].copy().reset_index(drop=True)
X_test = to_predict[feature_cols].copy()

cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

# 1. LightGBM Multiclasse (3 graines : 42, 1337, 2026 - 220 arbres optimaux)
p_test_mc = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
for s in SEEDS:
    print(f"LGB Multiclasse - Seed {s}...")
    params = get_regularized_lgb_params(seed=s)
    params["learning_rate"] = 0.03
    dtr = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
    clf = lgb.train(params, dtr, num_boost_round=220)
    p_test_mc += clf.predict(X_test) / len(SEEDS)

# 2. LightGBM Ordinal (3 seuils cumulatifs : >=1mm, >=10mm, >=30mm - 180 arbres)
p_test_ord = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
ord_seeds = [42, 1337]
for s in ord_seeds:
    print(f"LGB Ordinal - Seed {s}...")
    p_cum = np.zeros((len(to_predict), 3), dtype=np.float64)
    for t_idx, threshold in enumerate([1, 2, 3]):
        y_bin = (y >= threshold).astype(int)
        dtr = lgb.Dataset(X, label=y_bin, categorical_feature=cat_cols)
        params_bin = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.03,
            "num_leaves": 35,
            "max_depth": 6,
            "min_child_samples": 50,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
            "verbose": -1,
            "random_state": s + threshold,
        }
        clf = lgb.train(params_bin, dtr, num_boost_round=180)
        p_cum[:, t_idx] = clf.predict(X_test)
    p1 = p_cum[:, 0]
    p2 = np.minimum(p1, p_cum[:, 1])
    p3 = np.minimum(p2, p_cum[:, 2])
    p_4c = np.column_stack([
        np.clip(1.0 - p1, 0.0, 1.0),
        np.clip(p1 - p2, 0.0, 1.0),
        np.clip(p2 - p3, 0.0, 1.0),
        np.clip(p3, 0.0, 1.0),
    ])
    p_test_ord += safe_clip_and_renormalize(p_4c) / len(ord_seeds)

# 3. XGBoost Multiclasse (2 graines : 42, 1337 - 150 arbres)
p_test_xgb = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
X_xgb = X.copy()
X_test_xgb = X_test.copy()
for c in cat_cols:
    X_xgb[c] = X_xgb[c].cat.codes.astype("category")
    X_test_xgb[c] = X_test_xgb[c].cat.codes.astype("category")

for s in [42, 1337]:
    print(f"XGBoost - Seed {s}...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=150, learning_rate=0.04, max_depth=5, subsample=0.80,
        colsample_bytree=0.75, reg_alpha=0.3, reg_lambda=2.0,
        objective="multi:softprob", num_class=4, tree_method="hist",
        enable_categorical=True, random_state=s, n_jobs=-1
    )
    xgb_clf.fit(X_xgb, y, verbose=False)
    p_test_xgb += xgb_clf.predict_proba(X_test_xgb) / 2.0

# 4. Blend Triple : 55% LGB MC + 30% LGB Ordinal + 15% XGBoost
p_blend = 0.55 * p_test_mc + 0.30 * p_test_ord + 0.15 * p_test_xgb

# 5. Calibration de température T=1.02 et clipping sécurisé [1e-4, 1-1e-4]
calibrated_preds = apply_temperature_scaling(p_blend, temperature=1.02)

# 6. Construction et vérification de la soumission
submission = pd.DataFrame(calibrated_preds, columns=CLASSES)
submission.insert(0, "ID", to_predict["ID"].values)
final_submission = sub_raw[["ID"]].merge(submission, on="ID", how="left")

assert len(final_submission) == 1820
assert list(final_submission.columns) == ["ID"] + CLASSES
assert (final_submission["ID"].values == sub_raw["ID"].values).all()
assert not final_submission[CLASSES].isna().any().any()
assert (np.abs(final_submission[CLASSES].sum(axis=1) - 1.0).max()) < 1e-4

os.makedirs("submissions", exist_ok=True)
final_submission.to_csv("submissions/submission_notebook_reproductible.csv", index=False)
print("\\nSoumission générée avec succès : submissions/submission_notebook_reproductible.csv")
display(final_submission.head(10))'''

with open('notebook_reproductible.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][13]['source'] = [line + '\n' for line in new_source.strip().split('\n')]

with open('notebook_reproductible.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Updated cell 13 in notebook_reproductible.ipynb successfully.')
