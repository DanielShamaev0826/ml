"""ПР-7 (C): impurity-based vs permutation importance vs SHAP (если установлен)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

d = load_breast_cancer(as_frame=True); X, y = d.data.copy(), d.target
rng = np.random.default_rng(0)
X["noise_unique"] = rng.normal(size=len(X))        # шум с множеством уникальных значений: проверка смещения impurity-важности
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=300, random_state=42, oob_score=True, n_jobs=-1).fit(Xtr, ytr)
print(f"Деревьев: 300 | OOB={rf.oob_score_:.3f} | test acc={rf.score(Xte,yte):.3f}")

imp = pd.Series(rf.feature_importances_, X.columns)
perm = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=0, n_jobs=-1)
perm = pd.Series(perm.importances_mean, X.columns)
cmp = pd.DataFrame({"impurity": imp, "permutation": perm}); cmp["rank_imp"] = cmp.impurity.rank(ascending=False); cmp["rank_perm"] = cmp.permutation.rank(ascending=False)
print(cmp.sort_values("impurity", ascending=False).head(10).round(4))
print("Ранг шумового признака: impurity =", int(cmp.loc["noise_unique","rank_imp"]), "| permutation =", int(cmp.loc["noise_unique","rank_perm"]))

fig, ax = plt.subplots(1, 2, figsize=(14,6))
imp.nlargest(10).sort_values().plot.barh(ax=ax[0], title="Impurity-based (top-10)"); ax[0].set_xlabel("Важность")
perm.nlargest(10).sort_values().plot.barh(ax=ax[1], title="Permutation (test, top-10)"); ax[1].set_xlabel("Падение accuracy")
plt.tight_layout(); plt.savefig("figures/pr7_importance.png", dpi=120)

try:
    import shap
    sv = shap.TreeExplainer(rf).shap_values(Xte.iloc[:3])
    print("SHAP для 3 объектов вычислен; shap.force_plot / shap.summary_plot для визуализации.")
except ImportError:
    print("shap не установлен: pip install shap, затем shap.TreeExplainer(rf).shap_values(Xte.iloc[:3]).")
print("""Вывод: impurity считается по train и завышает признаки с большим числом уникальных значений;
permutation считается на test, но при коррелирующих признаках делит/занижает важность (перестановка одного компенсируется вторым);
SHAP даёт вклад признака в КОНКРЕТНОЕ предсказание со знаком и направлением (локальное объяснение) - чего нет у первых двух.
Важность не равна причинности.""")
