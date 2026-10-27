"""ПР-5 (C): сильный дисбаланс 1:20. База vs class_weight vs undersampling vs oversampling; одинаковая CV."""
import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

X, y = make_classification(n_samples=6000, n_features=15, n_informative=6, weights=[.95,.05], flip_y=.01, class_sep=.8, random_state=0)
print("Баланс классов:", np.bincount(y), f"(1:{np.bincount(y)[0]/np.bincount(y)[1]:.0f})")

def fit_predict(method, Xa, ya, Xb):
    cw = "balanced" if method == "class_weight" else None
    if method == "undersample":
        i0, i1 = np.where(ya==0)[0], np.where(ya==1)[0]; i0 = resample(i0, n_samples=len(i1), replace=False, random_state=0)
        idx = np.r_[i0, i1]; Xa, ya = Xa[idx], ya[idx]
    if method == "oversample":
        i0, i1 = np.where(ya==0)[0], np.where(ya==1)[0]; i1 = resample(i1, n_samples=len(i0), replace=True, random_state=0)
        idx = np.r_[i0, i1]; Xa, ya = Xa[idx], ya[idx]
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight=cw, max_iter=1000)).fit(Xa, ya)
    return m.predict(Xb), m.predict_proba(Xb)[:,1]

skf = StratifiedKFold(5, shuffle=True, random_state=42)       # ОДНА схема для всех методов
rows = []
for meth in ["baseline","class_weight","undersample","oversample"]:
    s = {"P":[], "R":[], "F1":[], "PR-AUC":[]}
    for tr, te in skf.split(X, y):                              # ресэмплинг только внутри train-фолда
        pred, proba = fit_predict(meth, X[tr], y[tr], X[te])
        s["P"].append(precision_score(y[te], pred, zero_division=0)); s["R"].append(recall_score(y[te], pred))
        s["F1"].append(f1_score(y[te], pred)); s["PR-AUC"].append(average_precision_score(y[te], proba))
    rows.append({"method": meth, **{k: f"{np.mean(v):.3f}±{np.std(v):.3f}" for k, v in s.items()}})
print(pd.DataFrame(rows).to_string(index=False))

Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=.2, random_state=1)
pred, _ = fit_predict("class_weight", Xtr, ytr, Xte)
ConfusionMatrixDisplay.from_predictions(yte, pred); plt.title("Матрица ошибок (class_weight)"); plt.savefig("figures/pr5_cm.png", dpi=120)
print("""Вывод (заполните по своей таблице): сравните recall/precision/F1/PR-AUC методов с baseline (см. ± по фолдам).
Если разница меньше ±std - улучшение может быть случайным. Выбор метода зависит от цены ошибки: если пропуск миноритарного класса дороже
ложной тревоги - берём метод с высоким recall (class_weight/oversample), иначе остаёмся на baseline с подбором порога.
class_weight проще: не теряет данные (в отличие от undersample) и не дублирует объекты (в отличие от oversample).""")
