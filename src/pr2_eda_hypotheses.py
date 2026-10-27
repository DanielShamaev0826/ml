"""ПР-2 (C): EDA на «грязном» датасете, 3 гипотезы + статистические тесты, утечки, важность признаков."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_breast_cancer

rng = np.random.default_rng(42)
d = load_breast_cancer(as_frame=True); df = d.frame[["mean radius","mean texture","mean area","mean smoothness","target"]].copy()
df["target"] = df["target"].map({0: "malignant", 1: "benign"})
df["clinic"] = rng.choice(["A","B","C"], len(df))                       # категориальный признак
for c in ["mean texture","mean smoothness"]: df.loc[rng.choice(len(df), 40, replace=False), c] = np.nan   # пропуски
df.loc[rng.choice(len(df), 5, replace=False), "mean area"] *= 5          # выбросы
df.to_csv("data/dirty.csv", index=False)

print(df.isna().sum()); print(df.describe().T)
q1, q3 = df["mean area"].quantile([.25,.75]); iqr = q3-q1
out = df[(df["mean area"] > q3+1.5*iqr)|(df["mean area"] < q1-1.5*iqr)]
print(f"Выбросы mean area по правилу 1.5*IQR: {len(out)}")

fig, ax = plt.subplots(1, 3, figsize=(15,4))
df["mean radius"].hist(bins=30, ax=ax[0]); ax[0].set(title="Гистограмма mean radius", xlabel="mean radius", ylabel="Частота")
df.boxplot("mean area", by="target", ax=ax[1]); ax[1].set(title="Boxplot mean area по классам", xlabel="target", ylabel="mean area")
ax[2].scatter(df["mean radius"], df["mean area"], s=8); ax[2].set(title="radius vs area", xlabel="mean radius", ylabel="mean area")
plt.tight_layout(); plt.savefig("figures/pr2_eda.png", dpi=120)

alpha = 0.05
# H1: средний радиус опухоли различается между классами -> t-тест Уэлча (числовой vs бинарный)
a = df[df.target=="malignant"]["mean radius"]; b = df[df.target=="benign"]["mean radius"]
t, p = stats.ttest_ind(a, b, equal_var=False); print(f"H1 t-тест: t={t:.2f}, p={p:.2e} -> {'отвергаем H0' if p<alpha else 'нет оснований отвергнуть H0'}")
# H2: класс не зависит от клиники -> хи-квадрат (категориальный vs категориальный)
chi, p, dof, _ = stats.chi2_contingency(pd.crosstab(df.clinic, df.target)); print(f"H2 хи-квадрат: chi2={chi:.2f}, p={p:.3f}")
# H3: radius и area коррелируют -> Спирмен (монотонная связь, устойчив к выбросам)
r, p = stats.spearmanr(df["mean radius"], df["mean area"]); print(f"H3 Спирмен: rho={r:.2f}, p={p:.2e}")
print("""
Утечки (специфика): clinic - случайный шум, при реальной связи с классом это сигнал смещения выборки;
признаки area/radius производны друг от друга (мультиколлинеарность, не утечка, но дублирование);
любые статистики (медиана для пропусков) считать только по train.
Предположение: mean radius и mean area будут наиболее значимы (H1, H3: большая разница между классами); clinic - незначим (H2).
Оговорка: корреляция не означает причинность.""")
