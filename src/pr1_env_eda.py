"""ПР-1 (C): воспроизводимое окружение + первичный анализ датасета + data dictionary.
Датасет: Diabetes (UCI/sklearn) - прогноз прогрессирования заболевания. Замените на свой (Kaggle/UCI/data.gov.kg)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

d = load_diabetes(as_frame=True, scaled=False)
df = d.frame
df.to_csv("data/diabetes.csv", index=False)       # относительный путь внутри репозитория

print("Размер:", df.shape); print(df.dtypes); print("Пропуски:\n", df.isna().sum())
print(df.describe().T)

dictionary = pd.DataFrame({
    "feature": ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6","target"],
    "description": ["Возраст (лет)","Пол (1/2)","Индекс массы тела","Среднее артериальное давление",
        "Общий холестерин сыворотки","LDL-холестерин","HDL-холестерин","Отношение холестерин/HDL",
        "Лог. уровень триглицеридов","Уровень глюкозы","Прогрессирование болезни через год (целевая)"],
    "type": ["числовой","категориальный","числовой"]+["числовой"]*7+["числовой (target)"]})
dictionary.to_csv("data/data_dictionary.csv", index=False); print(dictionary.to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
df["target"].hist(bins=25, ax=ax[0]); ax[0].set(title="Распределение target", xlabel="target", ylabel="Частота")
df["sex"].value_counts().plot.bar(ax=ax[1], title="Соотношение полов"); ax[1].set(xlabel="sex", ylabel="Количество")
ax[2].scatter(df["bmi"], df["target"], s=10); ax[2].set(title="BMI vs target", xlabel="bmi", ylabel="target")
plt.tight_layout(); plt.savefig("figures/pr1_eda.png", dpi=120)
print("\nНаблюдения: 1) target скошен вправо; 2) bmi положительно связан с target; 3) пропусков нет, пол ~ поровну.")
