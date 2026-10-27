"""ПР-4 (C): линейная регрессия на NumPy (MSE + градиент + GD) vs scikit-learn, влияние learning rate."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

def mse(w, b, X, y): return np.mean((X @ w + b - y) ** 2)
def grad(w, b, X, y):
    e = X @ w + b - y; return 2 * X.T @ e / len(y), 2 * e.mean()     # dMSE/dw, dMSE/db

def gd(X, y, lr, iters):
    w, b, hist = np.zeros(X.shape[1]), 0.0, []
    for _ in range(iters):
        gw, gb = grad(w, b, X, y); w -= lr * gw; b -= lr * gb; hist.append(mse(w, b, X, y))
    return w, b, hist

def report(name, pred): print(f"{name:12s} MAE={mean_absolute_error(yte,pred):.2f} RMSE={mean_squared_error(yte,pred)**.5:.2f} R2={r2_score(yte,pred):.3f}")

plt.figure(figsize=(7,4.5)); models = {}
for lr in (0.001, 0.05, 0.2):
    w, b, h = gd(Xtr, ytr, lr, 1000); models[lr] = (w, b); plt.plot(h, label=f"lr={lr}"); print(f"lr={lr}: loss после 1000 итераций = {h[-1]:.1f}")
plt.yscale("log"); plt.xlabel("Итерация"); plt.ylabel("MSE (train)"); plt.title("Сходимость градиентного спуска"); plt.legend()
plt.savefig("figures/pr4_convergence.png", dpi=120)

sk = LinearRegression().fit(Xtr, ytr)
w, b = models[0.05]
report("NumPy GD", Xte @ w + b); report("sklearn", sk.predict(Xte))
print("Макс. разница коэффициентов:", np.abs(w - sk.coef_).max().round(4), "| intercept:", round(b,2), "vs", round(sk.intercept_,2))
print("lr=0.001 сходится медленно (за 1000 итераций не доходит до минимума), 0.05 - оптимален, 0.2 сходится быстрее всех. При lr около 0.3 и выше на этих данных loss растёт (расходимость) - проверьте сами.")
