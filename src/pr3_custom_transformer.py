"""ПР-3 (C): собственный трансформер Winsorizer+log в sklearn Pipeline, тесты, оценка производительности."""
import time, numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class WinsorLog(BaseEstimator, TransformerMixin):
    """Обрезает выбросы по перцентилям (границы запоминаются на train), затем log1p."""
    def __init__(self, lower=1, upper=99, log=True): self.lower, self.upper, self.log = lower, upper, log
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.lower, axis=0); self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]; return self          # fit только запоминает статистики
    def transform(self, X):
        X = np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)
        return np.log1p(X - np.minimum(self.lo_, 0)) if self.log else X
    def get_feature_names_out(self, names=None): return np.asarray(names)

def make_data(n, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"income": r.lognormal(10, 1, n), "age": r.integers(18, 80, n).astype(float),
                       "city": r.choice(["Bishkek","Osh","Karakol", None], n)})
    df.loc[r.choice(n, n//20, replace=False), "income"] = np.nan
    return df

def build():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("wl", WinsorLog()), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="неизвестно")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, ["income","age"]), ("cat", cat, ["city"])])

df = make_data(5000); train, test = train_test_split(df, test_size=0.2, random_state=42)   # split до статистик
pre = build().fit(train); Xtr, Xte = pre.transform(train), pre.transform(test)             # fit только на train
print("train:", Xtr.shape, "test:", Xte.shape)

# ---- тесты ----
def test_no_refit():
    w = WinsorLog().fit(train[["income"]].fillna(0)); lo, hi = w.lo_.copy(), w.hi_.copy()
    w.transform(test[["income"]].fillna(0)); assert (w.lo_ == lo).all() and (w.hi_ == hi).all()
def test_new_data_clipped():
    w = WinsorLog(log=False).fit(np.arange(100).reshape(-1,1).astype(float))
    out = w.transform(np.array([[10_000.0],[-5.0]])); assert out[0,0] == w.hi_[0] and out[1,0] == w.lo_[0]
def test_shapes_match(): assert Xtr.shape[1] == Xte.shape[1]
for t in (test_no_refit, test_new_data_clipped, test_shapes_match): t(); print("OK:", t.__name__)

# ---- производительность ----
for n in (10_000, 100_000, 1_000_000):
    big = make_data(n, 1); p = build()
    t0 = time.perf_counter(); p.fit(big); tf = time.perf_counter()-t0
    t0 = time.perf_counter(); p.transform(big); tt = time.perf_counter()-t0
    print(f"n={n:>9,}: fit {tf:.2f}s, transform {tt:.2f}s")
print("Узкое место: OneHotEncoder/SimpleImputer на object-столбце (строковые операции), число-вая часть векторизована в NumPy.")
