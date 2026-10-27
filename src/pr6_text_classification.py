"""ПР-6 (C): классификация текстов (русский/кыргызский). Ожидает data/corpus.csv (text,label), >=500 документов.
Если файла нет - строится маленький ДЕМО-корпус только для проверки работоспособности кода."""
import os, re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

STOP_RU = {"и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты","к","у","же","вы","за","бы","по","только","ее","мне","было","вот","от","меня","еще","о","из","ему","теперь","когда","даже","ну","вдруг","ли","если","уже","или","ни","быть","был","него","до","вас","нибудь","опять","уж","вам","ведь","там","потом","себя","ничего","ей","может","они","тут","где","есть","надо","ней","для","мы","тебя","их","чем","была","сам","чтоб","без","будто","чего","раз","тоже","себе","под","будет","ж","тогда","кто","этот","того","потому","этого","какой","совсем","ним","здесь","этом","один","почти","мой","тем","чтобы","нее","сейчас","были","куда","зачем","всех","никогда","можно","при","наконец","два","об","другой","хоть","после","над","больше","тот","через","эти","нас","про","всего","них","какая","много","разве","три","эту","моя","впрочем","хорошо","свою","этой","перед","иногда","лучше","чуть","том","нельзя","такой","им","более","всегда","конечно","всю","между"} - {"не","ни","нет"}  # отрицания СОХРАНЕНЫ
# Кыргызский: список составлен вручную - дополните по своему корпусу (частотный анализ)
STOP_KY = {"жана","да","де","та","те","бул","ал","анын","мен","сен","биз","силер","алар","үчүн","менен","эле","болуп","болгон","дагы","ушул","ошол","бир","деп","же","бирок","ага","аны","ан","мына"} - {"эмес","жок"}

def clean(t, stop):
    t = re.sub(r"[^\w\s]|\d|_", " ", str(t).lower())
    return " ".join(w for w in t.split() if w not in stop)
def light_stem(w):   # упрощённый стеммер: отсекает частые окончания (кыргызские + русские)
    for suf in ("лардын","лерди","дарды","тердин","лар","лер","дар","дер","тар","тер","ого","его","ами","ыми","ов","ев","ах","ях","ой","ей","ом","ем","ам","ям","а","я","ы","и","у","ю"):
        if w.endswith(suf) and len(w)-len(suf) >= 3: return w[:-len(suf)]
    return w
def prep(t, stop=STOP_RU|STOP_KY): return " ".join(light_stem(w) for w in clean(t, stop).split())

if os.path.exists("data/corpus.csv"): df = pd.read_csv("data/corpus.csv")
else:
    print("!!! data/corpus.csv не найден - ДЕМО-корпус. Соберите >=500 текстов (новости/отзывы) и разметьте.")
    pos = ["Отличный сервис, очень доволен!","Супер, быстро и недорого","Очень хороший товар, рекомендую","Абдан жакшы кызмат, ыраазымын","Сапаты жогору, сунуштайм"]
    neg = ["Ужасный сервис, не рекомендую","Очень плохо, не доволен","Товар не работает, деньги потрачены зря","Начар кызмат, ыраазы эмесмин","Сапаты өтө начар, сунуштабайм"]
    r = np.random.default_rng(0); df = pd.DataFrame({"text": [r.choice(pos if i%2 else neg)+" "+str(i) for i in range(600)], "label": [i%2 for i in range(600)]})

print("Документов:", len(df), "| баланс:", df.label.value_counts().to_dict())
print("Пример предобработки:", df.text[0], "->", prep(df.text[0]))
df["clean"] = df.text.map(prep)
Xtr, Xte, ytr, yte = train_test_split(df.clean, df.label, test_size=.2, stratify=df.label, random_state=42)

pipes = {"MultinomialNB": MultinomialNB(), "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced")}
for name, clf in pipes.items():
    p = Pipeline([("tfidf", TfidfVectorizer(min_df=2, max_df=.9, ngram_range=(1,2))), ("clf", clf)]).fit(Xtr, ytr)   # векторизатор - только на train
    pr = p.predict(Xte); print(f"\n{name}: размерность признаков={len(p['tfidf'].vocabulary_)}, acc={accuracy_score(yte,pr):.3f}, macro-F1={f1_score(yte,pr,average='macro'):.3f}")
    print(classification_report(yte, pr, zero_division=0))
    err = Xte[pr != yte].head(5)
    for t in err.index: print("ОШИБКА:", df.text[t], "| истина:", yte[t], "| предсказано:", pr[list(Xte.index).index(t)])
print("Для отчёта: опишите процедуру разметки, источник, лицензию; сравните NB и LogReg; прокомментируйте ошибки (сарказм, редкие слова, отрицания).")
