import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / "test.csv")


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def predict_file(model, uploaded, threshold):
    try:
        batch = pd.read_csv(uploaded)
    except Exception:
        return None, "Не удалось прочитать файл: нужен CSV с заголовком в первой строке."

    features = list(model.feature_names_in_)
    missing = [column for column in features if column not in batch.columns]
    if missing:
        return None, f"В файле не хватает столбцов: {', '.join(missing)}"

    X = batch[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if X.empty:
        return None, "В файле нет ни одной строки с вином."
    if X.isna().any().any():
        return None, "В признаках есть пустые или нечисловые значения."

    result = batch.copy()
    result["proba"] = model.predict_proba(X)[:, 1]
    result["good"] = result["proba"] >= threshold
    return result, None


st.set_page_config(page_title="Хорошее ли вино", layout="wide")

for name in ["model.joblib", "config.json", "test.csv"]:
    if not (APP_DIR / name).exists():
        st.error(f"Нет файла {name}. Сначала выполните ячейку сохранения модели в ноутбуке.")
        st.stop()

model = load_model()
features = list(model.feature_names_in_)
config = json.loads((APP_DIR / "config.json").read_text())

test = load_test()
X_test = test[features]
y_test = test["target"]
proba = test_proba(model, X_test)

st.title("Хорошее ли вино")
st.caption("Логистическая регрессия: вероятность того, что вино получит оценку 7 и выше.")

threshold = st.sidebar.slider(
    "Порог «хорошее вино»", 0.0, 1.0, float(config["threshold"]), step=0.001, format="%.3f"
)
st.sidebar.write(f"Порог из config.json: {config['threshold']:.3f}")

tab_wine, tab_metrics, tab_batch, tab_coef = st.tabs(
    ["Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)

with tab_wine:
    values = {}
    columns = st.columns(3)
    numeric = [feature for feature in features if feature != "is_red"]
    for i, feature in enumerate(numeric):
        low = float(test[feature].min())
        high = float(test[feature].max())
        values[feature] = columns[i % 3].number_input(
            feature,
            min_value=low,
            max_value=high,
            value=float(test[feature].median()),
            step=(high - low) / 100,
            format="%.4f",
        )
    color = st.radio("Цвет", ["белое", "красное"], horizontal=True)
    values["is_red"] = 1 if color == "красное" else 0

    wine = pd.DataFrame([values])[features]
    wine_proba = model.predict_proba(wine)[0, 1]

    st.metric("Вероятность, что вино хорошее", f"{wine_proba:.1%}")
    st.progress(float(wine_proba))
    if wine_proba >= threshold:
        st.success(f"Хорошее вино: вероятность не ниже порога {threshold:.3f}.")
    else:
        st.warning(f"Не берём в подборку: вероятность ниже порога {threshold:.3f}.")

with tab_metrics:
    y_pred = proba >= threshold
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)

    left, middle, right = st.columns(3)
    left.metric("Precision", f"{precision:.3f}")
    middle.metric("Recall", f"{recall:.3f}")
    right.metric("F1", f"{f1:.3f}")

    st.subheader("Матрица ошибок")
    st.dataframe(
        pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["на деле обычное", "на деле хорошее"],
            columns=["модель: обычное", "модель: хорошее"],
        )
    )

    fpr, tpr, _ = metrics.roc_curve(y_test, proba)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, proba)
    point_fpr = fp / (fp + tn)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC-AUC = {metrics.roc_auc_score(y_test, proba):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.scatter(point_fpr, recall, color="red", zorder=3, label=f"порог {threshold:.3f}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR (recall)")
        ax.set_title("ROC-кривая")
        ax.legend()
        show(fig)
    with right:
        fig, ax = plt.subplots()
        ax.plot(pr_recall, pr_precision, label=f"AP = {metrics.average_precision_score(y_test, proba):.3f}")
        ax.scatter(recall, precision, color="red", zorder=3, label=f"порог {threshold:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR-кривая")
        ax.legend()
        show(fig)

    st.subheader("Для сомелье")
    st.write(
        f"Сейчас в подборку попадает {tp + fp} из {len(y_test)} вин тестовой выборки: "
        f"{tp} из них действительно хорошие, ещё {fn} хороших вин модель пропустила."
    )
    st.write(
        "Если порог **поднять**, подборка станет короче и, как правило, точнее: "
        "случайных вин меньше, но часть хороших вин в неё не попадёт. "
        "Если порог **опустить**, подборка станет шире: "
        "хороших вин теряется меньше, зато больше обычных."
    )

with tab_batch:
    st.write(f"Решение принимается при пороге из боковой панели: {threshold:.3f}.")
    uploaded = st.file_uploader("CSV с признаками вин", type="csv")
    if uploaded is None:
        st.info(f"Нужные столбцы: {', '.join(features)}")
    else:
        result, error = predict_file(model, uploaded, threshold)
        if error:
            st.error(error)
        else:
            st.metric("Хороших вин по прогнозу", int(result["good"].sum()))
            st.dataframe(result, hide_index=True)
            st.download_button(
                "Скачать предсказания",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                on_click="ignore",
            )

with tab_coef:
    coef = pd.Series(model.named_steps["logisticregression"].coef_[0], index=features).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coef.index, coef.values, color=["tab:green" if value > 0 else "tab:red" for value in coef])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Коэффициент после StandardScaler")
    ax.set_title("Зелёные — за хорошее вино, красные — против")
    show(fig)

    st.write("Сильнее всего **за** хорошее вино:", ", ".join(coef[coef > 0].index[::-1][:3]))
    st.write("Сильнее всего **против**:", ", ".join(coef[coef < 0].index[:3]))
    st.caption(
        "Признаки отмасштабированы, поэтому коэффициенты можно сравнивать: это изменение логита "
        "при росте признака на одно стандартное отклонение."
    )
