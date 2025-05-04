import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data():
    return pd.read_csv("water_potability.csv")


def model_training_page():
    st.title("SVM: Определение пригодности воды для питья")

    df = load_data()

    # Заполнение пропусков
    imputer = SimpleImputer(strategy="median")
    df["ph"] = imputer.fit_transform(df[["ph"]])
    df["Sulfate"] = imputer.fit_transform(df[["Sulfate"]])
    df["Trihalomethanes"] = imputer.fit_transform(df[["Trihalomethanes"]])

    # Разделение выборки
    X = df.drop(columns=["Potability"])
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # Масштабирование данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.subheader("Настройка гиперпараметров")

    kernel = st.selectbox("Ядро", options=["linear", "poly", "rbf", "sigmoid"])
    C = st.slider("C", min_value=0.01, max_value=10.0, value=1.0, step=0.01)

    model = SVC(kernel=kernel, C=C, probability=True, random_state=1)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    metrics_data = {
        "Метрика": ["Accuracy", "Precision", "Recall", "F1-мера"],
        "Значение": [
            f"{accuracy * 100:.2f}%",
            f"{precision * 100:.2f}%",
            f"{recall * 100:.2f}%",
            f"{f1 * 100:.2f}%",
        ],
    }
    metrics_df = pd.DataFrame(metrics_data)

    st.subheader("Основные метрики")
    st.table(metrics_df.set_index("Метрика"))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Матрица ошибок")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Реальное значение")
    st.pyplot(fig)

    # ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    st.subheader("ROC-кривая")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Receiver Operating Characteristic")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)


def main():
    model_training_page()


if __name__ == "__main__":
    main()
