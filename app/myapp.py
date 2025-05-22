import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Загрузка данных
data = pd.read_csv("diabetes_dataset.csv")

# Настройка страницы
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Заголовок приложения
st.title("Diabetes Prediction Models")

# Вкладки
tab1, tab2, tab3 = st.tabs(["Данные", "Обучение модели", "Результаты"])

with tab1:
    st.subheader("Описание датасета")
    st.markdown(
        """
    Источник: [Diabetes Prediction](https://www.kaggle.com/datasets/saurabh00007/diabetescsv)
    
    - **Pregnancies**: число беременностей
    - **Glucose**: концентрация глюкозы в плазме крови
    - **BloodPressure**: дистолическое давление крови
    - **SkinThickness**: толщина кожи трицепса
    - **Insulin**: содержание инсулина в крови
    - **BMI**: индекс массы тела
    - **DiabetesPedigreeFunction**: показатель функции генетического диабета
    - **Age**: возраст
    - **Outcome**: целевая переменная, где 0 — нет диабета, 1 — есть
    """
    )

    st.subheader("Первые строки датасета")
    st.dataframe(data.head())

    st.subheader("Статистика данных")
    st.dataframe(data.describe())

    st.subheader("Корреляционная матрица")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        data.corr(method="pearson"), ax=ax, annot=True, fmt=".2f", cmap="coolwarm"
    )
    st.pyplot(fig)


with tab2:
    st.header("Обучение модели")
    st.subheader("Выбор модели")

    model_name = st.selectbox(
        "Выберите модель:",
        ["Логистическая регрессия", "SVM", "Дерево принятия решений"],
    )

    if st.button("Обучить модель"):
        X = data.drop("Outcome", axis=1)
        y = data["Outcome"]

        # Разделение выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Масштабироание данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Выбор модели
        if model_name == "Логистическая регрессия":
            model = LogisticRegression()
        elif model_name == "SVM":
            model = SVC(probability=True)
        elif model_name == "Дерево принятия решений":
            model = DecisionTreeClassifier()

        # Обучение модели
        model.fit(X_train_scaled, y_train)

        # Сохранение результатов
        st.session_state.model = model
        st.session_state.X_test_scaled = X_test_scaled
        st.session_state.y_test = y_test
        st.session_state.trained = True
        st.success(f"Модель {model_name} успешно обучена")

with tab3:
    st.header("Результаты обучения")
    if "trained" in st.session_state and st.session_state.trained:
        model = st.session_state.model
        X_test_scaled = st.session_state.X_test_scaled
        y_test = st.session_state.y_test

        # Предсказания
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        st.subheader("Метрики")
        
        st.markdown("""
        **Accuracy**  
        Процент (долю в диапазоне от 0 до 1) правильно определенных классов.

        **Precision**  
        $precision = \\frac{TP}{TP+FP}$  
        Доля верно предсказанных классификатором положительных объектов, из всех объектов, которые классификатор верно или неверно определил как положительные.

        **Recall**  
        $recall = \\frac{TP}{TP+FN}$  
        Доля верно предсказанных классификатором положительных объектов, из всех действительно положительных объектов.

        **F1-мера**  
        $F_1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}$
        """)
        
        metrics = {
                    "Метрика": ["Accuracy", "Precision", "Recall", "F1 Score"],
                    "Значение": [
                        round(accuracy_score(y_test, y_pred), 3),
                        round(precision_score(y_test, y_pred), 3),
                        round(recall_score(y_test, y_pred), 3),
                        round(f1_score(y_test, y_pred), 3)
            ]
        }
        metrics_df = pd.DataFrame(metrics)

        # Отображение таблицы с выравниванием
        st.markdown(
            metrics_df.to_html(index=False, 
                               table_id="metrics-table", 
                               justify='center'),
            unsafe_allow_html=True
        )

        # Стилизация таблицы с использованием CSS
        st.markdown("""
            <style>
            #metrics-table {
                width: 50%;
                margin-left: auto; 
                margin-right: auto;
            }
            #metrics-table th {
                text-align: center;
            }
            #metrics-table td {
                text-align: center;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )
                
        # Матрица ошибок
        st.subheader("Матрица ошибок")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истина")
        st.pyplot(fig)

        # ROC-кривая
        st.subheader("ROC-кривая и AUC")

        def draw_roc_curve(
            y_true, y_score, pos_label=1, average="micro", title="ROC-кривая"
        ):
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
            roc_auc_value = roc_auc_score(y_true, y_score, average=average)
            plt.figure()
            lw = 2
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=lw,
                label="ROC кривая (площадь = %0.2f)" % roc_auc_value,
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.legend(loc="lower right")
            st.pyplot(plt.gcf())

        draw_roc_curve(y_test, y_prob)

    else:
        st.warning("Сначала обучите модель на вкладке 'Обучение модели'")
