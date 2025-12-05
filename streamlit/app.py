from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from model_utils import TEST_URL, TRAIN_URL, build_features


@st.cache_resource
def load_model():
    bundle_path = Path(__file__).parent / "artifacts" / "model_bundle.pkl"
    if not bundle_path.exists():
        st.error("Model bundle not found. Run `python streamlit/train_model.py` first.")
        st.stop()
    return joblib.load(bundle_path)


@st.cache_data
def load_data():
    df_train = build_features(pd.read_csv(TRAIN_URL))
    df_test = build_features(pd.read_csv(TEST_URL))
    return df_train, df_test


def eda_section(df: pd.DataFrame, df_test: pd.DataFrame):
    st.header("EDA: основные распределения")
    st.write(f"Размеры train: {df.shape[0]} строк, {df.shape[1]} колонок")
    st.write(f"Размеры test: {df_test.shape[0]} строк, {df_test.shape[1]} колонок")
    st.write(f"Дубликатов в train: {df.duplicated().sum()}")

    missing = df.isna().sum()
    if missing.any():
        st.subheader("Пропуски по колонкам")
        st.bar_chart(missing[missing > 0])

    # Приводим seats к категориальному для EDA
    df_eda = df.copy()
    df_test_eda = df_test.copy()
    for frame in (df_eda, df_test_eda):
        if "seats" in frame.columns:
            frame["seats"] = frame["seats"].astype(str)

    st.subheader("Распределение цены (train vs test)")
    fig, ax = plt.subplots()
    sns.kdeplot(df_eda["selling_price"].dropna(), label="train", ax=ax)
    if "selling_price" in df_test_eda.columns:
        sns.kdeplot(df_test_eda["selling_price"].dropna(), label="test", ax=ax)
    ax.legend()
    st.pyplot(fig)

    # Кастомный выбор признака для распределения
    st.subheader("Распределение выбранного признака")
    num_cols = [c for c in df_eda.select_dtypes(include=[np.number]).columns if c != "seats"]
    default_num = "mileage" if "mileage" in num_cols else (num_cols[0] if num_cols else None)
    choice = st.selectbox("Выберите числовой столбец", options=num_cols, index=num_cols.index(default_num) if default_num in num_cols else 0)
    plot_type = st.radio("Тип графика", ["KDE", "Scatter (против selling_price)"], horizontal=True)
    fig_choice, ax_choice = plt.subplots()
    if plot_type == "KDE":
        train_series = df_eda[choice].dropna()
        test_series = df_test_eda[choice].dropna() if choice in df_test_eda.columns else pd.Series(dtype=float)
        if train_series.empty and test_series.empty:
            st.warning("Нет данных для выбранного признака.")
        else:
            if not train_series.empty:
                sns.kdeplot(train_series, label="train", ax=ax_choice)
            if not test_series.empty:
                sns.kdeplot(test_series, label="test", ax=ax_choice)
            ax_choice.legend()
            st.pyplot(fig_choice)
    else:
        sns.scatterplot(data=df_eda, x=choice, y="selling_price", label="train", alpha=0.5, ax=ax_choice)
        if choice in df_test_eda.columns:
            sns.scatterplot(data=df_test_eda, x=choice, y="selling_price", label="test", alpha=0.5, ax=ax_choice)
        ax_choice.set_ylabel("selling_price")
        ax_choice.legend()
        st.pyplot(fig_choice)

    # боксплот по выбранной категориальной
    cat_cols = [c for c in df_eda.select_dtypes(exclude=[np.number]).columns if c != "name"]
    if cat_cols:
        st.subheader("Категориальный параметр vs цена")
        cat_choice = st.selectbox("Категориальный столбец", options=cat_cols, index=cat_cols.index("fuel") if "fuel" in cat_cols else 0)
        # ограничение категорий, чтобы влезало
        vc = df_eda[cat_choice].value_counts()
        top_vals = vc.head(10).index
        df_cat = df_eda.copy()
        df_cat[cat_choice] = df_cat[cat_choice].where(df_cat[cat_choice].isin(top_vals), "Other")
        fig_cat, ax_cat = plt.subplots()
        sns.boxplot(data=df_cat, x=cat_choice, y="selling_price", ax=ax_cat)
        ax_cat.set_xlabel(cat_choice)
        st.pyplot(fig_cat)


    # Корреляции
    if len(num_cols) > 1:
        st.subheader("Корреляции (Pearson)")
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        corr_p = df_eda[num_cols].corr(method="pearson")
        sns.heatmap(corr_p, annot=False, cmap="coolwarm", center=0, ax=ax4)
        st.pyplot(fig4)

        st.subheader("Корреляции (Spearman)")
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        corr_s = df_eda[num_cols].corr(method="spearman")
        sns.heatmap(corr_s, annot=False, cmap="coolwarm", center=0, ax=ax5)
        st.pyplot(fig5)

    cat_cols = [c for c in df_eda.select_dtypes(exclude=[np.number]).columns if c != "name"]
    if cat_cols:
        st.subheader("ANOVA p-value по категориям")
        anova_vals = {}
        for col in cat_cols:
            groups = [grp["selling_price"].values for _, grp in df_eda.groupby(col) if len(grp) > 1]
            if len(groups) > 1:
                from scipy.stats import f_oneway

                _, pval = f_oneway(*groups)
                anova_vals[col] = pval
        if anova_vals:
            anova_series = pd.Series(anova_vals).sort_values()
            fig6, ax6 = plt.subplots(figsize=(8, 4))
            safe_vals = anova_series.replace(0, anova_series[anova_series > 0].min() * 0.1)
            (-np.log10(safe_vals)).plot(kind="bar", ax=ax6)
            ax6.set_ylabel("-log10 p-value")
            st.pyplot(fig6)

        st.subheader("Cramer's V между категориальными")
        def cramers_v(confusion):
            from scipy.stats import chi2_contingency

            chi2 = chi2_contingency(confusion)[0]
            n = confusion.to_numpy().sum()
            r, k = confusion.shape
            return np.sqrt((chi2 / n) / (min(k - 1, r - 1))) if min(k, r) > 1 else np.nan

        cramers = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
        for i, c1 in enumerate(cat_cols):
            for j, c2 in enumerate(cat_cols):
                if j < i:
                    cramers.loc[c1, c2] = cramers.loc[c2, c1]
                else:
                    table = pd.crosstab(df_eda[c1], df_eda[c2])
                    cramers.loc[c1, c2] = cramers_v(table)
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        sns.heatmap(cramers, annot=False, vmin=0, vmax=1, cmap="magma", ax=ax7)
        st.pyplot(fig7)

    # Pairplot на сэмпле, чтобы не лагало
    if num_cols:
        st.subheader("Pairplot (сэмпл до 500 строк)")
        sample_df = df_eda.sample(min(500, len(df_eda)), random_state=42)
        g = sns.pairplot(sample_df, vars=num_cols, diag_kind=None, corner=True, plot_kws={"alpha": 0.4, "s": 12})
        st.pyplot(g.fig)


def predict_and_show(df: pd.DataFrame, model_bundle, title: str, allow_download: bool = True):
    st.subheader(title)
    pipeline = model_bundle["pipeline"]
    preds = pipeline.predict(df)
    result = df.copy()
    result["predicted_price"] = preds
    st.dataframe(result.head(50))
    if allow_download:
        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("Скачать с предсказаниями (CSV)", data=csv, file_name="predictions.csv")


def weights_section(model_bundle):
    st.header("Веса модели (Ridge)")
    pipe = model_bundle["pipeline"]
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()
    coefs = pd.Series(model.coef_, index=feature_names).sort_values(key=lambda s: s.abs(), ascending=False)
    st.write("Топ 20 по абсолютному весу")
    st.bar_chart(coefs.head(20))
    st.write("Пример полных весов")
    st.dataframe(coefs.reset_index().rename(columns={"index": "feature", 0: "coef"}))


def main():
    st.title("Car price: EDA, инференс и веса модели")

    model_bundle = load_model()
    df_train, df_test = load_data()

    tab1, tab2, tab3 = st.tabs(["EDA", "Предсказания", "Веса"])

    with tab1:
        eda_section(df_train, df_test)

    with tab2:
        st.subheader("1) CSV с новыми объектами")
        uploaded = st.file_uploader("Загрузите CSV с сырыми колонками как в исходных данных", type="csv")
        if uploaded:
            df_upload = pd.read_csv(uploaded)
            predict_and_show(df_upload, model_bundle, "Результаты для загруженного CSV")

        st.subheader("2) Ввод одного объекта вручную")
        with st.form(key="manual_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                year = st.number_input("year", min_value=1990, max_value=2024, value=2015)
                km_driven = st.number_input("km_driven", min_value=0, value=50000)
                mileage = st.number_input("mileage", min_value=0.0, value=18.0, step=0.1)
                engine = st.number_input("engine (cc)", min_value=600.0, value=1200.0, step=50.0)
                max_power = st.number_input("max_power (bhp)", min_value=40.0, value=80.0, step=1.0)
            with col_b:
                torque = st.text_input("torque (например, '113Nm@4500rpm')", value="113Nm@4500rpm")
                seats = st.number_input("seats", min_value=2, max_value=10, value=5)
                fuel = st.selectbox("fuel", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
                seller_type = st.selectbox("seller_type", ["Individual", "Dealer", "Trustmark Dealer"])
                transmission = st.selectbox("transmission", ["Manual", "Automatic"])
                owner = st.selectbox("owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
                name = st.text_input("name (бренд модель)", value="Maruti Swift")
            submitted = st.form_submit_button("Предсказать")

            if submitted:
                single = pd.DataFrame(
                    [
                        {
                            "year": year,
                        "km_driven": km_driven,
                        "mileage": mileage,
                        "engine": engine,
                        "max_power": max_power,
                        "torque": torque,
                        "seats": seats,
                        "fuel": fuel,
                        "seller_type": seller_type,
                        "transmission": transmission,
                        "owner": owner,
                            "name": name,
                        }
                    ]
                )
                predict_and_show(single, model_bundle, "Предсказание для введённого объекта", allow_download=False)

    with tab3:
        weights_section(model_bundle)


if __name__ == "__main__":
    main()
