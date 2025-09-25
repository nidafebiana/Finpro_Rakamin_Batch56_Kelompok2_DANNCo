# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

from utils import (
    load_or_train,
    predict_one,
    predict_batch,
    plot_confusion_matrix_fig,
    expected_savings_curve,
    DEFAULT_COST_PER_CHURN,
)

# EDA helpers (khusus halaman Dashboard)
from utils import (
    get_schema,
    plot_missing_bar,
    plot_target_counts,
    plot_churn_period_counts,
    facet_numeric_hist,
    cat_vs_target,
    corr_heatmap,
)

st.set_page_config(page_title="Employee Churn App", page_icon="📉", layout="wide")

# ===== Sidebar =====
st.sidebar.title("📌 Pengaturan")

# Path dataset
data_path = st.sidebar.text_input(
    "Path dataset (CSV)",
    value="employee_churn_prediction_updated.csv",
    help="Biarkan default jika file ada di folder yang sama dengan app.py",
)

# Threshold & biaya
threshold = st.sidebar.slider(
    "Ambang klasifikasi (threshold)", min_value=0.05, max_value=0.95, value=0.5, step=0.01
)
cost_per_churn = st.sidebar.number_input(
    "Biaya per churn yang bisa dihindari (Rp)",
    min_value=0,
    value=DEFAULT_COST_PER_CHURN,
    step=100_000,
)

page = st.sidebar.radio(
    "Pilih menu",
    ["Dashboard (EDA)", "Prediksi Perorangan", "Prediksi Batch", "Analisis Penghematan Biaya"],
    index=0,
)

# ===== Siapkan model + pipeline (sekali di awal) =====
with st.spinner("Menyiapkan model & pipeline..."):
    artifacts = load_or_train(data_path)  # train jika belum ada artifacts
model = artifacts["model"]
preprocess = artifacts["preprocess"]
eval_info = artifacts["eval_info"]  # X_train, X_test, y_test, y_prob_test, metrics

st.sidebar.success("Model siap ✅")

# ======================================================================================
# 1) DASHBOARD (EDA)
# ======================================================================================
if page == "Dashboard (EDA)":
    st.title("📊 Analisis Data (EDA)")

    # Muat dataset mentah untuk EDA
    try:
        df_raw = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Gagal membaca dataset dari: {data_path}\n\n{e}")
        st.stop()

    # Ringkasan awal
    st.subheader("Ringkasan Dataset")
    n_rows, n_cols = df_raw.shape
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Baris", f"{n_rows:,}")
    c2.metric("Jumlah Kolom", f"{n_cols:,}")
    c3.metric("Duplikat", f"{df_raw.duplicated().sum():,}")
    c4.metric("Total Missing Cells", f"{int(df_raw.isna().sum().sum()):,}")

    with st.expander("🔎 Lihat contoh data (head)"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    with st.expander("ℹ️ Info kolom & tipe data"):
        info_df = pd.DataFrame({
            "col": df_raw.columns,
            "dtype": df_raw.dtypes.astype(str),
            "missing": df_raw.isna().sum().values,
            "unique": [df_raw[c].nunique() for c in df_raw.columns]
        })
        st.dataframe(info_df, use_container_width=True)

    # Skema EDA
    numeric_cols, cat_ordered, cat_nominal, target_col = get_schema()

    # Missing values
    st.subheader("Missing Values")
    fig_miss = plot_missing_bar(df_raw)
    st.pyplot(fig_miss, use_container_width=True)

    # Distribusi target
    st.subheader("Distribusi Target (churn)")
    fig_t = plot_target_counts(df_raw, target_col=target_col)
    st.pyplot(fig_t, use_container_width=True)

    # Distribusi churn period (khusus churn=1)
    if "churn_period" in df_raw.columns:
        st.subheader("Distribusi Churn Period (hanya churn=1)")
        fig_cp = plot_churn_period_counts(df_raw)
        st.pyplot(fig_cp, use_container_width=True)

    # Filter opsional
    st.markdown("### Filter Opsional")
    colf1, colf2 = st.columns(2)
    loc_sel = colf1.multiselect(
        "Work Location",
        sorted(df_raw["work_location"].dropna().unique().tolist()) if "work_location" in df_raw else [],
        []
    )
    edu_sel = colf2.multiselect(
        "Education",
        sorted(df_raw["education"].dropna().unique().tolist()) if "education" in df_raw else [],
        []
    )
    df_view = df_raw.copy()
    if loc_sel and "work_location" in df_view:
        df_view = df_view[df_view["work_location"].isin(loc_sel)]
    if edu_sel and "education" in df_view:
        df_view = df_view[df_view["education"].isin(edu_sel)]

    # Distribusi numerik
    st.subheader("Distribusi Fitur Numerik")
    hue_flag = st.checkbox("Warnai menurut `churn`", value=True)
    fig_num = facet_numeric_hist(
        df_view, numeric_cols,
        hue=target_col if hue_flag and target_col in df_view.columns else None,
        max_cols=4
    )
    st.pyplot(fig_num, use_container_width=True)

    # Distribusi kategorikal vs target
    st.subheader("Distribusi Fitur Kategorikal")
    cats = []
    if cat_ordered: cats += cat_ordered
    if cat_nominal: cats += cat_nominal
    fig_cat = cat_vs_target(df_view, cats, target=target_col if target_col in df_view.columns else None, wrap=3)
    st.pyplot(fig_cat, use_container_width=True)

    # Korelasi numerik
    st.subheader("Heatmap Korelasi (Fitur Numerik)")
    fig_corr = corr_heatmap(df_view, numeric_cols)
    st.pyplot(fig_corr, use_container_width=True)

    st.caption("Analisis EDA menggunakan dataset mentah, tanpa metrik model.")

# ======================================================================================
# 2) PREDIKSI PERORANGAN
# ======================================================================================
elif page == "Prediksi Perorangan":
    st.title("🧍 Prediksi Churn Karyawan")

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=18, max_value=70, value=30)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        education = c3.selectbox("Education", ["High School", "Diploma", "Bachelor"])

        c4, c5, c6 = st.columns(3)
        experience_years = c4.number_input("Experience (years)", min_value=0, max_value=40, value=5)
        monthly_target = c5.number_input("Monthly Target", min_value=40, max_value=200, value=130)
        target_achievement = c6.number_input("Target Achievement (0-1.2)", min_value=0.0, max_value=1.2, value=0.85, step=0.01)

        c7, c8, c9 = st.columns(3)
        working_hours_per_week = c7.number_input("Working Hours / week", min_value=35, max_value=80, value=55)
        overtime_hours_per_week = c8.number_input("Overtime Hours / week", min_value=0, max_value=30, value=8)
        salary = c9.number_input("Salary (Rp)", min_value=2_000_000, max_value=10_000_000, value=5_500_000, step=50_000)

        c10, c11, c12 = st.columns(3)
        commission_rate = c10.number_input("Commission Rate (0-0.1)", min_value=0.0, max_value=0.1, value=0.05, step=0.01)
        job_satisfaction = c11.number_input("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
        work_location = c12.selectbox("Work Location", ["Urban", "Suburban", "Rural"])

        c13, c14, c15 = st.columns(3)
        manager_support_score = c13.number_input("Manager Support Score (1-4)", min_value=1, max_value=4, value=3)
        company_tenure_years = c14.number_input("Company Tenure (years)", min_value=0.0, max_value=40.0, value=2.0, step=0.1)
        marital_status = c15.selectbox("Marital Status", ["Single", "Married"])

        c16, _ = st.columns(2)
        distance_to_office_km = c16.number_input("Distance to Office (km)", min_value=0.0, max_value=60.0, value=15.0, step=0.5)

        submitted = st.form_submit_button("🚀 Prediksi")

    if submitted:
        raw = {
            "age": age,
            "gender": gender,
            "education": education,
            "experience_years": experience_years,
            "monthly_target": monthly_target,
            "target_achievement": target_achievement,
            "working_hours_per_week": working_hours_per_week,
            "overtime_hours_per_week": overtime_hours_per_week,
            "salary": salary,
            "commission_rate": commission_rate,
            "job_satisfaction": job_satisfaction,
            "work_location": work_location,
            "manager_support_score": manager_support_score,
            "company_tenure_years": company_tenure_years,
            "marital_status": marital_status,
            "distance_to_office_km": distance_to_office_km
        }
        proba, label = predict_one(raw, preprocess, model, threshold)
        st.markdown("### Hasil Prediksi")
        st.metric("Probabilitas churn", f"{proba:.2%}")
        st.metric("Prediksi", "Churn" if label == 1 else "Stay")
        if label == 1:
            st.success(f"Perkiraan **penghematan** jika churn dicegah: **Rp{cost_per_churn:,.0f}**")
        else:
            st.info("Risiko churn relatif rendah pada threshold saat ini.")

# ======================================================================================
# 3) PREDIKSI BATCH
# ======================================================================================
elif page == "Prediksi Batch":
    st.title("🗂️ Prediksi Churn (Batch CSV)")
    st.caption("Unggah CSV dengan kolom yang sama seperti dataset asli (boleh menyertakan 'employee_id').")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_input = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            st.stop()

        with st.spinner("Memproses & memprediksi..."):
            try:
                df_out = predict_batch(df_input, preprocess, model, threshold)
            except Exception as e:
                st.error(f"Terjadi error saat prediksi batch: {e}")
                st.stop()

        st.success("Selesai ✅")
        st.dataframe(df_out.head(20), use_container_width=True)
        st.download_button(
            "⬇️ Unduh hasil (CSV)",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info("Tidak ada file yang diunggah.")

# ======================================================================================
# 4) ANALISIS PENGHEMATAN BIAYA
# ======================================================================================
else:
    st.title("💰 Analisis Penghematan Biaya")

    colA, colB, colC = st.columns(3)
    eff = colA.slider("Efektivitas intervensi (%)", 0, 100, 30, 1,
                      help="Proporsi prediksi 'churn' (TP) yang berhasil dicegah setelah intervensi.")
    cost_action = colB.number_input("Biaya intervensi / karyawan (Rp)", min_value=0, value=250_000, step=50_000)
    n_preview = colC.slider("Titik evaluasi threshold", 10, 200, 100, step=10)

    st.subheader("Kurva Expected Net Saving vs Threshold")
    fig_sav = expected_savings_curve(
        eval_info["y_test"],
        eval_info["y_prob_test"],
        cost_per_churn=cost_per_churn,
        intervention_cost=cost_action,
        effectiveness=eff/100.0,
        n_points=n_preview
    )
    st.pyplot(fig_sav, use_container_width=True)

    st.subheader("Ringkasan pada Threshold Saat Ini")
    fig_cm = plot_confusion_matrix_fig(eval_info["y_test"], eval_info["y_prob_test"], threshold)
    st.pyplot(fig_cm, use_container_width=True)

    # Hitung ringkasan finansial
    y_true = eval_info["y_test"].values
    y_prob = eval_info["y_prob_test"]
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    n_intervened = int((y_pred == 1).sum())
    prevented = int(round(tp * (eff/100.0)))  # konservatif: hanya TP yang bisa dicegah
    gross_saving = prevented * cost_per_churn
    total_intervention = n_intervened * cost_action
    net_saving = gross_saving - total_intervention

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("TP (prediksi churn benar)", f"{tp}")
    s2.metric("Jumlah diintervensi", f"{n_intervened}")
    s3.metric("Prevented churn (estimasi)", f"{prevented}")
    s4.metric("Net saving (estimasi)", f"Rp{net_saving:,.0f}")
