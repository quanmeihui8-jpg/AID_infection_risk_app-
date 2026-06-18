import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb


# ============================================================
# 1. 本地模型路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "XGB_final_model_specOverSensEq1.json"

# 三层风险阈值：来自你保存的 XGB_3level_risk_cutoff_nestedCV.rds
T1_LOW_TO_INTERMEDIATE = 0.06623795
T2_INTERMEDIATE_TO_HIGH = 0.15242195


# ============================================================
# 2. 固定模型输入变量顺序
#    注意：这里的变量名是模型后台变量名，不能随意修改
# ============================================================

FEATURE_ORDER = [
    "PCT",
    "CRP",
    "WBC.HP",
    "Fever.1",
    "LY.",
    "LDH",
    "NLR",
    "K",
    "IgG",
    "Age",
    "PLR",
    "Alb",
    "CKDEPI",
    "CK",
    "ESR.CRP",
    "DD",
    "Na",
    "Fg",
    "LY",
    "IgA"
]


# ============================================================
# 3. 网页显示用英文全称和单位
# ============================================================

FEATURE_LABELS = {
    "PCT": "Procalcitonin (ng/mL)",
    "CRP": "C-reactive Protein (mg/L)",
    "WBC.HP": "White Blood Cells per High-Power Field",
    "Fever.1": "Fever",
    "LY.": "Lymphocyte Percentage (%)",
    "LDH": "Lactate Dehydrogenase (IU/L)",
    "NLR": "Neutrophil-to-Lymphocyte Ratio",
    "K": "Serum Potassium (mmol/L)",
    "IgG": "Immunoglobulin G (IU/mL)",
    "Age": "Age (years)",
    "PLR": "Platelet-to-Lymphocyte Ratio",
    "Alb": "Serum Albumin (g/L)",
    "CKDEPI": "Estimated Glomerular Filtration Rate",
    "CK": "Creatine Kinase (IU/L)",
    "ESR.CRP": "Erythrocyte Sedimentation Rate to C-reactive Protein Ratio",
    "DD": "D-dimer (μg/mL FEU)",
    "Na": "Serum Sodium (mmol/L)",
    "Fg": "Fibrinogen (g/L)",
    "LY": "Lymphocyte Count",
    "IgA": "Immunoglobulin A (IU/mL)"
}


# ============================================================
# 4. 读取 XGBoost 模型
# ============================================================

@st.cache_resource
def load_xgb_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到模型文件：{model_path}\n"
            "请确认 XGB_final_model_specOverSensEq1.json 已经放在 "
            "/Users/quanmeihui/Desktop/感染预测-半监督 文件夹中。"
        )

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


# ============================================================
# 5. 风险分层函数
# ============================================================

def classify_risk(probability: float):
    if probability < T1_LOW_TO_INTERMEDIATE:
        return "Low risk", "Low-risk group"
    elif probability < T2_INTERMEDIATE_TO_HIGH:
        return "Intermediate risk", "Intermediate-risk group"
    else:
        return "High risk", "High-risk group"


def generate_explanation(probability: float, risk_group: str):
    prob_percent = probability * 100

    if risk_group == "Low risk":
        return (
            f"The predicted probability of infection is {prob_percent:.1f}%. "
            "The patient is classified into the low-risk group. "
            "Clinical observation and dynamic reassessment may be considered according to the patient's symptoms, signs, and laboratory changes."
        )

    if risk_group == "Intermediate risk":
        return (
            f"The predicted probability of infection is {prob_percent:.1f}%. "
            "The patient is classified into the intermediate-risk group. "
            "Further assessment is recommended by integrating fever, inflammatory markers, imaging findings, microbiological tests, and clinical manifestations."
        )

    return (
        f"The predicted probability of infection is {prob_percent:.1f}%. "
        "The patient is classified into the high-risk group. "
        "Clinicians should pay close attention to the possibility of infection and consider comprehensive evaluation with microbiological tests, imaging, and indications for anti-infective therapy."
    )


def risk_color_style(risk_group: str):
    if risk_group == "Low risk":
        return "#2E7D32"
    elif risk_group == "Intermediate risk":
        return "#F9A825"
    else:
        return "#C62828"


# ============================================================
# 6. 构建输入数据
# ============================================================

def build_input_dataframe(
    age,
    fever,
    pct,
    crp,
    wbc_hp,
    ly_percent,
    ldh,
    nlr,
    potassium,
    igg,
    plr,
    alb,
    egfr,
    ck,
    esr_crp,
    ddimer,
    sodium,
    fibrinogen,
    lymphocyte_count,
    iga
):
    fever_value = 1 if fever == "Yes" else 0

    input_df = pd.DataFrame([{
        "PCT": pct,
        "CRP": crp,
        "WBC.HP": wbc_hp,
        "Fever.1": fever_value,
        "LY.": ly_percent,
        "LDH": ldh,
        "NLR": nlr,
        "K": potassium,
        "IgG": igg,
        "Age": age,
        "PLR": plr,
        "Alb": alb,
        "CKDEPI": egfr,
        "CK": ck,
        "ESR.CRP": esr_crp,
        "DD": ddimer,
        "Na": sodium,
        "Fg": fibrinogen,
        "LY": lymphocyte_count,
        "IgA": iga
    }])

    input_df = input_df[FEATURE_ORDER]

    return input_df


def predict_probability(model, input_df):
    dmatrix = xgb.DMatrix(input_df)
    probability = float(model.predict(dmatrix)[0])
    return probability


# ============================================================
# 7. Streamlit 网页主体
# ============================================================

st.set_page_config(
    page_title="AID Infection Risk Prediction Tool",
    page_icon="🧬",
    layout="wide"
)

st.title("AID Infection Risk Prediction Tool")
st.caption(
    "A web-based prototype for infection risk prediction and three-level risk stratification "
    "in patients with autoimmune diseases."
)

st.markdown("---")

try:
    model = load_xgb_model(MODEL_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()


# ============================================================
# 8. 页面布局
# ============================================================

left_col, right_col = st.columns([1.15, 1.0])


with left_col:
    st.subheader("Input Clinical and Laboratory Variables")

    st.markdown(
        """
        Please enter the patient's clinical and laboratory indicators.  
        All variables should be entered using the same units as those used during model development.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            FEATURE_LABELS["Age"],
            min_value=0.0,
            max_value=120.0,
            value=50.0,
            step=1.0,
            format="%.1f"
        )

        fever = st.selectbox(
            FEATURE_LABELS["Fever.1"],
            options=["No", "Yes"],
            index=0
        )

        pct = st.number_input(
            FEATURE_LABELS["PCT"],
            min_value=0.0,
            value=0.05,
            step=0.01,
            format="%.2f"
        )

        crp = st.number_input(
            FEATURE_LABELS["CRP"],
            min_value=0.0,
            value=6.0,
            step=1.0,
            format="%.2f"
        )

        wbc_hp = st.number_input(
            FEATURE_LABELS["WBC.HP"],
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f"
        )

        ly_percent = st.number_input(
            FEATURE_LABELS["LY."],
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            format="%.2f"
        )

        ldh = st.number_input(
            FEATURE_LABELS["LDH"],
            min_value=0.0,
            value=200.0,
            step=10.0,
            format="%.2f"
        )

        nlr = st.number_input(
            FEATURE_LABELS["NLR"],
            min_value=0.0,
            value=3.0,
            step=0.1,
            format="%.2f"
        )

        potassium = st.number_input(
            FEATURE_LABELS["K"],
            min_value=0.0,
            value=4.0,
            step=0.1,
            format="%.2f"
        )

        igg = st.number_input(
            FEATURE_LABELS["IgG"],
            min_value=0.0,
            value=10.0,
            step=0.1,
            format="%.2f"
        )

    with col2:
        plr = st.number_input(
            FEATURE_LABELS["PLR"],
            min_value=0.0,
            value=150.0,
            step=1.0,
            format="%.2f"
        )

        alb = st.number_input(
            FEATURE_LABELS["Alb"],
            min_value=0.0,
            value=40.0,
            step=1.0,
            format="%.2f"
        )

        egfr = st.number_input(
            FEATURE_LABELS["CKDEPI"],
            min_value=0.0,
            value=90.0,
            step=1.0,
            format="%.2f"
        )

        ck = st.number_input(
            FEATURE_LABELS["CK"],
            min_value=0.0,
            value=80.0,
            step=10.0,
            format="%.2f"
        )

        esr_crp = st.number_input(
            FEATURE_LABELS["ESR.CRP"],
            min_value=0.0,
            value=2.0,
            step=0.1,
            format="%.2f"
        )

        ddimer = st.number_input(
            FEATURE_LABELS["DD"],
            min_value=0.0,
            value=0.5,
            step=0.1,
            format="%.2f"
        )

        sodium = st.number_input(
            FEATURE_LABELS["Na"],
            min_value=0.0,
            value=140.0,
            step=1.0,
            format="%.2f"
        )

        fibrinogen = st.number_input(
            FEATURE_LABELS["Fg"],
            min_value=0.0,
            value=3.0,
            step=0.1,
            format="%.2f"
        )

        lymphocyte_count = st.number_input(
            FEATURE_LABELS["LY"],
            min_value=0.0,
            value=1.2,
            step=0.1,
            format="%.2f"
        )

        iga = st.number_input(
            FEATURE_LABELS["IgA"],
            min_value=0.0,
            value=2.0,
            step=0.1,
            format="%.2f"
        )

    predict_button = st.button("Calculate Infection Risk", type="primary")


with right_col:
    st.subheader("Prediction Result")

    if predict_button:
        input_df = build_input_dataframe(
            age=age,
            fever=fever,
            pct=pct,
            crp=crp,
            wbc_hp=wbc_hp,
            ly_percent=ly_percent,
            ldh=ldh,
            nlr=nlr,
            potassium=potassium,
            igg=igg,
            plr=plr,
            alb=alb,
            egfr=egfr,
            ck=ck,
            esr_crp=esr_crp,
            ddimer=ddimer,
            sodium=sodium,
            fibrinogen=fibrinogen,
            lymphocyte_count=lymphocyte_count,
            iga=iga
        )

        probability = predict_probability(model, input_df)
        risk_group, risk_group_full = classify_risk(probability)
        explanation = generate_explanation(probability, risk_group)
        color = risk_color_style(risk_group)

        st.markdown(
            f"""
            <div style="
                padding: 24px;
                border-radius: 14px;
                border: 1px solid #E0E0E0;
                background-color: #FAFAFA;
            ">
                <h2 style="margin-bottom: 4px;">Predicted Probability of Infection</h2>
                <h1 style="color: {color}; font-size: 48px; margin-top: 0;">
                    {probability * 100:.1f}%
                </h1>
                <h2 style="color: {color};">
                    {risk_group_full}
                </h2>
                <p style="font-size: 16px; line-height: 1.6;">
                    {explanation}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Input Variable Check")

        display_df = input_df.T.rename(columns={0: "Input Value"})
        display_df.index = [FEATURE_LABELS[x] for x in display_df.index]
        st.dataframe(display_df, use_container_width=True)

    else:
        st.info("Please enter the patient's variables and click 'Calculate Infection Risk'.")


# ============================================================
# 9. 模型说明
# ============================================================

st.markdown("---")

with st.expander("Model Information and Risk Stratification Thresholds", expanded=True):
    st.markdown(
        f"""
        This web tool uses the final XGBoost model to estimate the probability of infection.  
        The predicted probability is further categorized into three risk groups using the following thresholds:

        - **Low-risk group**: predicted probability < **{T1_LOW_TO_INTERMEDIATE:.4f}**
        - **Intermediate-risk group**: **{T1_LOW_TO_INTERMEDIATE:.4f} ≤ predicted probability < {T2_INTERMEDIATE_TO_HIGH:.4f}**
        - **High-risk group**: predicted probability ≥ **{T2_INTERMEDIATE_TO_HIGH:.4f}**

        This tool is intended for clinical decision support and should not replace physician judgment.
        """
    )


# ============================================================
# 10. 批量预测功能，可选
# ============================================================

st.markdown("---")
st.subheader("Optional: Batch Prediction from CSV or Excel")

st.markdown(
    """
    Upload a CSV or Excel file containing the 20 model variables.  
    The file columns should use the backend variable names, including:
    """
)

st.code(", ".join(FEATURE_ORDER))

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file for batch prediction",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            batch_df = pd.read_csv(uploaded_file)
        else:
            batch_df = pd.read_excel(uploaded_file)

        missing_cols = [col for col in FEATURE_ORDER if col not in batch_df.columns]

        if missing_cols:
            st.error(
                "The uploaded file is missing the following required columns: "
                + ", ".join(missing_cols)
            )
        else:
            batch_input = batch_df[FEATURE_ORDER].copy()
            batch_dmatrix = xgb.DMatrix(batch_input)
            batch_prob = model.predict(batch_dmatrix)

            result_df = batch_df.copy()
            result_df["Predicted_Probability"] = batch_prob
            result_df["Predicted_Probability_Percent"] = batch_prob * 100

            result_df["Risk_Group"] = [
                classify_risk(float(p))[1] for p in batch_prob
            ]

            st.success("Batch prediction completed.")
            st.dataframe(result_df, use_container_width=True)

            csv_data = result_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Download Batch Prediction Results",
                data=csv_data,
                file_name="AID_infection_risk_prediction_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Batch prediction failed: {e}")