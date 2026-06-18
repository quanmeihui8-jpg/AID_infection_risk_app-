from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


# ============================================================
# 1. Model path and risk thresholds
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "XGB_final_model_specOverSensEq1.json"

# Three-level risk thresholds
T1_LOW_TO_INTERMEDIATE = 0.06623795
T2_INTERMEDIATE_TO_HIGH = 0.15242195


# ============================================================
# 2. Fixed model input order
#    Do not change this order unless the model is retrained.
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
# 3. Display labels
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
# 4. Missing value handling
# ============================================================

MISSING_TOKENS = {
    "",
    "缺失",
    "无",
    "无记录",
    "未检测",
    "missing",
    "Missing",
    "MISSING",
    "NA",
    "N/A",
    "na",
    "n/a",
    "NaN",
    "nan",
    "None",
    "none",
    "-"
}


def parse_numeric_or_missing(value, variable_name):
    """
    Convert user input into float or np.nan.
    Users may enter a numeric value or missing-value tokens such as '缺失', 'missing', or 'NA'.
    """
    if value is None:
        return np.nan

    value = str(value).strip()

    if value in MISSING_TOKENS:
        return np.nan

    try:
        return float(value)
    except ValueError:
        st.error(
            f"Invalid input for {variable_name}: '{value}'. "
            "Please enter a numeric value or 'missing'/'缺失'."
        )
        st.stop()


def text_numeric_input(label, default_value=""):
    """
    Text input that accepts either numeric values or missing-value tokens.
    """
    return st.text_input(
        label,
        value=default_value,
        placeholder="Enter a value or missing/缺失"
    )


def format_value_for_display(x):
    """
    Display missing values as 'Missing' and numeric values with one decimal place.
    """
    if pd.isna(x):
        return "Missing"
    return f"{float(x):.1f}"


# ============================================================
# 5. Load XGBoost model
# ============================================================

@st.cache_resource
def load_xgb_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please make sure XGB_final_model_specOverSensEq1.json is in the same folder as app.py."
        )

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


# ============================================================
# 6. Risk stratification functions
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
# 7. Build input data, predict, and calculate SHAP
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
    if fever == "Yes":
        fever_value = 1.0
    elif fever == "No":
        fever_value = 0.0
    else:
        fever_value = np.nan

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

    return input_df[FEATURE_ORDER]


def predict_probability(model, input_df):
    input_df = input_df.astype(float)
    dmatrix = xgb.DMatrix(input_df, missing=np.nan)
    probability = float(model.predict(dmatrix)[0])
    return probability


def calculate_shap_contributions(model, input_df):
    """
    Calculate individual-level SHAP contributions using XGBoost built-in pred_contribs.
    SHAP values are on the model margin scale, usually log-odds for binary classification.
    Positive values increase predicted infection risk; negative values decrease predicted infection risk.
    """
    input_df = input_df.astype(float)
    dmatrix = xgb.DMatrix(input_df, missing=np.nan)

    contribs = model.predict(dmatrix, pred_contribs=True)[0]

    shap_values = contribs[:-1]
    base_value = contribs[-1]

    shap_df = pd.DataFrame({
        "Feature": FEATURE_ORDER,
        "Variable": [FEATURE_LABELS[x] for x in FEATURE_ORDER],
        "Input Value": input_df.iloc[0].values,
        "SHAP Value": shap_values
    })

    shap_df["Abs SHAP Value"] = shap_df["SHAP Value"].abs()
    shap_df["Direction"] = np.where(
        shap_df["SHAP Value"] > 0,
        "Increases predicted infection risk",
        "Decreases predicted infection risk"
    )

    shap_df["Input Value"] = shap_df["Input Value"].apply(format_value_for_display)
    shap_df = shap_df.sort_values("Abs SHAP Value", ascending=False)

    return shap_df, base_value


def plot_individual_shap_bar(shap_df, top_n=8):
    """
    Plot top N SHAP contributions for one patient.
    """
    plot_df = shap_df.head(top_n).copy()
    plot_df = plot_df.sort_values("SHAP Value", ascending=True)

    colors = [
        "#E74C3C" if v > 0 else "#2ECC71"
        for v in plot_df["SHAP Value"]
    ]

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    ax.barh(plot_df["Variable"], plot_df["SHAP Value"], color=colors, height=0.55)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.8)

    ax.set_title("Top feature contributions", color="white", fontsize=11, pad=8)
    ax.set_xlabel("SHAP value", color="white", fontsize=9)
    ax.set_ylabel("")

    ax.tick_params(axis="x", colors="white", labelsize=8)
    ax.tick_params(axis="y", colors="white", labelsize=8)

    for spine in ax.spines.values():
        spine.set_color("#666666")

    ax.grid(axis="x", linestyle="--", alpha=0.12, color="white")

    plt.tight_layout()
    return fig


# ============================================================
# 8. Streamlit page
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


left_col, right_col = st.columns([1.15, 1.0])


with left_col:
    st.subheader("Input Clinical and Laboratory Variables")

    st.markdown(
        """
        Please enter the patient's clinical and laboratory indicators.  
        All variables should be entered using the same units as those used during model development.
        """
    )

    st.info(
        "If a variable is unavailable, enter 'missing' or '缺失'. "
        "Do not enter 0 for missing values unless the true measured value is 0."
    )

    col1, col2 = st.columns(2)

    with col1:
        age = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["Age"], default_value="50.0"),
            FEATURE_LABELS["Age"]
        )

        fever = st.selectbox(
            FEATURE_LABELS["Fever.1"],
            options=["No", "Yes", "Missing"],
            index=0
        )

        pct = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["PCT"], default_value="0.1"),
            FEATURE_LABELS["PCT"]
        )

        crp = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["CRP"], default_value="6.0"),
            FEATURE_LABELS["CRP"]
        )

        wbc_hp = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["WBC.HP"], default_value="0.0"),
            FEATURE_LABELS["WBC.HP"]
        )

        ly_percent = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["LY."], default_value="20.0"),
            FEATURE_LABELS["LY."]
        )

        ldh = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["LDH"], default_value="200.0"),
            FEATURE_LABELS["LDH"]
        )

        nlr = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["NLR"], default_value="3.0"),
            FEATURE_LABELS["NLR"]
        )

        potassium = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["K"], default_value="4.0"),
            FEATURE_LABELS["K"]
        )

        igg = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["IgG"], default_value="10.0"),
            FEATURE_LABELS["IgG"]
        )

    with col2:
        plr = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["PLR"], default_value="150.0"),
            FEATURE_LABELS["PLR"]
        )

        alb = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["Alb"], default_value="40.0"),
            FEATURE_LABELS["Alb"]
        )

        egfr = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["CKDEPI"], default_value="90.0"),
            FEATURE_LABELS["CKDEPI"]
        )

        ck = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["CK"], default_value="80.0"),
            FEATURE_LABELS["CK"]
        )

        esr_crp = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["ESR.CRP"], default_value="2.0"),
            FEATURE_LABELS["ESR.CRP"]
        )

        ddimer = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["DD"], default_value="0.5"),
            FEATURE_LABELS["DD"]
        )

        sodium = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["Na"], default_value="140.0"),
            FEATURE_LABELS["Na"]
        )

        fibrinogen = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["Fg"], default_value="3.0"),
            FEATURE_LABELS["Fg"]
        )

        lymphocyte_count = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["LY"], default_value="1.2"),
            FEATURE_LABELS["LY"]
        )

        iga = parse_numeric_or_missing(
            text_numeric_input(FEATURE_LABELS["IgA"], default_value="2.0"),
            FEATURE_LABELS["IgA"]
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

        with st.container(border=True):
            result_col1, result_col2 = st.columns([1, 1.3])

            with result_col1:
                st.caption("Predicted Probability of Infection")
                st.markdown(
                    f"""
                    <div style="
                        color: {color};
                        font-size: 42px;
                        font-weight: 700;
                        line-height: 1.2;
                        margin-top: -8px;
                    ">
                        {probability * 100:.1f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result_col2:
                st.caption("Risk Group")
                st.markdown(
                    f"""
                    <div style="
                        color: {color};
                        font-size: 28px;
                        font-weight: 700;
                        line-height: 1.3;
                        margin-top: -4px;
                    ">
                        {risk_group_full}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with st.expander("Interpretation", expanded=False):
            st.write(explanation)

        with st.expander("Individual SHAP Explanation", expanded=False):
            shap_df, base_value = calculate_shap_contributions(model, input_df)

            st.markdown(
                """
                Positive SHAP values increase the predicted infection risk, whereas negative SHAP values decrease the predicted infection risk.  
                SHAP values reflect each feature's contribution to the individual prediction and should not be interpreted as causal effects.
                """
            )

            fig = plot_individual_shap_bar(shap_df, top_n=8)
            st.pyplot(fig, clear_figure=True)

            st.markdown("#### SHAP Contribution Table")
            st.dataframe(
                shap_df[
                    [
                        "Variable",
                        "Input Value",
                        "SHAP Value",
                        "Direction"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        missing_count = int(input_df.isna().sum().sum())
        if missing_count > 0:
            st.warning(
                f"{missing_count} missing value(s) were entered. "
                "The prediction was generated using XGBoost's built-in missing-value handling. "
                "Please interpret the result with caution."
            )

        with st.expander("Input Variable Check", expanded=False):
            display_df = input_df.T.rename(columns={0: "Input Value"})
            display_df.index = [FEATURE_LABELS[x] for x in display_df.index]
            display_df["Input Value"] = display_df["Input Value"].apply(format_value_for_display)

            st.dataframe(display_df, use_container_width=True)

    else:
        st.info("Please enter the patient's variables and click 'Calculate Infection Risk'.")


# ============================================================
# 9. Model information
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

        Missing values are passed to the XGBoost model as `NaN`.  
        SHAP values are used to provide individual-level interpretation of model predictions.  
        This tool is intended for clinical decision support and should not replace physician judgment.
        """
    )


# ============================================================
# 10. Optional batch prediction
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

            for col in FEATURE_ORDER:
                batch_input[col] = batch_input[col].replace(list(MISSING_TOKENS), np.nan)

            # Support Yes/No/Missing text values for Fever.1 in batch files
            if "Fever.1" in batch_input.columns:
                batch_input["Fever.1"] = batch_input["Fever.1"].replace({
                    "Yes": 1,
                    "yes": 1,
                    "Y": 1,
                    "y": 1,
                    "No": 0,
                    "no": 0,
                    "N": 0,
                    "n": 0,
                    "Missing": np.nan,
                    "missing": np.nan,
                    "缺失": np.nan
                })

            for col in FEATURE_ORDER:
                batch_input[col] = pd.to_numeric(batch_input[col], errors="coerce")

            batch_dmatrix = xgb.DMatrix(batch_input.astype(float), missing=np.nan)
            batch_prob = model.predict(batch_dmatrix)

            result_df = batch_df.copy()
            result_df["Predicted_Probability"] = np.round(batch_prob, 3)
            result_df["Predicted_Probability_Percent"] = np.round(batch_prob * 100, 1)
            result_df["Risk_Group"] = [classify_risk(float(p))[1] for p in batch_prob]

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
