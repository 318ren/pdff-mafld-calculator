import streamlit as st
import pandas as pd
import catboost
import plotly.express as px

# -------------------------- Load model --------------------------
model = catboost.CatBoostClassifier()
model.load_model("catboost_model.cbm")

# Feature order must be identical to training!
feature_names = [
    "age","ALT","ALB","DBIL","IBIL","TG","HDL","WBC","PLT","PCT",
    "Fatpercentage","WC","HR","MIN","BMR","AST","ALP","GGT","TC",
    "TBA","LPA","HB","RBC","Weight","BMI","TBIL","LDL"
]

# -------------------------- Page title --------------------------
st.title("CatBoost-Based Prediction Calculator")
st.subheader("Input clinical features and click Predict to get four-class probabilities")
st.caption("Powered by CatBoost, Streamlit, Pandas and Plotly")

# -------------------------- Sidebar inputs --------------------------
with st.sidebar:
    st.header("Feature Input")
    age = st.number_input("Age", value=None, step=0.1)
    ALT = st.number_input("ALT", value=None, step=0.1)
    ALB = st.number_input("ALB", value=None, step=0.1)
    DBIL = st.number_input("DBIL", value=None, step=0.1)
    IBIL = st.number_input("IBIL", value=None, step=0.1)
    TG = st.number_input("TG", value=None, step=0.1)
    HDL = st.number_input("HDL", value=None, step=0.1)
    WBC = st.number_input("WBC", value=None, step=0.1)
    PLT = st.number_input("PLT", value=None, step=0.1)
    PCT = st.number_input("PCT", value=None, step=0.1)
    Fatpercentage = st.number_input("Fat percentage", value=None, step=0.1)
    WC = st.number_input("Waist circumference", value=None, step=0.1)
    HR = st.number_input("Hip circumference", value=None, step=0.1)
    MIN = st.number_input("MIN", value=None, step=0.1)
    BMR = st.number_input("Basal metabolic rate (BMR)", value=None, step=0.1)
    AST = st.number_input("AST", value=None, step=0.1)
    ALP = st.number_input("ALP", value=None, step=0.1)
    GGT = st.number_input("GGT", value=None, step=0.1)
    TC = st.number_input("TC", value=None, step=0.1)
    TBA = st.number_input("TBA", value=None, step=0.1)
    LPA = st.number_input("LPA", value=None, step=0.1)
    HB = st.number_input("HB", value=None, step=0.1)
    RBC = st.number_input("RBC", value=None, step=0.1)
    Weight = st.number_input("Weight", value=None, step=0.1)
    BMI = st.number_input("BMI", value=None, step=0.1)
    TBIL = st.number_input("TBIL", value=None, step=0.1)
    LDL = st.number_input("LDL", value=None, step=0.1)

    run_btn = st.button("Predict", type="primary")

# -------------------------- Prediction pipeline --------------------------
if run_btn:
    var_list = [
        age,ALT,ALB,DBIL,IBIL,TG,HDL,WBC,PLT,PCT,
        Fatpercentage,WC,HR,MIN,BMR,AST,ALP,GGT,TC,
        TBA,LPA,HB,RBC,Weight,BMI,TBIL,LDL
    ]
    # 校验：检查是否有空值
    if any(x is None for x in var_list):
        st.error("Please fill in all clinical features before prediction.")
    else:
        input_data = pd.DataFrame([var_list], columns=feature_names)
        prob = model.predict_proba(input_data)[0]
        res_df = pd.DataFrame({
            "Category": ["Class 0","Class 1","Class 2","Class 3"],
            "Probability": prob
        })
        res_df["Probability (%)"] = (res_df["Probability"] * 100).round(1)

        st.subheader("Prediction results (four-class probabilities)")
        st.dataframe(res_df, use_container_width=True)

        # Plot (ggplot-style)
        fig = px.bar(
            res_df,
            x="Category",
            y="Probability",
            color="Category",
            text="Probability (%)",
            range_y=[0, 1],
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            title="Probability distribution across categories",
            title_x=0.5,
            yaxis_title="Predicted probability",
            xaxis_title="Outcome category",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
