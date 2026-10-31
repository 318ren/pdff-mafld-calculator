import streamlit as st
import pandas as pd
import catboost
import plotly.express as px

# -------------------------- 加载模型 --------------------------
model = catboost.CatBoostClassifier()
model.load_model("catboost_model.cbm")

# 特征顺序必须和训练时完全一致！
feature_names = [
    "age","ALT","ALB","DBIL","IBIL","TG","HDL","WBC","PLT","PCT",
    "Fatpercentage","WC","HR","MIN","BMR","AST","ALP","GGT","TC",
    "TBA","LPA","HB","RBC","Weight","BMI","TBIL","LDL"
]

# -------------------------- 页面标题 --------------------------
st.set_page_config(page_title="PDFF MAFLD预测计算器", layout="wide")
st.title("CatBoost模型预测计算器")
st.markdown("### 请输入各项临床指标，点击预测获取4分类概率")

# -------------------------- 侧边栏输入框 --------------------------
with st.sidebar:
    st.header("指标输入")
    age = st.number_input("年龄 age", value=None, step=0.1)
    ALT = st.number_input("ALT", value=None, step=0.1)
    ALB = st.number_input("ALB", value=None, step=0.1)
    DBIL = st.number_input("DBIL", value=None, step=0.1)
    IBIL = st.number_input("IBIL", value=None, step=0.1)
    TG = st.number_input("TG", value=None, step=0.1)
    HDL = st.number_input("HDL", value=None, step=0.1)
    WBC = st.number_input("WBC", value=None, step=0.1)
    PLT = st.number_input("PLT", value=None, step=0.1)
    PCT = st.number_input("PCT", value=None, step=0.1)
    Fatpercentage = st.number_input("脂肪百分比 Fatpercentage", value=None, step=0.1)
    WC = st.number_input("腰围 WC", value=None, step=0.1)
    HR = st.number_input("心率 HR", value=None, step=0.1)
    MIN = st.number_input("MIN", value=None, step=0.1)
    BMR = st.number_input("基础代谢 BMR", value=None, step=0.1)
    AST = st.number_input("AST", value=None, step=0.1)
    ALP = st.number_input("ALP", value=None, step=0.1)
    GGT = st.number_input("GGT", value=None, step=0.1)
    TC = st.number_input("TC", value=None, step=0.1)
    TBA = st.number_input("TBA", value=None, step=0.1)
    LPA = st.number_input("LPA", value=None, step=0.1)
    HB = st.number_input("HB", value=None, step=0.1)
    RBC = st.number_input("RBC", value=None, step=0.1)
    Weight = st.number_input("体重 Weight", value=None, step=0.1)
    BMI = st.number_input("BMI", value=None, step=0.1)
    TBIL = st.number_input("TBIL", value=None, step=0.1)
    LDL = st.number_input("LDL", value=None, step=0.1)

    run_btn = st.button("开始预测", type="primary")

# -------------------------- 预测逻辑 --------------------------
if run_btn:
    input_data = pd.DataFrame([[
        age,ALT,ALB,DBIL,IBIL,TG,HDL,WBC,PLT,PCT,
        Fatpercentage,WC,HR,MIN,BMR,AST,ALP,GGT,TC,
        TBA,LPA,HB,RBC,Weight,BMI,TBIL,LDL
    ]], columns=feature_names)

    prob = model.predict_proba(input_data)[0]
    res_df = pd.DataFrame({
        "Category": ["Class0","Class1","Class2","Class3"],
        "Probability": prob
    })
    res_df["Probability(%)"] = (res_df["Probability"] * 100).round(1)

    st.subheader("预测结果（4分类概率）")
    st.dataframe(res_df, use_container_width=True)

    # 绘图（和ggplot风格接近）
    fig = px.bar(
        res_df,
        x="Category",
        y="Probability",
        color="Category",
        text="Probability(%)",
        range_y=[0, 1],
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(
        title="各分类预测概率分布",
        title_x=0.5,
        yaxis_title="预测概率",
        xaxis_title="结局分类",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- 依赖声明文件 requirements.txt --------------------------
# 新建一个纯文本文件，命名为 requirements.txt，内容如下：
"""
catboost
streamlit
pandas
plotly
"""
