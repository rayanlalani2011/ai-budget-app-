import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

st.title("AI Budget Coach")
st.write("Enter your income and spending to analyze your budget using machine learning.")

# -------- Inputs --------
income = st.number_input("Monthly Income", min_value=0)
food = st.number_input("Food Spending", min_value=0)
transport = st.number_input("Transport Spending", min_value=0)
entertainment = st.number_input("Entertainment Spending", min_value=0)

total_spending = food + transport + entertainment

# -------- Train ML Model --------
data = {
    "income": [1000, 1000, 1000, 1500, 1500, 1500, 2000, 2000, 2000],
    "total_spending": [700, 900, 1100, 1000, 1300, 1600, 1200, 1800, 2200],
    "label": [
        "Healthy", "Tight", "Overspending",
        "Healthy", "Tight", "Overspending",
        "Healthy", "Tight", "Overspending"
    ]
}

df = pd.DataFrame(data)
X = df[["income", "total_spending"]]
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# -------- Prediction --------
if st.button("Analyze Budget"):
    prediction = model.predict(np.array([[income, total_spending]]))[0]

    st.subheader("Results")
    st.write(f"**Total Spending:** {total_spending}")
    st.write(f"**Budget Classification (ML):** {prediction}")

    if prediction == "Healthy":
        st.success("Your spending is well balanced compared to your income.")
    elif prediction == "Tight":
        st.warning("Your budget is tight. Small changes could help.")
    else:
        st.error("You are spending more than your income.")

