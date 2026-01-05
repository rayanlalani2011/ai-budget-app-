import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import openai

# --- CONFIGURE YOUR OPENAI API KEY ---
# Replace 'YOUR_API_KEY_HERE' with your actual key
openai.api_key = "YOUR_API_KEY_HERE"

st.title("AI Budget Coach")
st.write("Enter your monthly income and spending, and the app will classify your budget and explain it in simple language.")

# --- USER INPUT ---
income = st.number_input("Monthly Income", min_value=0)
food = st.number_input("Food Spending", min_value=0)
transport = st.number_input("Transport Spending", min_value=0)
entertainment = st.number_input("Entertainment Spending", min_value=0)

total_spending = food + transport + entertainment

# --- TRAIN SIMPLE ML MODEL ---
data = {
    'income': [1000, 1000, 1000, 1500, 1500, 1500, 2000, 2000, 2000],
    'total_spending': [700, 900, 1100, 1000, 1300, 1600, 1200, 1800, 2200],
    'label': ['Healthy', 'Tight', 'Overspending',
              'Healthy', 'Tight', 'Overspending',
              'Healthy', 'Tight', 'Overspending']
}

df = pd.DataFrame(data)
X = df[['income', 'total_spending']]
y = df['label']
model = LogisticRegression()
model.fit(X, y)

# --- ML PREDICTION ---
budget_type = model.predict(np.array([[income, total_spending]]))[0]

# --- AI EXPLANATION ---
def generate_ai_explanation(income, total_spending, budget_type):
    prompt = f"My income is {income}, my total spending is {total_spending}. My budget type is {budget_type}. Explain this simply for a student."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

if st.button("Analyze Budget"):
    explanation = generate_ai_explanation(income, total_spending, budget_type)
    st.write(f"**Budget Type:** {budget_type}")
    st.write("**AI Explanation:**")
    st.write(explanation)
