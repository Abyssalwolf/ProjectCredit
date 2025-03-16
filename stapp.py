import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Updated caching with new Streamlit commands
@st.cache_resource  # For model and scaler (resources)
def load_artifacts():
    model = load_model('creditnetxai_model.h5')
    scaler = joblib.load('standard_scaler.pkl')
    feature_cols = pd.read_csv('feature_columns.csv').iloc[:, 0].tolist()
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

X_train=pd.read_csv('X_train.csv')
X_test=pd.read_csv('X_test.csv')

@st.cache_data  # For data (X_train)
def load_shap_data():
    return pd.read_csv('X_train.csv')

X_train_shap = load_shap_data().iloc[:100]

def preprocess_input(input_data):
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    df = df[feature_cols]
    return scaler.transform(df)

# Metrics from training
ACCURACY = 0.8101666666666667
SENSITIVITY = 0.45925361766945927
SPECIFICITY = 0.9084702368252614

# Streamlit interface
st.title("Credit Default Prediction System")

# Metrics section
st.header("Model Performance")
st.metric("Accuracy", f"{ACCURACY:.2%}")
st.metric("Sensitivity", f"{SENSITIVITY:.2%}")
st.metric("Specificity", f"{SPECIFICITY:.2%}")

# SHAP visualization with fixes
st.header("Feature Importance Analysis")
with st.spinner("Calculating SHAP values..."):
    
    # Initialize explainer with preprocessed data
    explainer = shap.GradientExplainer(model, X_train[:100].values)
    shap_values = explainer.shap_values(X_test[:100].values)
    
    X_test_subset = X_test[:100]

    # Generate and display plot
    shap_values_fixed = shap_values[:, :, 0]

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values_fixed, X_test_subset)
    fig.savefig('shap_plot.png', bbox_inches='tight', dpi=300)
    st.pyplot(fig)
    plt.close(fig)  # Prevent duplicate display

from groq import Groq

st.header("Analyzing the plot")

image_path = "shap_plot.png"
base64_image = encode_image(image_path)
client = Groq(api_key="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")
completion = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Given the data metadata:\n{'uci_id': 350, 'name': 'Default of Credit Card Clients', 'repository_url': 'https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients', 'data_url': 'https://archive.ics.uci.edu/static/public/350/data.csv', 'abstract': \"This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods.\", 'area': 'Business', 'tasks': ['Classification'], 'characteristics': ['Multivariate'], 'num_instances': 30000, 'num_features': 23, 'feature_types': ['Integer', 'Real'], 'demographics': ['Sex', 'Education Level', 'Marital Status', 'Age'], 'target_col': ['Y'], 'index_col': ['ID'], 'has_missing_values': 'no', 'missing_values_symbol': None, 'year_of_dataset_creation': 2009, 'last_updated': 'Fri Mar 29 2024', 'dataset_doi': '10.24432/C55S3H', 'creators': ['I-Cheng Yeh'], 'intro_paper': {'ID': 365, 'type': 'NATIVE', 'title': 'The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients', 'authors': 'I. Yeh, Che-hui Lien', 'venue': 'Expert systems with applications', 'year': 2009, 'journal': None, 'DOI': '10.1016/j.eswa.2007.12.020', 'URL': 'https://www.semanticscholar.org/paper/1cacac4f0ea9fdff3cd88c151c94115a9fddcf33', 'sha': None, 'corpus': None, 'arxiv': None, 'mag': None, 'acl': None, 'pmid': None, 'pmcid': None}, 'additional_info': {'summary': \"This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods. From the perspective of risk management, the result of predictive accuracy of the estimated probability of default will be more valuable than the binary result of classification - credible or not credible clients. Because the real probability of default is unknown, this study presented the novel Sorting Smoothing Method to estimate the real probability of default. With the real probability of default as the response variable (Y), and the predictive probability of default as the independent variable (X), the simple linear regression result (Y = A + BX) shows that the forecasting model produced by artificial neural network has the highest coefficient of determination; its regression intercept (A) is close to zero, and regression coefficient (B) to one. Therefore, among the six data mining techniques, artificial neural network is the only one that can accurately estimate the real probability of default.\", 'purpose': None, 'funded_by': None, 'instances_represent': None, 'recommended_data_splits': None, 'sensitive_data': None, 'preprocessing_description': None, 'variable_info': 'This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:\\r\\nX1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.\\r\\nX2: Gender (1 = male; 2 = female).\\r\\nX3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).\\r\\nX4: Marital status (1 = married; 2 = single; 3 = others).\\r\\nX5: Age (year).\\r\\nX6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.\\r\\nX12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. \\r\\nX18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.\\r\\n', 'citation': None}}\n\nTake in the image of the figure plotted with shap and explain what you information you can derive from the image. Your explanation should not be larger than 4 sentences and needs to cover the feature that most impacts and how, the feature that least impacts and how and a conclusion"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
)

st.write(completion.choices[0].message.content)


# Prediction interface
st.header("Customer Default Prediction")
st.write("Please enter the customer's information:")

col1, col2 = st.columns(2)
with col1:
    limit_bal = st.number_input("Credit Limit (NT$)", min_value=1)
    sex = st.selectbox("Gender", ["Female", "Male"])
    education = st.selectbox("Education Level", 
                            ["Graduate School", "University", "High School", "Others"])
    marriage = st.selectbox("Marital Status", ["Married", "Single", "Others"])
    age = st.number_input("Age", min_value=18, max_value=100)

with col2:
    pay_0 = st.number_input("Payment Status - September (PAY_0)", min_value=-2, max_value=8)
    pay_2 = st.number_input("Payment Status - August (PAY_2)", min_value=-2, max_value=8)
    pay_3 = st.number_input("Payment Status - July (PAY_3)", min_value=-2, max_value=8)
    pay_4 = st.number_input("Payment Status - June (PAY_4)", min_value=-2, max_value=8)
    pay_5 = st.number_input("Payment Status - May (PAY_5)", min_value=-2, max_value=8)
    pay_6 = st.number_input("Payment Status - April (PAY_6)", min_value=-2, max_value=8)

col3, col4 = st.columns(2)
with col3:
    bill_amt1 = st.number_input("Bill Amount - September (NT$)", min_value=0)
    bill_amt2 = st.number_input("Bill Amount - August (NT$)", min_value=0)
    bill_amt3 = st.number_input("Bill Amount - July (NT$)", min_value=0)
    bill_amt4 = st.number_input("Bill Amount - June (NT$)", min_value=0)
    bill_amt5 = st.number_input("Bill Amount - May (NT$)", min_value=0)
    bill_amt6 = st.number_input("Bill Amount - April (NT$)", min_value=0)

with col4:
    pay_amt1 = st.number_input("Payment Amount - September (NT$)", min_value=0)
    pay_amt2 = st.number_input("Payment Amount - August (NT$)", min_value=0)
    pay_amt3 = st.number_input("Payment Amount - July (NT$)", min_value=0)
    pay_amt4 = st.number_input("Payment Amount - June (NT$)", min_value=0)
    pay_amt5 = st.number_input("Payment Amount - May (NT$)", min_value=0)
    pay_amt6 = st.number_input("Payment Amount - April (NT$)", min_value=0)

# Create input dictionary
input_data = {
    'LIMIT_BAL': limit_bal,
    'SEX': 2 if sex == "Male" else 1,
    'EDUCATION': {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}[education],
    'MARRIAGE': {"Married": 1, "Single": 2, "Others": 3}[marriage],
    'AGE': age,
    'PAY_0': pay_0,
    'PAY_2': pay_2,
    'PAY_3': pay_3,
    'PAY_4': pay_4,
    'PAY_5': pay_5,
    'PAY_6': pay_6,
    'BILL_AMT1': bill_amt1,
    'BILL_AMT2': bill_amt2,
    'BILL_AMT3': bill_amt3,
    'BILL_AMT4': bill_amt4,
    'BILL_AMT5': bill_amt5,
    'BILL_AMT6': bill_amt6,
    'PAY_AMT1': pay_amt1,
    'PAY_AMT2': pay_amt2,
    'PAY_AMT3': pay_amt3,
    'PAY_AMT4': pay_amt4,
    'PAY_AMT5': pay_amt5,
    'PAY_AMT6': pay_amt6
}

if st.button("Predict Default Risk"):
    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0][0]
    probability = prediction * 100
    
    st.subheader("Prediction Result:")
    if prediction > 0.4:
        st.error(f"🚨 High Risk: {probability:.1f}% probability of default")
    else:
        st.success(f"✅ Low Risk: {probability:.1f}% probability of default")