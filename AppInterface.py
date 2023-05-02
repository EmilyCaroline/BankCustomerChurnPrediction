import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the saved model
with open('gb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the saved feature scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the saved list of features
with open('features.txt', 'r') as f:
    features = f.read().splitlines()

# Define a function to make predictions on new data
def predict_churn(data):
    data = scaler.transform(data)
    predictions = model.predict(data)
    churn_probabilities = model.predict_proba(data)[:,1]
    return predictions, churn_probabilities

# Define the input interface using Streamlit widgets
st.title('Bank Customer Churn Prediction')
st.write('Please enter or select the following customer details to predict if they will churn or not:')

creditscore = st.slider('Credit Score', min_value=300, max_value=850, step=1)
age = st.slider('Age', min_value=18, max_value=100, step=1)
tenure = st.slider('Tenure', min_value=0, max_value=10, step=1)
balance = st.number_input('Balance')
numofproducts = st.slider('Number of Products', min_value=1, max_value=4, step=1)
hascrcard = st.selectbox('Has Credit Card?', ['Yes', 'No'])
if hascrcard == 'Yes':
    hascrcard = 1.0
else:
    hascrcard = 0.0
isactivemember = st.selectbox('Is Active Member?', ['Yes', 'No'])
if isactivemember == 'Yes':
    isactivemember = 1.0
else:
    isactivemember = 0.0
estimatedsalary = st.number_input('Estimated Salary')
education = st.slider('Education', min_value=1, max_value=5, step=1)
investment = st.number_input('Investment')
activity = st.number_input('Activity')
yearlyamt = st.number_input('Yearly Amount')
avgdailytax = st.number_input('Average Daily Tax')
yearlytax = st.number_input('Yearly Tax')
avgtaxamt = st.number_input('Average Tax Amount')
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany', 'United States'])
gender = st.selectbox('Gender', ['Male', 'Female'])
educationgroup = st.selectbox('Education Group', ['High school graduate', 'Bachelors degree', 'Doctorate', 'Masters degree'])

# Create a DataFrame with the new data (one row)
new_data = pd.DataFrame({
    'CreditScore': [creditscore],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [numofproducts],
    'HasCrCard': [hascrcard],
    'IsActiveMember': [isactivemember],
    'EstimatedSalary': [estimatedsalary],
    'Education': [education],
    'Investment': [investment],
    'Activity': [activity],
    'Yearly Amt': [yearlyamt],
    'Avg Daily Tax': [avgdailytax],
    'Yearly Tax': [yearlytax],
    'Avg Tax Amt': [avgtaxamt],
    'Geography': [geography],
    'Gender': [gender],
    'Education Group': [educationgroup]
})

for feature in ['Geography', 'Gender', 'Education Group']:
    le = LabelEncoder()
    le.fit(new_data[feature])
    new_data[feature + '_dummy'] = le.transform(new_data[feature])

# Make predictions on the new data
predictions, churn_probabilities = predict_churn(new_data[features])


# Display the predicted churn probability and label
st.write(f"The predicted churn probability is {churn_probabilities[0]:.2%}.")
st.write(f"The predicted churn label is {predictions[0]}.")

if predictions == 0:
    st.write('The customer is predicted to NOT churn or NOT be at risk of churning.')
else:
    st.write('The customer is predicted to churn or be at risk of churning.')