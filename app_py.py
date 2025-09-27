import streamlit as st
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# Load the trained model
model = tf.keras.models.load_model('model_weights.h5')

df = pd.read_csv(r"C:\Users\Arjun\Downloads\full_data.csv")

X = pd.DataFrame(df.iloc[:, 0:10].values)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_2 = LabelEncoder()
X.loc[:,0] = labelencoder_X_2.fit_transform(X.iloc[:, 0])
X.loc[:,4] = labelencoder_X_2.fit_transform(X.iloc[:, 4])
X.loc[:,5] = labelencoder_X_2.fit_transform(X.iloc[:, 5])
X.loc[:,6] = labelencoder_X_2.fit_transform(X.iloc[:, 6])
X.loc[:,9] = labelencoder_X_2.fit_transform(X.iloc[:, 9])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)

# Define a function to preprocess the input data
def preprocess_input(data):
    # Convert categorical columns to integer encoding
    data['work_type'] = data['work_type'].map({'Private': 1, 'Self-employed': 2, 'Govt_job': 0, 'children': 3})
    data['smoking_status'] = data['smoking_status'].map({'never smoked': 2, 'formerly smoked': 1, 'smokes': 3, 'Unknown': 0})
    # Drop the id column
    # Normalize the numerical columns
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    #data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()
    # Convert the preprocessed data to a numpy array
    data = data.to_numpy()
    return data

# Define the input fields
st.header("Brain Stroke Prediction")
st.subheader("Please enter the patient details:")

gender = st.selectbox("Gender   ", [0, 1])
age = st.number_input("Age", min_value=0.0, max_value=120.0, value=25.0, step=1.0, format="%.2f")
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", [0, 1])
work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children'])
Residence_type = st.selectbox("Residence Type", [0, 1])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=55.0, max_value=335.0, value=100.0, step=1.0, format="%.2f")
bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=20.0, step=0.1,format="%.2f")
smoking_status = st.selectbox("Smoking Status", ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

# Create a pandas dataframe from the input values
input_df = pd.DataFrame({
    'gender' : [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [Residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

# Preprocess the input data and make a prediction
if st.button("Predict"):
    #print(input_df)
    input_data = preprocess_input(input_df)
    input_data = sc.transform(input_data)
    #input_data_f[0][1] = float(input_data[0][1])
    #input_data_f[0][8] = float(input_data[0][8])
    #input_data_f[0][7] = float(input_data[0][7])
    print(input_data)
    prediction = model.predict(input_data)

    print(prediction)
    if prediction > 0.1:
        st.error("This patient has a high risk of stroke!")
        
    else:
        st.success("This patient has a low risk of stroke.")
        

