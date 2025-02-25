import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
from streamlit_option_menu import option_menu
# set page configuration
st.set_page_config(page_title="Prediction od Disease Outbreaks",
                   layout='wide',
                   page_icon='doctor')
# getting the working directory of the main,py
working_dir = os.path.dirname(os.path.abspath(__file__))
# loading the saved models
'''diabetes_model = pickle.load(open(f'{working_dir}/Training_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/Training_models/heart_model.sav', 'rb'))
parkinsons_model= pickle.load(open(f'{working_dir}/Training_models/parkinsons_model.sav', 'rb'))'''


# Load the saved models for Diabetes XGBoost model using Pickle
with open('Training_models/xgboost_diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler used during training
with open('Training_models/X_train_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the trained model for Heart disease
model_filename = "Training_models/heart_disease_model.pkl"
with open(model_filename, "rb") as model_file:
    modelh = pickle.load(model_file)

# Load the saved models for Parkinsons
parkinsons_model= pickle.load(open(f'{working_dir}/Training_models/parkinsons_model.sav', 'rb'))

# sidebar for nevigation
with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreaks System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                            menu_icon='hospital-fill',
                            icons=['activity', 'heart', 'person'],
                            default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    '''# page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucoss = st.text_input('Glucoss Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')

    with col1:
        skinthikness = st.text_input('Skin Thikness Value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI Value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value') 
        
    with col2:
        Age = st.text_input('Age of the Person')

    # code for prediction
    diab_diagnosis = ''

    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        user_input = [Pregnancies, Glucoss, BloodPressure, skinthikness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        user_input = [float(x) for x in user_input]

        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        user_input_df = pd.DataFrame([user_input], columns=feature_names)

        # Make prediction
        diab_diagnosis = diabetes_model.predict(user_input_df)

        #diab_diagnosis = diabetes_model.predict([user_input])
        if diab_diagnosis[0] == 1:
            diab_diagnosis = 'The person is not diabetic'
        else:
            diab_diagnosis =  'The person is diabetic'

    st.success(diab_diagnosis)'''


    st.title("Diabetes Prediction App")
    st.markdown("""
    This app predicts the likelihood of diabetes based on user input data.
    Fill in the values below and click 'Predict' to see the result.
    """)

    # Input fields for user data
    Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
    Glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
    BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
    SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    Insulin = st.number_input('Insulin', min_value=0, max_value=800, value=85)
    BMI = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    Age = st.number_input('Age', min_value=0, max_value=120, value=30)

    # Collect user inputs into a DataFrame with feature names
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    user_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, 
                            Insulin, BMI, DiabetesPedigreeFunction, Age]], 
                            columns=feature_names)

    # Standardize user inputs
    user_data_scaled = scaler.transform(user_data)


    # Make prediction when the button is clicked
    if st.button('Predict'):
        prediction = model.predict(user_data_scaled)
        probability = model.predict_proba(user_data_scaled)[0][1]
        
        if prediction[0] == 1:
            st.error(f'High risk of diabetes. Probability: {probability:.2f}')
        else:
            st.success(f'Low risk of diabetes. Probability: {probability:.2f}')






# Herat Disease Prediction Page
if selected == 'Heart Disease Prediction':

    '''# page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex= st.text_input('sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        restbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholertoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        maxhra = st.text_input('Maximum Heart Rate Achived')

    with col3:
        exang = st.text_input('Exercise Induced angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak Exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating button for prediction
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, restbps, chol, fbs, restecg, maxhra, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        heart_Prediction = heart_disease_model.predict([user_input])

        if heart_Prediction[0] == 1:
            heart_diagnosis = 'The Person is having Heart Disease'
        else:
            heart_diagnosis = 'The Person does not have any Heart Disease'
        
    st.success(heart_diagnosis)'''
    st.title("Heart Disease Prediction App")
    st.write("Enter the details below to predict the risk of heart disease.")

    # Create input fields
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
    restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise-Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    # Convert categorical inputs
    sex = 1 if sex == "Male" else 0

    # Create input data for prediction
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                            columns=["age", "sex", "cp", "trestbps", "chol", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

    # Make prediction
    if st.button("Predict"):
        prediction = modelh.predict(input_data)
        result = "Heart Disease Detected" if prediction[0] == 0 else "No Heart Disease"
        
        st.subheader("Prediction Result:")
        st.write(f"🩺 **{result}**")

# Parkinsons Prediction Page
if selected == 'Parkinsons Prediction':
    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('MDVP:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    
    with col3:
        APQ = st.text_input('MDVP:APQ')
    
    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    Parkinsons_diagnosis = ''

    # creaing a button for prediction
    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_abs, RAP, PPQ, DDP, Shimmer,
                      Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        user_input = [float(x) for x in user_input]
        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            Parkinsons_diagnosis = "The Person have Parkinson's Disease"
        else:
            Parkinsons_diagnosis = "The Person does not have Parkinson's Disease"
    st.success(Parkinsons_diagnosis)




    

    


