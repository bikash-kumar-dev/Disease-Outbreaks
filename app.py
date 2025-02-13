import os
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
# set page configuration
st.set_page_config(page_title="Prediction od Disease Outbreaks",
                   layout='wide',
                   page_icon='doctor')
# getting the working directory of the main,py
working_dir = os.path.dirname(os.path.abspath(__file__))
# loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/Training_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/Training_models/heart_model.sav', 'rb'))
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

    # page title
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

    st.success(diab_diagnosis)

# Herat Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
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
        
    st.success(heart_diagnosis)

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




    

    


