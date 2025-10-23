import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# loading the saved models
diabetes_model = pickle.load(open('save models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('save models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('save models/parkinsons_model.sav', 'rb'))

breast_model = pickle.load(open("save models/breast_cancer_model.sav",'rb'))

lung_cancer = pickle.load(open("save models/lung_cancer_model.sav",'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Disease Information',
                            'Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Breast Cancer Prediction',
                            'Lung Cancer Prediction',],
                           menu_icon='hospital-fill',
                           icons=['circle','activity', 'heart', 'person','gender-female'],
                           default_index=0)

if selected=='Disease Information':
        st.markdown(
            """
            <style>
                body {
                    font-family: Cambria;
                    height: 100%;
                    background-image: linear-gradient(#4e0374, #c37ee6);
                    margin: 0;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }
                h1 {
                    text-align: center;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<h1>Disease Information</h1>", unsafe_allow_html=True)
        
        st.markdown("<h3>Information about the Diseases:</h3>", unsafe_allow_html=True)

        diseases = [
            {
                "name": "Diabetes",
                "description": "Diabetes mellitus refers to a group of diseases that affect how your body uses blood sugar (glucose). Glucose is vital to your health because it's an important source of energy for the cells that make up your muscles and tissues. It's also your brain's main source of fuel. The underlying cause of diabetes varies by type. But, no matter what type of diabetes you have, it can lead to excess sugar in your blood. Too much sugar in your blood can lead to serious health problems.",
                "symptoms": ["Increased thirst", "Frequent urination", "Extreme hunger", "Unexplained weight loss", "Blurred vision"]
            },
            {
                "name": "Heart/ Cardiovascular Disease",
                "description": "Cardiovascular disease (heart disease) refers to a group of diseases that affect the heart and blood vessels of your body. These diseases can affect one or many parts of your heart and /or blood vessels. A person may be symptomatic (physically experience the disease) or be asymptomatic (not feel anything at all).",
                "symptoms": ["Pounding or racing heart (palpitations)", "Chest pain", "Sweating", "Lightheadedness", "Shortness of breath"]
            },
            {
                "name": "Parkinsons Disease",
                "description": "Parkinsons disease is a brain disorder that causes unintended or uncontrollable movements, such as shaking, stiffness, and difficulty with balance and coordination.",
                "symptoms": ["Tremor in hands, arms, legs, jaw, or head","Muscle stiffness, where muscle remains contracted for a long time","Slowness of movement","Impaired balance and coordination, sometimes leading to falls"]
            },
            {
                "name": "Breast Cancer",
                "description": "Breast cancer is a type of cancer that starts in the breast. It can start in one or both breasts. After skin cancer, breast cancer is the most common cancer diagnosed in women in the United States. Breast cancer can occur in both men and women, but it's far more common in women.",
                "symptoms": ["New lump in the breast or underarm (armpit)", "Thickening or swelling of part of the breast.", "Irritation or dimpling of breast skin", "Redness or flaky skin in the nipple area or the breast", "Pulling in of the nipple or pain in the nipple area"]
            },
            {
                "name": "Lung Cancer",
                "description": "Lung cancer is a type of cancer that starts when abnormal cells grow in an uncontrolled way in the lungs. It is a serious health issue that can cause severe harm and death.",
                "symptoms": ["cough that does not go away","chest pain","shortness of breath","coughing up blood (haemoptysis)","fatigue","weight loss"]
            }
        ]

        for disease in diseases:
            st.markdown(f"<h4>{disease['name']}</h4>", unsafe_allow_html=True)
            st.markdown(f"<p>{disease['description']}</p>", unsafe_allow_html=True)
            st.markdown("<h5>Symptoms</h5>", unsafe_allow_html=True)
            st.markdown("<ul>", unsafe_allow_html=True)
            for symptom in disease["symptoms"]:
                st.markdown(f"<li>{symptom}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
            

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies',min_value=0)
        SkinThickness = st.number_input('Skin Thickness value',min_value=0.00)
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value',min_value=0.00)

    with col2:
        Glucose = st.number_input('Glucose Level',min_value=0.00)
        Insulin = st.number_input('Insulin Level',min_value=0.00)
        Age = st.number_input('Age',min_value=0)

    with col3:
        BloodPressure = st.number_input('Blood Pressure value',min_value=0.00)
        BMI = st.number_input('BMI value',min_value=0.00)

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age',min_value=0)
        trestbps = st.number_input('Resting Blood Pressure',min_value=0.00)
        restecg = st.number_input('Resting Electrocardiographic results',min_value=0.00)
        oldpeak = st.number_input('ST depression induced by exercise',min_value=0.00)
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',min_value=0,max_value=2)

    with col2:
        sex = st.number_input("GENDER: 0 = male and 1 = female",min_value=0,max_value=1)
        chol = st.number_input('Serum Cholestoral in mg/dl',min_value=0.00)
        thalach = st.number_input('Maximum Heart Rate achieved',min_value=0.00)
        slope = st.number_input('Slope of the peak exercise ST segment',min_value=0.00)

    with col3:
        cp = st.number_input('Chest Pain types',min_value=0.00)
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl',min_value=120.00)
        exang = st.number_input('Exercise Induced Angina',min_value=0.00)
        ca = st.number_input('Major vessels colored by flourosopy',min_value=0.00)

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)',min_value=0.00)
        RAP = st.number_input('MDVP:RAP',min_value=0.00)
        APQ3 = st.number_input('Shimmer:APQ3',min_value=0.00)
        HNR = st.number_input('HNR',min_value=0.00)
        D2 = st.number_input('D2',min_value=0.00)

    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)',min_value=0.00)
        PPQ = st.number_input('MDVP:PPQ',min_value=0.00)
        APQ5 = st.number_input('Shimmer:APQ5',min_value=0.00)
        RPDE = st.number_input('RPDE',min_value=0.00)
        PPE = st.number_input('PPE',min_value=0.00)

    with col3:
        flo = st.number_input('MDVP:Flo(Hz)',min_value=0.00)
        DDP = st.number_input('Jitter:DDP',min_value=0.00)
        APQ = st.number_input('MDVP:APQ',min_value=0.00)
        DFA = st.number_input('DFA',min_value=0.00)

    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)',min_value=0.00)
        Shimmer = st.number_input('MDVP:Shimmer',min_value=0.00)
        DDA = st.number_input('Shimmer:DDA',min_value=0.00)
        spread1 = st.number_input('spread1',min_value=0.00)

    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)',min_value=0.00)
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)',min_value=0.00)
        NHR = st.number_input('NHR',min_value=0.00)
        spread2 = st.number_input('spread2',min_value=0.00)

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    # Page title
    st.title('Breast Cancer Prediction using ML')

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mean_radius = st.number_input('Mean Radius',min_value=0.00)
        mean_smoothness = st.number_input('Mean Smoothness',min_value=0.00)
        mean_symmetry = st.number_input('Mean Symmetry',min_value=0.00)
        perimeter_error = st.number_input('Perimeter Error',min_value=0.00)

    with col2:
        mean_texture = st.number_input('Mean Texture',min_value=0.00)
        mean_compactness = st.number_input('Mean Compactness',min_value=0.00)
        mean_fractal_dimension = st.number_input('Mean Fractal Dimension',min_value=0.00)      
        area_error = st.number_input('Area Error',min_value=0.00)

    with col3:
        mean_perimeter = st.number_input('Mean Perimeter',min_value=0.00)
        mean_concavity = st.number_input('Mean Concavity',min_value=0.00)        
        radius_error = st.number_input('Radius Error',min_value=0.00)    
        smoothness_error = st.number_input('Smoothness Error',min_value=0.00)

    with col4:
        mean_area = st.number_input('Mean Area',min_value=0.00)
        mean_concave_points = st.number_input('Mean Concave Points',min_value=0.00)
        texture_error = st.number_input('Texture Error',min_value=0.00)
        compactness_error = st.number_input('Compactness Error',min_value=0.00)


    with col1:
        concavity_error = st.number_input('Concavity Error',min_value=0.00)
        worst_radius = st.number_input('Worst Radius',min_value=0.00)
        worst_smoothness = st.number_input('Worst Smoothness',min_value=0.00)
        worst_symmetry = st.number_input('Worst Symmetry',min_value=0.00)

    with col2:        
        concave_points_error = st.number_input('Concave Points Error',min_value=0.00)
        worst_texture = st.number_input('Worst Texture',min_value=0.00)
        worst_compactness = st.number_input('Worst Compactness',min_value=0.00)
        worst_fractal_dimension = st.number_input('Worst Fractal Dimension',min_value=0.00)

    with col3:
        symmetry_error = st.number_input('Symmetry Error',min_value=0.00)
        worst_perimeter = st.number_input('Worst Perimeter',min_value=0.00)
        worst_concavity = st.number_input('Worst Concavity',min_value=0.00)
        worst_concave_points_error = st.number_input('Worst Concave Points error',min_value=0.00)


    with col4:
        fractal_dimension_error = st.number_input('Fractal Dimension Error',min_value=0.00)
        worst_area = st.number_input('Worst Area',min_value=0.00)
        worst_concave_points = st.number_input('Worst Concave Points',min_value=0.00)

        
    # Code for prediction
    cancer_diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Breast Cancer Test Result'):
        if not all([mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness,
                    mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error,
                    texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error,
                    concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture,
                    worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity,
                    worst_concave_points, worst_symmetry, worst_fractal_dimension, worst_concave_points_error]):
            st.warning("Please fill in all the fields.")
        else:
            cancer_prediction = breast_model.predict([[mean_radius, mean_texture, mean_perimeter, mean_area,
                                                       mean_smoothness, mean_compactness, mean_concavity,
                                                       mean_concave_points, mean_symmetry, mean_fractal_dimension,
                                                       radius_error, texture_error, perimeter_error, area_error,
                                                       smoothness_error, compactness_error, concavity_error,
                                                       concave_points_error, symmetry_error, fractal_dimension_error,
                                                       worst_radius, worst_texture, worst_perimeter, worst_area,
                                                       worst_smoothness, worst_compactness, worst_concavity,
                                                       worst_concave_points, worst_symmetry, worst_fractal_dimension, worst_concave_points_error]])
            
            if cancer_prediction[0] == 1:
                cancer_diagnosis = 'The person is diagnosed with breast cancer.'
            else:
                cancer_diagnosis = 'The person is not diagnosed with breast cancer.'
        
    st.success(cancer_diagnosis)

# Lung Cancer Prediction Page:

if(selected == "Lung Cancer Prediction"):
    
    # page title
    st.title("Lung Cancer Prediction using ML")

    # getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        GENDER = st.number_input("GENDER: 0 = male and 1 = female",min_value=0,max_value=1)
        ANXIETY = st.number_input("ANXIETY 1=No 2=Yes",min_value=1,max_value=2)
        ALLERGY = st.number_input("ALLERGY 1=No 2=Yes",min_value=1,max_value=2)
        SHORTNESS_OF_BREATH = st.number_input("SHORTNESS OF BREATH 1=No 2=Yes",min_value=1,max_value=2)
        
    with col2:
        AGE = st.number_input("AGE",min_value=0)
        PEER_PRESSURE = st.number_input("PEER_PRESSURE 1=No 2=Yes",min_value=1,max_value=2)
        WHEEZING = st.number_input("WHEEZING 1=No 2=Yes",min_value=1,max_value=2)
        SWALLOWING_DIFFICULTY = st.number_input("SWALLOWING DIFFICULTY 1=No 2=Yes",min_value=1,max_value=2)
    
    with col3:
        SMOKING = st.number_input("SMOKING 1=No 2=Yes",min_value=1,max_value=2)
        CHRONIC_DISEASE = st.number_input("CHRONIC DISEASE 1=No 2=Yes",min_value=1,max_value=2)
        ALCOHOL_CONSUMING = st.number_input("ALCOHOL CONSUMING 1=No 2=Yes",min_value=1,max_value=2)
        CHEST_PAIN = st.number_input("CHEST PAIN 1=No 2=Yes",min_value=1,max_value=2)
    
    with col4:
        YELLOW_FINGERS = st.number_input("YELLOW_FINGERS 1=No 2=Yes",min_value=1,max_value=2)
        FATIGUE = st.number_input("FATIGUE 1=No 2=Yes",min_value=1,max_value=2)
        COUGHING = st.number_input("COUGHING 1=No 2=Yes",min_value=1,max_value=2)        
    
    # code for Prediction
    lung_cancer_result = ''
    
    # creating a button for Prediction
    
    if st.button("Lung Cancer Test Result"):
        user_input=[GENDER, AGE, SMOKING, YELLOW_FINGERS, 
            ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE,
              FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, 
              COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]
        
        user_input=[float(x) for x in user_input]
        lung_cancer_report = lung_cancer.predict([user_input])
        
        if lung_cancer_report[0] == 0:
            lung_cancer_result = "Hurrah! You have no Lung Cancer."
        else:
            lung_cancer_result = "Sorry! You have Lung Cancer."
        
    st.success(lung_cancer_result)   
  

