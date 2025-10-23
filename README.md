# 🧠 Multi-Disease Prediction Web App

A **Streamlit-based machine learning web app** for predicting multiple diseases, including **Diabetes, Breast Cancer, Lung Cancer, Heart Disease, and Parkinson’s**.

---

## 🔍 Features

- 🩺 **Multi-disease detection** — supports prediction for 5 major diseases.  
- 🤖 **Independent ML models** — each disease has a separate trained model for higher accuracy.  
- 🧩 **User-friendly interface** — select a disease, input medical details, and get instant results.  
- 📊 **Evaluation metrics** — models evaluated using accuracy, confusion matrix, and other key metrics.  
- ⚙️ **Data preprocessing & scaling** — applied to improve prediction reliability.  
- 💡 **Responsive UI** — built with Streamlit’s layout features for a clean and accessible experience.  

---

## 🧠 Tech Stack

**Frontend:** Streamlit  
**Backend / ML:** Python, Scikit-learn, NumPy, Pandas  
**Deployment:** Streamlit Cloud / Local  

---

## 🚀 How It Works

1. Choose a disease from the sidebar.  
2. Enter the required medical parameters in the input form.  
3. Click **Predict** to view the diagnosis result based on the trained model.  

---

## 📦 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/multi-disease-prediction.git

# Navigate to project directory
cd multi-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
