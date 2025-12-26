# ❤️ Health Risk Predictor

<div align="center">
  <img src="Screenshot 2025-12-26 183039.png" alt="Project Screenshot" width="600"/>
</div>

A **Machine Learning-based health prediction system** that helps users assess their health risks related to **heart disease and diabetes**. Users provide their health information, and the system predicts their risk level as a **percentage of being healthy or at risk**.

---

## 🚀 Features

- Predicts **heart disease risk** based on user input
- Predicts **diabetes risk** based on user input
- Displays **prediction percentages** for healthy vs risky
- Simple **HTML interface** for user input
- Built with **Python ML model** (scikit-learn / other libraries)

---

## 🛠️ Tech Stack

- **Python** – for ML model and backend logic  
- **scikit-learn** – for building predictive models  
- **HTML / CSS / JS** – for frontend interface  
- **Flask (optional)** – to connect frontend with Python backend  

---

## 💻 How It Works

1. User opens the web interface (`index.html`)  
2. Fills in health parameters (age, blood pressure, glucose level, etc.)  
3. Clicks **Predict**  
4. ML model processes the data and returns:  
   - ✅ Healthy (% probability)  
   - ⚠️ Risky (% probability)  

---

## 📈 Model Details

- **Data Preprocessing:** Handles missing values, scaling, and normalization  
- **Algorithm:** Logistic Regression / Random Forest / Decision Tree (choose your model)  
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1-Score  

---

---

## 📂 Project Structure

```text
HEALTH_PROJECT/
│
├── __pycache__/           # Python auto-generated cache files
│
├── .venv/                 # Virtual environment for dependencies
│
├── models/                # Trained machine learning models
│   └── health_model.pkl
│
├── outputs/               # Prediction outputs / logs
│
├── templates/             # HTML templates
│   └── index.html         # User interface for input & prediction
│
├── app.py                 # Flask application (connects frontend & ML model)
│
├── health_project2.py     # ML model training & prediction logic
│
└── README.md              # Project documentation



