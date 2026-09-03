# Student Mental Health Analysis & AI Recommendations

A machine learning based web application that analyzes student health-related information and provides mental health predictions along with AI-based recommendations.

## 🚀 Live Demo

**Frontend:**
https://YOUR-VERCEL-LINK.vercel.app

**Backend API:**
https://YOUR-RENDER-LINK.onrender.com

**API Documentation:**
https://YOUR-RENDER-LINK.onrender.com/docs

## ✨ Features

* Student mental health prediction using Machine Learning
* FastAPI REST API
* Data preprocessing using saved encoders and scaler
* AI-based recommendations
* Simple and responsive frontend
* Live frontend and backend deployment

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn
* Pydantic
* HTML
* CSS
* JavaScript
* Tailwind CSS

## 📁 Project Structure

```text
student-mental-health-analysis-and-ai-recommendations/
│
├── Backend/
│   ├── best_model.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   ├── main.py
│   └── model.ipynb
│
├── Frontend/
│   ├── index.html
│   ├── style.css
│   ├── output.css
│   └── tailwind.config.js
│
├── StudentHealthDataset.csv
├── about.html
├── package.json
├── package-lock.json
├── requirements.txt
└── README.md
```

## 🔄 How It Works

1. User enters the required information on the frontend.
2. Frontend sends the data to the FastAPI backend.
3. Backend processes the input using the saved preprocessing files.
4. The trained ML model generates a prediction.
5. The result and recommendations are displayed on the frontend.

## 🤖 Machine Learning

The application uses a trained Scikit-learn model with:

* `best_model.pkl` — trained model
* `scaler.pkl` — feature scaler
* `encoders.pkl` — categorical encoders

## 📊 Dataset

The model was developed using `StudentHealthDataset.csv`, containing student-related health and lifestyle information.

## ⚠️ Disclaimer

This project is developed for educational and informational purposes. The predictions and recommendations are not a medical diagnosis and should not replace professional mental health advice.

## 👨‍💻 Author

**EngrTayab**

GitHub: https://github.com/EngrTayab
