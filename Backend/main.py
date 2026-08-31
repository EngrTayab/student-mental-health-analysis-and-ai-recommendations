from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Annotated
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

import joblib

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load The Model
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")


class InputFields(BaseModel):
    Age: Annotated[int, "Age", Field(gt=9, lt=31)]
    Gender: Literal["Male", "Female"]
    Country: Literal[
        "Afghanistan",
        "Albania",
        "Andorra",
        "Argentina",
        "Armenia",
        "Australia",
        "Austria",
        "Azerbaijan",
        "Bahamas",
        "Bahrain",
        "Bangladesh",
        "Belarus",
        "Belgium",
        "Bhutan",
        "Bolivia",
        "Bosnia",
        "Brazil",
        "Bulgaria",
        "Canada",
        "Chile",
        "China",
        "Colombia",
        "Costa Rica",
        "Croatia",
        "Cyprus",
        "Czech Republic",
        "Denmark",
        "Ecuador",
        "Egypt",
        "Estonia",
        "Finland",
        "France",
        "Georgia",
        "Germany",
        "Ghana",
        "Greece",
        "Hong Kong",
        "Hungary",
        "Iceland",
        "India",
        "Indonesia",
        "Iraq",
        "Ireland",
        "Israel",
        "Italy",
        "Jamaica",
        "Japan",
        "Jordan",
        "Kazakhstan",
        "Kenya",
        "Kosovo",
        "Kuwait",
        "Kyrgyzstan",
        "Latvia",
        "Lebanon",
        "Liechtenstein",
        "Lithuania",
        "Luxembourg",
        "Malaysia",
        "Maldives",
        "Malta",
        "Mexico",
        "Moldova",
        "Monaco",
        "Montenegro",
        "Morocco",
        "Nepal",
        "Netherlands",
        "New Zealand",
        "Nigeria",
        "North Macedonia",
        "Norway",
        "Oman",
        "Other",
        "Pakistan",
        "Panama",
        "Paraguay",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Qatar",
        "Romania",
        "Russia",
        "San Marino",
        "Serbia",
        "Singapore",
        "Slovakia",
        "Slovenia",
        "South Africa",
        "South Korea",
        "Spain",
        "Sri Lanka",
        "Sweden",
        "Switzerland",
        "Syria",
        "Taiwan",
        "Tajikistan",
        "Thailand",
        "Trinidad",
        "Turkey",
        "UAE",
        "UK",
        "USA",
        "Ukraine",
        "Uruguay",
        "Uzbekistan",
        "Vatican City",
        "Venezuela",
        "Vietnam",
        "Yemen",
    ]
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"]
    Most_Used_Platform: Literal[
        "Facebook",
        "Instagram",
        "KakaoTalk",
        "LINE",
        "LinkedIn",
        "Snapchat",
        "TikTok",
        "Twitter",
        "VKontakte",
        "WeChat",
        "WhatsApp",
        "YouTube",
    ]
    Purpose_Of_Use: Literal["Education", "Entertainment", "Networking", "News"]
    Avg_Daily_Usage_Hours: Annotated[
        float, "Daily Usage of Social Media", Field(gt=0, lt=14)
    ]
    Study_Hours: Annotated[float, "Study Hours/Day", Field(gt=0, lt=16)]
    Physical_Activity_Hours: Annotated[float, "Activities", Field(lt=14)]
    Sleep_Hours_Per_Night: Annotated[float, "Sleep hours", Field(gt=0, lt=14)]
    Mental_Health_Score: Annotated[float, "Health Score", Field(gt=0, lt=14)]


@app.get("/")
def home():
    return "Here You have to Enter Input fields coreectly and trained ML model will predict Student Mental Health"


@app.post("/predict")
def predict(data: InputFields):
    input = pd.DataFrame(
        [
            {
                "Age": data.Age,
                "Gender": data.Gender,
                "Country": data.Country,
                "Academic_Level": data.Academic_Level,
                "Most_Used_Platform": data.Most_Used_Platform,
                "Purpose_Of_Use": data.Purpose_Of_Use,
                "Avg_Daily_Usage_Hours": data.Avg_Daily_Usage_Hours,
                "Study_Hours": data.Study_Hours,
                "Sleep_Hours_Per_Night": data.Sleep_Hours_Per_Night,
                "Physical_Activity_Hours": data.Physical_Activity_Hours,
                "Sleep_Hours_Per_Night": data.Sleep_Hours_Per_Night,
                "Mental_Health_Score": data.Mental_Health_Score,
            }
        ]
    )
    # Debugging: Print expected feature names and input columns
    print("Expected feature names:", scaler.feature_names_in_)
    print("Input DataFrame columns:", input.columns)

    # Ensure the column order matches the training data
    input = input[scaler.feature_names_in_]

    for col, encoder in encoders.items():
        if col in input.columns:
            input[col] = encoder.transform(input[col])

    input_data_scaled = scaler.transform(input)

    prediction = model.predict(input_data_scaled)
    
    probability = model.predict_proba(input_data_scaled)
    predicted_class_probability=max(probability[0])
    

    recommendations = []

    if prediction == 0:
        result = "Low Stress"
        recommendations.append(
            "Keep up the good work! Continue maintaining a healthy routine by  ."
        )
    elif prediction == 1:
        result = "Medium Stress"
        recommendations.append(
            "Maintain a balance between study and leisure activities."
        )
        recommendations.append("Practice mindfulness or meditation to manage stress.")
    elif prediction == 2:
        result = "High Strees"
        recommendations.append("Consider reducing daily social media usage.")
        recommendations.append("Engage in regular physical activity to reduce stress.")
        recommendations.append("Ensure you get at least 7-8 hours of sleep per night.")
    else:
        result = "Very High"
        recommendations.append("Engage in regular physical activity to reduce stress.")
        recommendations.append("Ensure you get at least 7-8 hours of sleep per night.")
        
    if data.Avg_Daily_Usage_Hours > 6 and data.Mental_Health_Score < 5:
        recommendations.append("Reduce screen time to avoid mental fatigue.")
        recommendations.append("Take regular breaks from social media to relax your mind.")
        recommendations.append("Seek support from friends, family, or a mental health professional.")
        recommendations.append("Engage in activities that promote relaxation and well-being.")

    if data.Study_Hours < 2 and data.Physical_Activity_Hours < 1:
        recommendations.append("Increase study hours to improve academic performance.")
        recommendations.append("Create a structured study schedule to manage time effectively.")
        recommendations.append("Seek academic support or tutoring if needed.")
        recommendations.append("Engage in activities like walking, jogging, or yoga to reduce stress.")
        recommendations.append("Consider joining a sports club or fitness class for motivation.")



    if data.Sleep_Hours_Per_Night < 6 and data.Mental_Health_Score < 5:
        
        recommendations.append("Prioritize sleep to improve mental health.")
        recommendations.append("Establish a consistent sleep schedule.")
        recommendations.append("Avoid caffeine and electronic devices before bedtime.")

    if data.Mental_Health_Score >= 8:
        recommendations.append("Keep up the good work! Continue maintaining a healthy routine.")
        recommendations.append("Engage in activities that promote mental well-being.")

        recommendations.append("Consider sharing your positive habits with others to inspire them.")
    
    if data.Age >= 18:
        recommendations.append("Maintain a healthy work-life balance to manage stress effectively.")
        recommendations.append("Engage in activities that promote personal growth and development.")
       
    if data.Academic_Level == "High School":
        recommendations.append("Focus on building a strong foundation for future academic pursuits.")
        recommendations.append("Seek guidance from teachers and counselors for academic support.")

    if data.Academic_Level == "Undergraduate":
        recommendations.append("Focus on academic performance and skill development.")

    if data.Academic_Level == "Graduate":
        recommendations.append("Focus on advanced academic and research pursuits.")
        recommendations.append("Seek mentorship and guidance from faculty and professionals.")

        
    if data.Most_Used_Platform in ["Facebook", "Instagram", "Snapchat", "TikTok"]:
        recommendations.append("Limit time spent on social media to reduce stress and anxiety.")

    model_used = "KNN Classifier" 
    model_accuracy=0.86                    

    return {"prediction": result, "recommendation": recommendations , 'Probability':predicted_class_probability, 'Best_Model':model_used,"model_accuracy":model_accuracy, "input_data": data.dict(),  }
