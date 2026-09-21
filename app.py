from flask import Flask, request, render_template
import joblib
import pandas as pd



app = Flask(__name__)




preprocessor = joblib.load("model/preprocessor.pkl")
model = joblib.load("model/xgboost_churn.pkl")
threshold = joblib.load("model/threshold.pkl")



@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    probability = None

    if request.method == "POST":

        
        tenure = float(request.form["tenure"])
        monthly_charges = float(request.form["MonthlyCharges"])
        total_charges = float(request.form["TotalCharges"])

        

        senior_citizen = int(request.form["SeniorCitizen"])

    

        gender = request.form["gender"]
        partner = request.form["Partner"]
        dependents = request.form["Dependents"]
        phone_service = request.form["PhoneService"]
        multiple_lines = request.form["MultipleLines"]
        internet_service = request.form["InternetService"]
        online_security = request.form["OnlineSecurity"]
        online_backup = request.form["OnlineBackup"]
        device_protection = request.form["DeviceProtection"]
        tech_support = request.form["TechSupport"]
        streaming_tv = request.form["StreamingTV"]
        streaming_movies = request.form["StreamingMovies"]
        contract = request.form["Contract"]
        paperless_billing = request.form["PaperlessBilling"]
        payment_method = request.form["PaymentMethod"]


       

        data = {
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],

            "SeniorCitizen": [senior_citizen],

            "gender": [gender],
            "Partner": [partner],
            "Dependents": [dependents],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method]
        }

        X_new = pd.DataFrame(data)


        
        X_new_encoded = preprocessor.transform(X_new)


        

        probability = model.predict_proba(X_new_encoded)[0, 1]


        
        if probability >= threshold:
            prediction = "Client susceptible de partir"
        else:
            prediction = "Client susceptible de rester"


    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        threshold=threshold
    )


if __name__ == "__main__":
    app.run(debug=True)