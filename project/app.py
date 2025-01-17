from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('booking_prediction_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs from the form
    num_passengers = int(request.form['num_passengers'])
    sales_channel = request.form['sales_channel']
    trip_type = request.form['trip_type']
    purchase_lead = int(request.form['purchase_lead'])
    flight_hour = int(request.form['flight_hour'])
    flight_day = request.form['flight_day']
    flight_duration = float(request.form['flight_duration'])

    # Prepare the data for prediction
    user_data = {
        "num_passengers": [num_passengers],
        "sales_channel": [sales_channel],
        "trip_type": [trip_type],
        "purchase_lead": [purchase_lead],
        "flight_hour": [flight_hour],
        "flight_day": [flight_day],
        "flight_duration": [flight_duration]
    }
    user_df = pd.DataFrame(user_data)

    # Encode the input to match training data
    trained_columns = model.feature_names_in_  # Use the column names saved in the model
    user_df_encoded = pd.get_dummies(user_df)

    # Reindex to match the training data columns
    user_df_encoded = user_df_encoded.reindex(columns=trained_columns, fill_value=0)

    # Make the prediction
    prediction = model.predict(user_df_encoded)
    prediction_prob = model.predict_proba(user_df_encoded)[:, 1]

    result = {
        'prediction': "Booking will be completed." if prediction[0] == 1 else "Booking will not be completed.",
        'confidence': f"{prediction_prob[0]*100:.2f}%"
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
