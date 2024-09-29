from flask import Flask, request, jsonify, render_template
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load pretrained models and tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load data
try:
    weather_data = pd.read_csv('weather_data.csv')
    reviews_data = pd.read_csv('reviews_data.csv')
    print("Data loaded successfully.")
except Exception as e:
    print("Error loading data:", str(e))

# ARIMA Weather Model Training (for each city)
def train_weather_model(city, weather_data):
    try:
        weather_data['time'] = pd.to_datetime(weather_data['time'], format='%d-%m-%Y')
        city_weather = weather_data[weather_data['City'] == city]
        city_weather.set_index('time', inplace=True)
        model = ARIMA(city_weather['tavg'], order=(5, 1, 0))
        model_fit = model.fit()
        print(f"Model trained successfully for {city}.")
        return model_fit
    except Exception as e:
        print(f"Error training model for {city}:", str(e))
        raise

# Train models for specific cities
try:
    mumbai_weather_model = train_weather_model('Mumbai', weather_data)
    delhi_weather_model = train_weather_model('Delhi', weather_data)
except Exception as e:
    print("Error training models:", str(e))

# Review Model Training
def train_review_model(review_data):
    try:
        X = review_data[['Entrance Fee in INR', 'Google review rating']].values
        y = review_data['Google review rating'].values
        model = LinearRegression()
        model.fit(X, y)
        print("Review model trained successfully.")
        return model
    except Exception as e:
        print("Error training review model:", str(e))
        raise

review_model = train_review_model(reviews_data)

# Recommend Places Based on Budget, Weather, and City
def recommend_places(city, model, budget, avg_temp):
    try:
        print(f"City: {city}, Budget: {budget}, Average Temperature: {avg_temp}")

        # Define favorable temperature range (you can adjust these as needed)
        favorable_temp_min = 20  # Minimum favorable temperature
        favorable_temp_max = 32  # Maximum favorable temperature
        
        # Filter the reviews data based on the city, rating, and temperature criteria
        filtered_data = reviews_data[
            (reviews_data['City'].str.lower() == city.lower()) &
            (reviews_data['Google review rating'] >= 4) &
            (avg_temp >= favorable_temp_min) &
            (avg_temp <= favorable_temp_max)
        ]

        # Sort the filtered data by 'Entrance Fee in INR' in ascending order
        filtered_data = filtered_data.sort_values(by='Entrance Fee in INR', ascending=True)

        # Always include places with fee 0, and then add more places until the budget is met
        recommended_places = []
        total_fee = 0

        for index, row in filtered_data.iterrows():
            if row['Entrance Fee in INR'] == 0 or total_fee + row['Entrance Fee in INR'] <= budget:
                recommended_places.append(row)
                total_fee += row['Entrance Fee in INR']
                print(f"Added {row['Name']}, total fee now: {total_fee}")
            else:
                print(f"Skipping {row['Name']} due to budget constraints.")

        # If no places are found within the budget, return a message indicating so
        if len(recommended_places) == 0:
            return [{'Name': 'No places found within the budget, rating, and temperature criteria.'}]

        # Return the recommended places
        return pd.DataFrame(recommended_places)[['Name', 'City', 'Google review rating', 'Entrance Fee in INR']].to_dict(orient='records')
    
    except Exception as e:
        print("Error in recommend_places:", str(e))
        return [{'Name': 'An error occurred during recommendation.'}]


@app.route('/')
def index():
    return render_template('index.html')

# Flask Route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        
        city = data.get('city')
        start_date = pd.to_datetime(data.get('start_date'))
        end_date = pd.to_datetime(data.get('end_date'))
        budget = float(data.get('budget'))
        
        print("City:", city)
        print("Start Date:", start_date)
        print("End Date:", end_date)
        print("Budget:", budget)
        
        if city.lower() == 'mumbai':
            weather_model = mumbai_weather_model
        elif city.lower() == 'delhi':
            weather_model = delhi_weather_model
        else:
            return jsonify({'response': f"Sorry, I don't have weather data for {city}."})
        
        # Predict weather for the specified date range
        weather_predictions = predict_weather(weather_model, start_date, end_date)
        print("Weather Predictions:", weather_predictions)
        
        # Calculate the average temperature during the specified period
        avg_temp = weather_predictions.mean()
        print(f"Average Temperature: {avg_temp}")
        
        # Recommend places based on the city, budget, and average temperature
        recommendations = recommend_places(city, review_model, budget, avg_temp)
        print("Recommendations:", recommendations)
        
        if recommendations and "The weather in" not in recommendations[0]['Name']:
            response_text = f"The weather in {city} is expected to be favorable during your travel dates. Here are some recommended places: {', '.join([place['Name'] for place in recommendations])}."
        else:
            response_text = recommendations[0]['Name']  # Handle cases where weather is not favorable or no places match criteria
        
        return jsonify({'response': response_text})
    
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'response': f"An error occurred: {str(e)}"}), 500


# Predict Weather
def predict_weather(model, start_date, end_date):
    predictions = model.predict(start=start_date, end=end_date)
    return predictions

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
























# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.linear_model import LinearRegression
# from datetime import datetime

# # Initialize Flask app
# app = Flask(__name__)

# # Load pretrained models and tokenizer from Hugging Face
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# # Load data
# try:
#     weather_data = pd.read_csv('weather_data.csv')
#     reviews_data = pd.read_csv('reviews_data.csv')
#     print("Data loaded successfully.")
# except Exception as e:
#     print("Error loading data:", str(e))

# # ARIMA Weather Model Training (for each city)
# def train_weather_model(city, weather_data):
#     try:
#         weather_data['time'] = pd.to_datetime(weather_data['time'], format='%d-%m-%Y')
#         city_weather = weather_data[weather_data['City'] == city]
#         city_weather.set_index('time', inplace=True)
#         model = ARIMA(city_weather['tavg'], order=(5, 1, 0))
#         model_fit = model.fit()
#         print(f"Model trained successfully for {city}.")
#         return model_fit
#     except Exception as e:
#         print(f"Error training model for {city}:", str(e))
#         raise

# # Train models for specific cities
# try:
#     mumbai_weather_model = train_weather_model('Mumbai', weather_data)
#     delhi_weather_model = train_weather_model('Delhi', weather_data)
# except Exception as e:
#     print("Error training models:", str(e))

# # Review Model Training
# def train_review_model(review_data):
#     try:
#         X = review_data[['Entrance Fee in INR', 'Google review rating']].values
#         y = review_data['Google review rating'].values
#         model = LinearRegression()
#         model.fit(X, y)
#         print("Review model trained successfully.")
#         return model
#     except Exception as e:
#         print("Error training review model:", str(e))
#         raise

# review_model = train_review_model(reviews_data)

# # Recommend Places Based on Budget and Weather
# # Recommend Places Based on Budget, Weather, and City
# def recommend_places(city, model, budget, avg_temp):
#     try:
#         print(f"City: {city}, Budget: {budget}, Average Temperature: {avg_temp}")

#         # Define favorable temperature range (you can adjust these as needed)
#         favorable_temp_min = 20  # Minimum favorable temperature
#         favorable_temp_max = 35  # Maximum favorable temperature
        
#         # Filter the reviews data based on the criteria
#         filtered_data = reviews_data[
#             (reviews_data['City'].str.lower() == city.lower()) &
#             (reviews_data['Entrance Fee in INR'] <= budget) &
#             (reviews_data['Google review rating'] >= 4) &
#             (avg_temp >= favorable_temp_min) &
#             (avg_temp <= favorable_temp_max)
#         ]

#         # Log the number of places after filtering
#         filtered_count = len(filtered_data)
#         print(f"Total places after filtering: {filtered_count}")

#         # Check if the total entrance fee is within the user's budget
#         total_fee = filtered_data['Entrance Fee in INR'].sum()
#         print(f"Total entrance fee: {total_fee}")

#         if total_fee > budget:
#             return [{'Name': 'The total entrance fee exceeds your budget. Please adjust your criteria.'}]

#         # If no places are found, return a message indicating so
#         if filtered_count == 0:
#             return [{'Name': 'No places found within the budget, rating, and temperature criteria.'}]

#         # Return the recommended places
#         return filtered_data[['Name', 'City', 'Google review rating']].to_dict(orient='records')
    
#     except Exception as e:
#         print("Error in recommend_places:", str(e))
#         return [{'Name': 'An error occurred during recommendation.'}]


# @app.route('/')
# def index():
#     return render_template('index.html')

# # Flask Route to handle form submissions
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#         print("Received data:", data)
        
#         city = data.get('city')
#         start_date = pd.to_datetime(data.get('start_date'))
#         end_date = pd.to_datetime(data.get('end_date'))
#         budget = float(data.get('budget'))
        
#         print("City:", city)
#         print("Start Date:", start_date)
#         print("End Date:", end_date)
#         print("Budget:", budget)
        
#         if city.lower() == 'mumbai':
#             weather_model = mumbai_weather_model
#         elif city.lower() == 'delhi':
#             weather_model = delhi_weather_model
#         else:
#             return jsonify({'response': f"Sorry, I don't have weather data for {city}."})
        
#         # Predict weather for the specified date range
#         weather_predictions = predict_weather(weather_model, start_date, end_date)
#         print("Weather Predictions:", weather_predictions)
        
#         # Calculate the average temperature during the specified period
#         avg_temp = weather_predictions.mean()
#         print(f"Average Temperature: {avg_temp}")
        
#         # Recommend places based on the city, budget, and average temperature
#         recommendations = recommend_places(city, review_model, budget, avg_temp)
#         print("Recommendations:", recommendations)
        
#         response_text = f"The weather in {city} is expected to be favorable during your travel dates. Here are some recommended places: {', '.join([place['Name'] for place in recommendations])}."
        
#         return jsonify({'response': response_text})
    
#     except Exception as e:
#         print("Error during prediction:", str(e))
#         return jsonify({'response': f"An error occurred: {str(e)}"}), 500


# # Predict Weather
# def predict_weather(model, start_date, end_date):
#     predictions = model.predict(start=start_date, end=end_date)
#     return predictions

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')

















