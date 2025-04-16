from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import requests
import gzip
from io import BytesIO
from flasgger import Swagger

app = Flask(__name__)
# Swagger config
app.config['SWAGGER'] = {
    'title': 'Car Prices',
    'uiversion': 3
}
swagger = Swagger(app)

# SQLite DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///listings.db'
db = SQLAlchemy(app)

# Define a database model
class CarListing(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # Unique ID
    brand = db.Column(db.String(50), nullable=False)  # Car brand 
    model = db.Column(db.String(100), nullable=False)  # Car model 
    model_year = db.Column(db.Integer, nullable=False)  # Car year
    milage = db.Column(db.Integer, nullable=False)  # Current mileage on car
    fuel_type = db.Column(db.String(30), nullable=False)  # Gasoline, Diesel, Electric, etc.
    engine = db.Column(db.String(100), nullable=False)  # Engine details (e.g., "3.5L V6 24V")
    transmission = db.Column(db.String(50), nullable=False)  # Automatic/Manual/8-Speed
    ext_col = db.Column(db.String(30), nullable=False)  # Exterior color
    int_col = db.Column(db.String(30), nullable=False)  # Interior color
    accident = db.Column(db.String(100), nullable=False)  # Stores raw accident description
    clean_title = db.Column(db.Boolean, nullable=False)  # True = Clean title, False = Salvaged title
    price = db.Column(db.Float, nullable=False)  # Car price

    def __repr__(self):
        return f"<CarListing {self.brand} {self.model} ({self.model_year})>"

# Create the database
def create_database():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    create_database()
    app.run(debug=True)


def preprocess_data(df):
    # Clean the price column
    df['price'] = df['price'].replace({r'\$': '', ',': ''}, regex=True).astype(float)

    # Drop rows where any of the key fields are NaN
    df = df.dropna(subset=['price', 'model_year', 'milage', 'fuel_type', 'transmission'])


    # One more time, fill any missing numerical values with the median, just in case.
    df['milage'] = df['milage'].fillna(df['milage'].median())
    df['model_year'] = df['model_year'].fillna(df['model_year'].median())
    df['price'] = df['price'].fillna(df['price'].median()) 


    # Fill missing categorical values (neighbourhood) with the most frequent value
    df['brand'] = df['brand'].fillna(df['brand'].mode()[0])
    df['model'] = df['model'].fillna(df['model'].mode()[0])
    df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])
    df['transmission'] = df['transmission'].fillna(df['transmission'].mode()[0])
    df['ext_col'] = df['ext_col'].fillna(df['ext_col'].mode()[0])
    df['int_col'] = df['int_col'].fillna(df['int_col'].mode()[0])

    # One-hot encode the 'neighbourhood_cleansed' column
    encoder = OneHotEncoder(sparse_output=False)
    categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col'] 

    encoded_data = encoder.fit_transform(df[categorical_cols])

    # Create a DataFrame for the one-hot encoded neighborhoods
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

    # Concatenate the encoded neighborhood with the original dataframe
    df = pd.concat([df, encoded_df], axis=1).drop(columns=categorical_cols)

    # Drop any rows that still have NaN values at this point (forcefully)
    df = df.dropna()
    return df, encoder

# Global variables for model and encoder
model = None
encoder = None

@app.route('/reload', methods=['POST'])
def reload_data():
    '''
    Reload data from the Airbnb dataset, clear the database, load new data, and return summary stats
    ---
    responses:
      200:
        description: Summary statistics of reloaded data
    '''
    global model, encoder

    # Step 1: Download and decompress data
    url = "https://raw.githubusercontent.com/impolichetti/airbnb/refs/heads/master/used_cars.csv"

    # Step 2: Load data into pandas
    cars_df = pd.read_csv(url)

    # Step 3: Clear the database
    db.session.query(CarListing).delete()
    db.session.commit()

    # Step 4: Process data and insert it into the database
    cars_df = cars_df[['price', 'brand', 'model_year', 'milage', 'fuel_type', 'transmission']].dropna()

    # Ensure 'price' is numeric
    cars_df['price'] = pd.to_numeric(cars_df['price'], errors='coerce')

    # Insert each row as a new record in the database
    for _, row in cars_df.iterrows(): 
        new_car = CarListing(
            price=row['price'],
            brand=row['brand'],
            model_year=int(row['model_year']),
            milage=row['milage'],
            fuel_type=row['fuel_type'],
            transmission=row['transmission'],
            ext_col="Unknown",  # Provide default values for missing columns
            int_col="Unknown",
            accident="Unknown",
            clean_title=True
        )
        db.session.add(new_car)
    db.session.commit()  
    # Step 5: Preprocess and train model
    df, encoder = preprocess_data(cars_df)
    X = df.drop(columns='price')
    y = df['price']
    model = LinearRegression()
    model.fit(X, y)

    # Step 6: Generate summary statistics
    summary = {
        'total_cars': len(cars_df),
        'average_price': cars_df['price'].mean(),
        'min_price': cars_df['price'].min(),
        'max_price': cars_df['price'].max(),
        'average_mileage': cars_df['milage'].mean(),
        'common_brand': cars_df['brand'].value_counts().idxmax(),
        'common_fuel': cars_df['fuel_type'].value_counts().idxmax(),
    }

    return jsonify(summary)  

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict car prices
    ---
    parameters:
  - name: body
    in: body
    required: true
    schema:
      type: object
      properties:
        model_year:
          type: integer
        milage:
          type: integer
        fuel_type:
          type: string
        transmission:
          type: string
        brand:
          type: string

    responses:
      200:
        description: Predicted car prices
    '''
    global model, encoder  # Ensure that the encoder and model are available for prediction


    # Check if the model and encoder are initialized
    if model is None or encoder is None:
        return jsonify({"error": "The data has not been loaded. Please refresh the data by calling the '/reload' endpoint first."}), 400

    data = request.json
    try:
        data = request.json
        model_year = pd.to_numeric(data.get('model_year'), errors='coerce')
        mileage = pd.to_numeric(data.get('milage'), errors='coerce')
        fuel_type = data.get('fuel_type')
        transmission = data.get('transmission')
        brand = data.get('brand')

        if None in [model_year, milage, fuel_type, transmission, brand]:
            return jsonify({"error": "Missing or invalid required parameters"}), 400
        
        categorical_values = [[brand, fuel_type, transmission]]
        categorical_encoded = encoder.transform(categorical_values)
        input_data = np.hstack(([model_year, mileage], categorical_encoded[0])).reshape(1, -1)
        predicted_price = float(model.predict(input_data)[0])

        return jsonify({"predicted_price": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

