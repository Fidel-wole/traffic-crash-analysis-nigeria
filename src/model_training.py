import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_processed_data(file_path):
    """Load processed data for training."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """Save the trained model."""
    try:
        joblib.dump(model, model_path)
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def save_preprocessor(preprocessor, preprocessor_path):
    """Save the preprocessing pipeline."""
    try:
        joblib.dump(preprocessor, preprocessor_path)
    except Exception as e:
        print(f"Error saving preprocessor: {e}")
        raise

if __name__ == "__main__":
    # Load the cleaned data
    processed_data_path = os.path.join('data', 'processed', 'cleaned_Nigerian_Road_Traffic_Crashes_2020_2024.csv')
    df = load_processed_data(processed_data_path)
    
    features = ['Quarter_Num', 'Year', 'State', 'Total_Vehicles_Involved', 'SPV', 'DAD', 'PWR', 'FTQ']
    target = 'Total_Crashes'

    # Ensure the target is numeric if performing regression
    if df[target].dtype == 'object':
        raise ValueError(f"Target column {target} should be numeric for regression.")

    # Feature selection
    X = df[features]  # Features
    y = df[target]  # Target column
    
    # Handling categorical features
    categorical_features = ['State']
    numeric_features = [feature for feature in features if feature not in categorical_features]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)
    
    # Transform both training and test data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train the model
    model = train_model(X_train_processed, y_train)
    
    # Save the model and preprocessor
    model_save_path = os.path.join('models', 'Nigerian_Road_Traffic_Crashes_2020_2024.pkl')
    preprocessor_save_path = os.path.join('models', 'preprocessor.pkl')
    save_model(model, model_save_path)
    save_preprocessor(preprocessor, preprocessor_save_path)

    print(f"Model saved to {model_save_path}")
    print(f"Preprocessor saved to {preprocessor_save_path}")
