import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import os

def load_model(model_path):
    """Load the saved model."""
    try:
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_preprocessor(preprocessor_path):
    """Load the saved preprocessing pipeline."""
    try:
        return joblib.load(preprocessor_path)
    except Exception as e:
        print(f"Error loading preprocessor: {e}")
        raise

def load_test_data(file_path):
    """Load test data for evaluation."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading test data: {e}")
        raise

if __name__ == "__main__":
    # Load model and preprocessor
    model_path = os.path.join('models', 'Nigerian_Road_Traffic_Crashes_2020_2024.pkl')
    preprocessor_path = os.path.join('models', 'preprocessor.pkl')
    
    model = load_model(model_path)
    preprocessor = load_preprocessor(preprocessor_path)
    
    # Load test data
    test_data_path = os.path.join('data', 'processed', 'cleaned_Nigerian_Road_Traffic_Crashes_2020_2024.csv')
    df = load_test_data(test_data_path)
    
    features = ['Quarter_Num', 'Year', 'State', 'Total_Vehicles_Involved', 'SPV', 'DAD', 'PWR', 'FTQ']
    target = 'Total_Crashes'
    
    # Feature selection
    X_test = df[features]
    y_test = df[target]
    
    # Convert continuous target to categorical bins
    bins = [0, 5, 10, 15, 20, float('inf')]
    labels = [1, 2, 3, 4, 5]  # Adjust as needed
    y_test_binned = pd.cut(y_test, bins=bins, labels=labels, right=False)
    
    # Encode target variable for classification metrics
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test_binned)
    
    # Apply preprocessing to test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    
    # Bin the predictions to match the test target bins
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels, right=False)
    y_pred_encoded = le.transform(y_pred_binned)
    
    # Print classification metrics
    print("Classification Report:\n", classification_report(y_test_encoded, y_pred_encoded, target_names=[str(label) for label in labels]))
    print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred_encoded))
