from src.data_processing import load_data, clean_data, save_processed_data
from src.model_training import train_model, save_model
from src.model_evaluation import load_model, load_test_data
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
def main():
    # Load raw data
    raw_data_path = os.path.join('data', 'raw', 'Nigerian_Road_Traffic_Crashes_2020_2024.csv')
    df = load_data(raw_data_path)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Save cleaned data
    processed_data_path = os.path.join('data', 'processed', 'cleaned_Nigerian_Road_Traffic_Crashes_2020_2024.csv')
    save_processed_data(df_cleaned, processed_data_path)
    
    # Load the cleaned data for training
    df_cleaned = pd.read_csv(processed_data_path)
    
    # Feature selection
    features = ['Quarter_Num', 'Year', 'State', 'Total_Vehicles_Involved', 'SPV', 'DAD', 'PWR', 'FTQ']
    target = 'Total_Crashes'
    X = df_cleaned[features]
    y = df_cleaned[target]
    
    # Handling categorical features
    categorical_features = ['State']
    numeric_features = [feature for feature in features if feature not in categorical_features]
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    # Fit and transform preprocessing on the cleaned data
    X_processed = preprocessor.fit_transform(X)
    
    # Train the model
    model = train_model(X_processed, y)
    
    # Save the model and preprocessor
    model_save_path = os.path.join('models', 'Nigerian_Road_Traffic_Crashes_2020_2024.pkl')
    preprocessor_save_path = os.path.join('models', 'preprocessor.pkl')
    save_model(model, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)
    
    # Load the model and preprocessor for evaluation
    model = load_model(model_save_path)
    preprocessor = joblib.load(preprocessor_save_path)
    
    # Apply preprocessing to test data
    df_test = pd.read_csv(processed_data_path)
    X_test = df_test[features]
    y_test = df_test[target]
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    
    # Bin the continuous target variable for classification metrics
    bins = [0, 5, 10, 15, 20, float('inf')]
    labels = [1, 2, 3, 4, 5]
    y_binned = pd.cut(y_test, bins=bins, labels=labels, right=False)
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels, right=False)
    
    # Encode target variable for classification metrics
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_binned)
    y_pred_encoded = le.transform(y_pred_binned)
    
    # Print classification metrics
    print("Classification Report:\n", classification_report(y_encoded, y_pred_encoded, target_names=[str(label) for label in labels]))
    print("Confusion Matrix:\n", confusion_matrix(y_encoded, y_pred_encoded))
    
    # Print regression metrics
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

if __name__ == "__main__":
    main()
