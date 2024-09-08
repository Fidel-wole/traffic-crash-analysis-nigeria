import pandas as pd
import os

def load_data(file_path):
    """Load the CSV file."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def clean_data(df):
    """Handle missing values, outliers, and data types."""
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Print column names to debug
    print("Columns in DataFrame:", df.columns)
    
    # Convert 'Quarter' column to a numerical format
    if 'Quarter' in df.columns:
        # Extract the quarter number and year
        df[['Quarter_Num', 'Year']] = df['Quarter'].str.extract(r'Q(\d) (\d{4})')
        
        # Convert extracted columns to integer
        df['Quarter_Num'] = df['Quarter_Num'].astype(int)
        df['Year'] = df['Year'].astype(int)
        
        # Optionally, drop the original 'Quarter' column
        df.drop('Quarter', axis=1, inplace=True)
    
    # Print DataFrame head to verify changes
    print("DataFrame head after cleaning:\n", df.head())
    
    return df

def save_processed_data(df, save_path):
    """Save the processed data to the processed folder."""
    try:
        df.to_csv(save_path, index=False)
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

if __name__ == "__main__":
    # Load raw data
    raw_data_path = os.path.join('data', 'raw', 'Nigerian_Road_Traffic_Crashes_2020_2024.csv')
    df = load_data(raw_data_path)
    
    # Print the first few rows and columns of the DataFrame
    print("DataFrame head:\n", df.head())
    print("Columns in DataFrame:", df.columns)
    
    # Clean the data
    df_cleaned = clean_data(df)
    
    # Save the cleaned data
    processed_data_path = os.path.join('data', 'processed', 'cleaned_Nigerian_Road_Traffic_Crashes_2020_2024.csv')
    save_processed_data(df_cleaned, processed_data_path)

    print(f"Cleaned data saved to {processed_data_path}")
