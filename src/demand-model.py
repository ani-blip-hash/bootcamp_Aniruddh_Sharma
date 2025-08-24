import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor

def train_model(data_path):
    df = pd.read_csv(data_path)
    features = ["hour", "temperature", "humidity", "day_of_week"]
    target = "demand"
    
    X = df[features]
    y = df[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "sample_bike_data.csv")
    
    model = train_model(data_path)
    print("Model trained successfully!")
