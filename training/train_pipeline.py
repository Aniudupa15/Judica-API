import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Data Ingestion class to load data
class DataIngestion:
    @staticmethod
    def load_data(file_path):
        return pd.read_csv(file_path)

# Data Transformation class to handle preprocessing
class DataTransformation:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def fit_transform(self, data):
        # Assuming categorical columns need encoding
        categorical_columns = ['statute', 'offense_category', 'penalty']  # Adjust as per your data

        for col in categorical_columns:
            if col in data.columns:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                self.label_encoders[col] = encoder

        # Scaling numerical features
        numerical_columns = ['imprisonment_duration_served', 'risk_score', 'penalty_severity']
        data[numerical_columns] = self.scaler.fit_transform(data[numerical_columns])

        return data

# Model Trainer class for training and saving model
class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        accuracy = self.model.score(X_test, y_test)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

# Ensure the ipc_vector_db folder exists
models_dir = os.path.join("ipc_vector_db")  # Directory for models and preprocessing objects
os.makedirs(models_dir, exist_ok=True)

# Load and preprocess data
data = DataIngestion.load_data('data/a.csv')  # Replace with your data path
transformer = DataTransformation()
data = transformer.fit_transform(data)

# Split data into features and target
X = data.drop(columns=['case_id', 'bail_eligibility'])
y = data['bail_eligibility']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
trainer = ModelTrainer()
trainer.train(X_train, y_train)

# Evaluate the model
trainer.evaluate(X_test, y_test)

# Save model and preprocessing objects
model_path = os.path.join(models_dir, 'bail_reckoner_model.pkl')
preprocessing_path = os.path.join(models_dir, 'preprocessing_objects.pkl')

trainer.save_model(model_path)
joblib.dump({'label_encoders': transformer.label_encoders, 'scaler': transformer.scaler}, preprocessing_path)

print(f"Model and preprocessing objects saved to {models_dir}")
