import os
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# --------------------- Data Ingestion ---------------------
class DataIngestion:
    @staticmethod
    def load_data(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_csv(file_path)

# --------------------- Data Transformation ---------------------
class DataTransformation:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def safe_len(self, x):
        """Safely count list-like strings."""
        try:
            val = ast.literal_eval(x)
            return len(val) if isinstance(val, list) else 0
        except (ValueError, SyntaxError, TypeError):
            return 0

    def fit_transform(self, data):
        # Derived features
        data['ipc_section_count'] = data['ipc_sections'].apply(lambda x: self.safe_len(x) if pd.notna(x) else 0)
        data['has_special_law'] = data['special_laws'].apply(lambda x: 0 if pd.isna(x) else 1)
        data['prior_cases_count'] = data['prior_cases'].apply(lambda x: self.safe_len(x) if pd.notna(x) else 0)

        # Keep only relevant columns
        features = [
            'ipc_section_count', 'has_special_law', 'bail_type',
            'bail_cancellation_case', 'prior_cases_count', 'crime_type'
        ]
        data = data[features + ['bail_outcome']]

        # Encode categorical columns
        categorical_columns = ['bail_type', 'crime_type']
        for col in categorical_columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col].astype(str))
            self.label_encoders[col] = encoder

        # Encode target (bail_outcome: Granted = 1, Rejected = 0)
        data['bail_outcome'] = data['bail_outcome'].apply(
            lambda x: 1 if str(x).lower() == 'granted' else 0
        )

        # Convert binary and numeric columns
        data['bail_cancellation_case'] = data['bail_cancellation_case'].fillna(0).astype(int)
        numeric_cols = ['ipc_section_count', 'prior_cases_count']
        data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])

        return data

# --------------------- Model Trainer ---------------------
class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, y_train, X_test, y_test):
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        print(f"Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

# --------------------- Paths ---------------------
data_path = "C:/Users/ASUS/Desktop/Projects/Judica-API/training/data/indian_bail_judgements.csv"
models_dir = "models/models_v2"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "bail_reckoner_model.pkl")
preprocessing_path = os.path.join(models_dir, "preprocessing_objects.pkl")

# --------------------- Pipeline Execution ---------------------
print("Starting training pipeline v2...")

# Load and preprocess data
data = DataIngestion.load_data(data_path)
print(f"Loaded dataset with shape: {data.shape}")

transformer = DataTransformation()
data = transformer.fit_transform(data)
print("Preprocessing completed.")

# Split features and target
X = data.drop(columns=['bail_outcome'])
y = data['bail_outcome']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
trainer = ModelTrainer()
trainer.train(X_train, y_train)
trainer.evaluate(X_train, y_train, X_test, y_test)

# Save model and preprocessing objects
trainer.save_model(model_path)
joblib.dump({'label_encoders': transformer.label_encoders, 'scaler': transformer.scaler}, preprocessing_path)

print(f"✅ Model and preprocessing objects saved to {models_dir}")
