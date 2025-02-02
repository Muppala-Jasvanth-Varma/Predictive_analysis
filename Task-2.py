import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv(r"W:\CODETECH\Dataset.csv")  # Update with actual file path

# Handle missing values
data['Borough'].fillna('Unknown', inplace=True)
data['Zone'].fillna('Unknown', inplace=True)
data['service_zone'].fillna('Unknown', inplace=True)

# Check dataset structure
print("Dataset Sample:\n", data.head())

# Encode categorical features
label_enc_zone = LabelEncoder()
label_enc_service = LabelEncoder()

data['Zone'] = label_enc_zone.fit_transform(data['Zone'])
data['service_zone'] = label_enc_service.fit_transform(data['service_zone'])

# Define features and target
X = data[['LocationID', 'Zone', 'service_zone']]
y = data['Borough']  # Target variable

# Encode the target (Borough)
label_enc_borough = LabelEncoder()
y = label_enc_borough.fit_transform(y)

# Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Fixing the mismatch by using np.unique(y_test) to ensure consistent class labels
print('Classification Report:\n', classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=label_enc_borough.classes_))

# Save encoders and model for future use
import pickle
with open('borough_model.pkl', 'wb') as f:
    pickle.dump((model, label_enc_zone, label_enc_service, label_enc_borough), f)
