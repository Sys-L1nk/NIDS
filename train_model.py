import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

# 1. Load the cleaned dataset
print("Loading dataset...")
df = pd.read_csv('cleaned_combined_dataset.csv')

# Clean column names
df.columns = df.columns.str.strip()

# --- THE FIX: Clean out any remaining NaN or infinite values ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['Label'], inplace=True)
df.dropna(inplace=True)
# -------------------------------------------------------------

# 2. Separate Features (X) from the Target Label (y)
X = df.drop(columns=['Label']) 
y = df['Label']

# 3. Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize the Algorithm
print("Training the Random Forest Model (This may take a few minutes)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 5. Train the Model
model.fit(X_train, y_train)

# 6. Test the Model's Accuracy
print("Evaluating the model on test data...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Performance Report:")
print(classification_report(y_test, predictions))

# 7. Save the Trained Model
joblib.dump(model, 'nids_model.pkl')
print("Model saved successfully as 'nids_model.pkl'")