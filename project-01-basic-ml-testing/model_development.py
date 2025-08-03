# ============================================
# PROJECT 1: BASIC ML TESTING FRAMEWORK
# File: model_development.py
# Goal: Build simple model to test
# ============================================

print("🚀 PROJECT 1: Building ML Model for Testing")

# Import libraries (copy-paste approach - no need to understand deeply)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Step 1: Create test data (we control the quality)
print("📊 Creating test dataset...")
X, y = make_classification(n_samples=1000, n_features=5, n_redundant=0, 
                          n_informative=5, n_clusters_per_class=1, random_state=42)

# Convert to DataFrame (like Excel spreadsheet)
feature_names = ['age', 'income', 'education', 'experience', 'score']
df = pd.DataFrame(X, columns=feature_names)
df['approved'] = y  # 0 = rejected, 1 = approved

print(f"✅ Dataset created with {len(df)} records")
print("Sample data:")
print(df.head())

# Step 2: Split data for testing (basic ML practice)
print("\n🔀 Splitting data for training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train simple model
print("\n🤖 Training the model...")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 4: Get basic results
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"✅ Model training complete!")
print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Save results for testing
predictions = model.predict(X_test)
print(f"\n📈 Model made {len(predictions)} predictions on test data")
print("✅ Ready for testing phase!")
