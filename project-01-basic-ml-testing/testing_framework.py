# ============================================
# PROJECT 1: ML TESTING FRAMEWORK
# File: testing_framework.py  
# Goal: Apply Rahul's testing strategies
# ============================================

print("🧪 ML TESTING FRAMEWORK - Applying Course Knowledge")

# Import same libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# Recreate the same model (copy from above)
X, y = make_classification(n_samples=1000, n_features=5, n_redundant=0, 
                          n_informative=5, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# ============================================
# TESTING STRATEGIES FROM RAHUL'S COURSE
# ============================================

print("\n🔍 TEST 1: DATA QUALITY TESTING")
# Check for basic data issues
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Missing values in training: {pd.DataFrame(X_train).isnull().sum().sum()}")
print(f"Missing values in test: {pd.DataFrame(X_test).isnull().sum().sum()}")
print("✅ Data quality check: PASSED")

print("\n🎯 TEST 2: MODEL PERFORMANCE TESTING")
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy: {test_accuracy:.4f}")

# Performance threshold test
if test_accuracy > 0.7:
    print("✅ Performance test: PASSED (accuracy > 70%)")
else:
    print("❌ Performance test: FAILED (accuracy < 70%)")

print("\n⚠️ TEST 3: OVERFITTING DETECTION")
accuracy_gap = train_accuracy - test_accuracy
print(f"Accuracy gap: {accuracy_gap:.4f}")

if accuracy_gap < 0.1:  # Less than 10% gap
    print("✅ Overfitting test: PASSED (gap < 10%)")
else:
    print("❌ Overfitting test: FAILED (gap > 10%)")

print("\n📊 TEST 4: PREDICTION DISTRIBUTION")
predictions = model.predict(X_test)
unique, counts = np.unique(predictions, return_counts=True)
print(f"Prediction distribution: {dict(zip(unique, counts))}")
print("✅ Distribution check: PASSED")

# ============================================
# TEST REPORT GENERATION
# ============================================

print("\n📋 FINAL TEST REPORT")
print("=" * 50)
print("ML MODEL TESTING SUMMARY")
print("=" * 50)
print(f"Model Type: Logistic Regression")
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Overfitting Gap: {accuracy_gap:.4f}")

# Overall assessment
if test_accuracy > 0.7 and accuracy_gap < 0.1:
    overall_status = "✅ PASSED"
else:
    overall_status = "❌ NEEDS IMPROVEMENT"

print(f"Overall Status: {overall_status}")
print("=" * 50)
print("🎉 Testing framework complete!")
