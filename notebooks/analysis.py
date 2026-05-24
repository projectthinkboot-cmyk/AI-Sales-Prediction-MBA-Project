import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==============================
# LOAD DATASET
# ==============================

data = pd.read_csv("../dataset/Advertising Budget and Sales.csv")

# ==============================
# DISPLAY DATA
# ==============================

print("========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== DATASET INFO ==========")
print(data.info())

print("\n========== DATASET DESCRIPTION ==========")
print(data.describe())

# ==============================
# DATA VISUALIZATION
# ==============================

plt.figure(figsize=(8,5))

plt.scatter(
    data['TV Ad Budget ($)'],
    data['Sales ($)']
)

plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")

plt.show()

# ==============================
# FEATURES & TARGET
# ==============================

X = data[
    [
        'TV Ad Budget ($)',
        'Radio Ad Budget ($)',
        'Newspaper Ad Budget ($)'
    ]
]

y = data['Sales ($)']

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# TRAIN MODEL
# ==============================

model = LinearRegression()

model.fit(X_train, y_train)

# ==============================
# PREDICTIONS
# ==============================

predictions = model.predict(X_test)

print("\n========== PREDICTED SALES ==========")

for i in range(5):
    print(f"Prediction {i+1}: {predictions[i]:.2f}")

# ==============================
# MODEL ACCURACY
# ==============================

score = r2_score(y_test, predictions)

print("\n========== MODEL ACCURACY ==========")
print(f"R2 Score: {score:.2f}")

# ==============================
# AI BUSINESS RECOMMENDATION
# ==============================

average_prediction = predictions.mean()

print("\n========== AI BUSINESS RECOMMENDATIONS ==========")

if average_prediction < 10:
    print("• Increase marketing budget.")
    print("• Focus on customer awareness campaigns.")
    print("• Improve product promotions.")

elif average_prediction < 20:
    print("• Focus on digital advertising.")
    print("• Launch seasonal discount campaigns.")
    print("• Improve customer engagement.")

else:
    print("• Increase inventory stock.")
    print("• Expand marketing campaigns.")
    print("• Focus on high-performing products.")
    print("• Explore business expansion opportunities.")

# ==============================
# SAMPLE CUSTOM PREDICTION
# ==============================

print("\n========== SAMPLE FUTURE SALES PREDICTION ==========")

sample_data = [[230, 40, 50]]

future_sales = model.predict(sample_data)

print(f"Predicted Future Sales: {future_sales[0]:.2f}")