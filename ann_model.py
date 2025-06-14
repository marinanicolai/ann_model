# 📦 1. Import Libraries
import streamlit as st
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

st.title(" ANN Model Explorer by Marina")

# 📥 2. Load the Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
st.subheader("Sample of Dataset")
st.dataframe(dataset.head())

# 🧹 3. Select Features and Target
X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, -1].values

# 🔠 4. Label Encode the Gender Column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# 🌍 5. One-Hot Encode the Geography Column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# 🧪 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 🧮 7. Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 🏗️ 8. Build ANN
ann = Sequential()
ann.add(Dense(units=6, activation='relu'))
ann.add(Dense(units=6, activation='relu'))
ann.add(Dense(units=1, activation='sigmoid'))

# ⚙️ 9. Compile the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🏋️ 10. Train the model
with st.spinner("Training the model..."):
    history = ann.fit(X_train, y_train, validation_split=0.1, batch_size=10, epochs=10, verbose=0)

# 📊 11. Plot training history
def plotModelHistory(h):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    ax[0].plot(h.history['loss'])
    ax[0].plot(h.history['val_loss'])
    ax[0].legend(['loss', 'val_loss'])
    ax[0].set_title("Train Loss vs Validation Loss")

    ax[1].plot(h.history['accuracy'])
    ax[1].plot(h.history['val_accuracy'])
    ax[1].legend(['accuracy', 'val_accuracy'])
    ax[1].set_title("Train Accuracy vs Validation Accuracy")

    return fig

st.subheader("📈 Model Training History")
st.write(f"**Max Training Accuracy:** {max(history.history['accuracy']):.2f}")
st.write(f"**Max Validation Accuracy:** {max(history.history['val_accuracy']):.2f}")
st.pyplot(plotModelHistory(history))

# 🧪 12. Predict on Test Set
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
pred_df = pd.DataFrame(np.concatenate((y_pred.reshape(-1,1), y_test.reshape(-1,1)), axis=1), columns=['Predicted', 'Actual'])

st.subheader("🔍 Test Set Predictions (first 10 rows)")
st.dataframe(pred_df.head(10))

# 🧾 13. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

st.subheader("🔢 Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax_cm)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig_cm)

# 📋 14. Classification Report
clr = classification_report(y_test, y_pred, output_dict=False)
st.subheader("📋 Classification Report")
st.text(clr)

# ✅ 15. Accuracy Score
acc = accuracy_score(y_test, y_pred) * 100
st.subheader("✅ Accuracy Score")
st.write(f"{acc:.2f}%")

# 🔮 16. Predicting a Single Customer
st.subheader("🔮 Predict a Single Customer")

input_sample = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
scaled_sample = sc.transform(input_sample)
single_prediction = ann.predict(scaled_sample)[0][0]
prediction_result = "LEAVES the bank" if single_prediction > 0.5 else "STAYS in the bank"
st.write(f"Model prediction: **This customer {prediction_result}.**")
