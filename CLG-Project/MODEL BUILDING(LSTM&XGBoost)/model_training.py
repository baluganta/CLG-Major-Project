from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import numpy as np
import pandas as pd
import warnings

df = pd.read_csv("up real data1.csv")

# Drop unused columns
df.drop(columns=["Transaction ID", "User ID", "Timestamp", "IP Address"], inplace=True)

# Encode categorical features
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Normalize features
scaler = MinMaxScaler()
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Separate input and target
X = df.drop(columns=["Fraudulent"])
y = df["Fraudulent"].astype(int)

# Balance using SMOTE
smote = SMOTE(sampling_strategy=0.7, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Reshape for LSTM (samples, time steps, features)
X_resampled = np.array(X_resampled).reshape(X_resampled.shape[0], 1, X_resampled.shape[1])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled)

# LSTM model
lstm_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = LSTM(128, return_sequences=True)(lstm_input)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = LSTM(64)(x)
x = Dropout(0.3)(x)
lstm_output = Dense(1, activation='sigmoid')(x)
lstm_model = Model(inputs=lstm_input, outputs=lstm_output)
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train LSTM
lstm_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Extract LSTM features
X_train_lstm = lstm_model.predict(X_train)
X_test_lstm = lstm_model.predict(X_test)

# XGBoost classifier on LSTM output
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_lstm, y_train)

# Evaluate
y_pred_xgb = xgb_model.predict(X_test_lstm)
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# Save models
lstm_model.save('lstm_model.h5')
import joblib
joblib.dump(xgb_model, 'xgb_model.pkl')
