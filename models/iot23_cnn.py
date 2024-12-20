# -*- coding: utf-8 -*-
"""IoT23 - CNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1daP3s042fFeZLgnSArlU-aQM7YDyuf7V
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
import joblib

filepath = "iot23_combined.csv"
df = pd.read_csv(filepath)

df

# Remove unnamed column if exists
if 'Unnamed: 0' in df.columns:
    del df['Unnamed: 0']

# Display first few rows
print("First few rows of the dataset:")
df.head()

# Basic dataset information
print("Dataset Information:")
df.info()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Analyze attack distribution
plt.figure(figsize=(15, 6))
df['label'].value_counts().plot(kind='bar')
plt.title('Distribution of Attack Types')
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nClass Distribution:")
print(df['label'].value_counts())
print("\nClass Distribution (%):")
print(df['label'].value_counts(normalize=True) * 100)

# Correlation analysis for numeric features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()

# Prepare features
features = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts',
           'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'proto_icmp', 'proto_tcp',
           'proto_udp', 'conn_state_OTH', 'conn_state_REJ', 'conn_state_RSTO',
           'conn_state_RSTOS0', 'conn_state_RSTR', 'conn_state_RSTRH', 'conn_state_S0',
           'conn_state_S1', 'conn_state_S2', 'conn_state_S3', 'conn_state_SF',
           'conn_state_SH', 'conn_state_SHR']

X = df[features].values
Y = pd.get_dummies(df['label']).values

print("Features shape:", X.shape)
print("Labels shape:", Y.shape)

# Normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Save scaler for future use
joblib.dump(scaler, 'iot23_scaler.pkl')

# Analyze normalized data
print("Feature Statistics after Normalization:")
print(pd.DataFrame(X_normalized, columns=features).describe())

X.shape

Y = pd.get_dummies(df['label']).values

Y.shape

#X = df[['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'proto_icmp', 'proto_tcp', 'proto_udp', 'conn_state_OTH', 'conn_state_REJ', 'conn_state_RSTO', 'conn_state_RSTOS0', 'conn_state_RSTR', 'conn_state_RSTRH', 'conn_state_S0', 'conn_state_S1', 'conn_state_S2', 'conn_state_S3', 'conn_state_SF', 'conn_state_SH', 'conn_state_SHR']]
#Y = df[['label']]

X

df

X_normalized

X_normalized.shape

scaler.fit(Y)

Y_normalized= scaler.transform(Y)

Y_normalized

# Visualize normalized features
plt.figure(figsize=(15, 6))
pd.DataFrame(X_normalized, columns=features).boxplot()
plt.title('Distribution of Normalized Features')
plt.xlabel('Features')
plt.ylabel('Normalized Values')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Reshape data for CNN
X_reshaped = X_normalized.reshape((X_normalized.shape[0], X_normalized.shape[1], 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, Y,
                                                    random_state=10,
                                                    test_size=0.2)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Define CNN model
def create_model():
    model = Sequential()

    # Convolutional layers
    model.add(Conv1D(32, 3, padding="same", activation="relu",
                    input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.5))
    model.add(Conv1D(64, 3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.5))

    # Dense layers
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation="softmax"))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

    return model

# Create model
model = create_model()
model.summary()

# Create checkpoint callback
checkpoint_path = "iot23_model_CNN_v1.keras"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train model
start_time = time.time()
history = model.fit(X_train, y_train,
                   epochs=50,
                   batch_size=256,
                   validation_data=(X_test, y_test),
                   callbacks=[checkpoint],
                   verbose=1)
training_time = time.time() - start_time

# Save final model
model.save('iot23_final_model_CNN_v1.keras')

print(f"\nTraining Time: {training_time:.2f} seconds")

# Evaluate model on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.tight_layout()
plt.show()

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Classification report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Confusion matrix
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

def load_model_and_predict(data):
    """
    Load saved model and make predictions on new data
    """
    # Load model and scaler
    model = tf.keras.models.load_model('iot23_final_model_CNN_v1.keras')
    scaler = joblib.load('iot23_scaler.pkl')

    # Preprocess data
    data_normalized = scaler.transform(data)
    data_reshaped = data_normalized.reshape((data_normalized.shape[0],
                                           data_normalized.shape[1], 1))

    # Make predictions
    predictions = model.predict(data_reshaped)
    predicted_classes = np.argmax(predictions, axis=1)

    return predicted_classes, predictions

# Demo with sample data
sample_data = X[:5]  # Take first 5 samples
predicted_classes, predictions = load_model_and_predict(sample_data)
print("Predicted classes for sample data:", predicted_classes)
print("\nPrediction probabilities:")
print(predictions)