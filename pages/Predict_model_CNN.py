import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cài đặt giao diện trang
st.set_page_config(
    page_title="Traffic Detection System",
    page_icon="🔍",
    layout="wide"
)

# Đường dẫn tới model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_cnn_kdd_v1.keras")

# Features vector
FEATURES = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
            "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label",
            "difficulty"]

def load_model():
    """Load the trained model"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at: {MODEL_PATH}")
            return None
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocessing_anomaly(data):
    """
    Preprocessing function exactly matching the training preprocessing
    """
    try:
        # Create a copy of the data
        df = data.copy()

        # Mã hóa các cột phân loại
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Tạo nhãn one-hot
        if 'label' in df.columns:
            labels = df['label'].apply(lambda x: 0 if x == 'normal' else 1).values
            y = np.zeros((len(labels), 2))
            for i, label in enumerate(labels):
                y[i][label] = 1
            df = df.drop('label', axis=1)
        else:
            y = None

        # Xóa cột difficulty
        if 'difficulty' in df.columns:
            df = df.drop('difficulty', axis=1)

        # Chuyển đổi tất cả dữ liệu thành số
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)

        # Chuẩn hóa dữ liệu về [0, 1]
        x = MinMaxScaler(feature_range=(0, 1)).fit_transform(df)
        
        # Reshape for CNN
        x = x.reshape(x.shape[0], x.shape[1], 1)

        # Debug information
        st.write("### Preprocessing Debug Information")
        st.write(f"Input shape: {data.shape}")
        st.write(f"Output shape: {x.shape}")
        if y is not None:
            st.write(f"Labels shape: {y.shape}")
        st.write("Sample of preprocessed features:")
        st.write(pd.DataFrame(x.reshape(x.shape[0], -1)).head())

        return x, y

    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None

def plot_predictions(predictions, actual=None):
    """Plot prediction results"""
    pred_classes = predictions.argmax(axis=1)
    pred_probs = predictions.max(axis=1)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Distribution of predictions
    plt.subplot(2, 2, 1)
    sns.countplot(x=['Normal' if p == 0 else 'Anomaly' for p in pred_classes])
    plt.title('Distribution of Predictions')
    plt.ylabel('Count')
    
    # 2. Prediction probabilities
    plt.subplot(2, 2, 2)
    plt.hist(predictions[:, 1], bins=50)
    plt.title('Distribution of Anomaly Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    
    # 3. Additional plots if actual labels are available
    if actual is not None:
        plt.subplot(2, 2, 3)
        actual_classes = actual.argmax(axis=1)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(actual_classes, pred_classes)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.subplot(2, 2, 4)
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(actual_classes, predictions[:, 1])
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display metrics
    st.write("### Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        normal_count = (pred_classes == 0).sum()
        st.metric("Normal Traffic", normal_count)
    with col2:
        anomaly_count = (pred_classes == 1).sum()
        st.metric("Anomalous Traffic", anomaly_count)
    with col3:
        if actual is not None:
            accuracy = (pred_classes == actual_classes).mean()
            st.metric("Accuracy", f"{accuracy:.2%}")
        else:
            ratio = anomaly_count / len(pred_classes)
            st.metric("Anomaly Ratio", f"{ratio:.2%}")

def main():
    st.title("🔍 Network Traffic Anomaly Detection")
    st.write("""
    This application detects anomalies in network traffic using a trained CNN model.
    Upload your CSV file containing network traffic data to analyze it.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file, names=FEATURES)
        
        # Display sample
        with st.expander("View Data Sample"):
            st.write("### Raw Data Sample")
            st.write(df.head())
            st.write(f"Data shape: {df.shape}")
        
        # Load model
        model = load_model()
        if model is None:
            return
        
        # Preprocess data
        with st.spinner("Preprocessing data..."):
            X, y = preprocessing_anomaly(df)
            if X is None:
                return
        
        # Make predictions
        with st.spinner("Making predictions..."):
            predictions = model.predict(X)
        
        # Display results
        st.success("Analysis completed!")
        plot_predictions(predictions, y)
        
        # Prepare results for download
        results_df = df.copy()
        results_df['predicted_label'] = ['normal' if p == 0 else 'anomaly' 
                                       for p in predictions.argmax(axis=1)]
        results_df['anomaly_probability'] = predictions[:, 1]
        
        # Download button
        st.download_button(
            label="Download Results",
            data=results_df.to_csv(index=False),
            file_name="traffic_analysis_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()