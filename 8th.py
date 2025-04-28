import os
import numpy as np
import joblib
import streamlit as st
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- Image Processing Function ---
def extract_features_from_path(img_path):
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')  # Ensure image is in RGB format
        img = img.resize((64, 64))  # Resize the image to 64x64 pixels

        # Convert image to numpy array
        img_array = np.array(img)

        # Normalize the pixel values to be between 0 and 1
        img_array = img_array / 255.0

        # Flatten the image to use the pixel values as features
        features = img_array.flatten()

        return features
    except Exception as e:
        print(f"Error opening image file {img_path}: {e}")
        return None

# --- Model Prediction Function ---
def predict(features, model, scaler):
    # Scale the features
    features_scaled = scaler.transform([features])
    # Predict using the trained model
    prob = model.predict_proba(features_scaled)[0]
    label = model.predict(features_scaled)[0]
    
    return label, prob

# --- Performance Metrics Function ---
def plot_performance_metrics():
    # Example of performance metrics for multiple models
    models = ["Model 1", "Model 2", "Model 3"]
    
    # Example data (replace with your actual model evaluations)
    accuracy = [0.90, 0.85, 0.88]
    precision = [0.91, 0.86, 0.87]
    recall = [0.88, 0.84, 0.85]
    
    # Accuracy vs Model (Bar Graph)
    st.write("### Accuracy vs Model")
    accuracy_comparison = pd.DataFrame({
        "Model": models,
        "Accuracy": accuracy
    })
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=accuracy_comparison, ax=ax)
    st.pyplot(fig)

    # Precision and Recall Comparison (Grouped Bar Graph)
    st.write("### Precision and Recall Comparison")
    precision_recall_comparison = pd.DataFrame({
        "Model": models,
        "Precision": precision,
        "Recall": recall
    })
    fig2, ax2 = plt.subplots()
    precision_recall_comparison.set_index('Model').plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    # Confusion Matrix Heatmap (for each model)
    st.write("### Confusion Matrix Heatmap (for each model)")
    y_true = [0, 1, 0, 1, 0, 1]  # Actual labels
    y_pred = [0, 1, 0, 0, 0, 1]  # Predicted labels (Dummy for now)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, columns=["Predicted Happy", "Predicted Depressed"], index=["Actual Happy", "Actual Depressed"])
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues", ax=ax3)
    st.pyplot(fig3)

    # ROC Curve (Optional)
    st.write("### ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    fig4, ax4 = plt.subplots()
    ax4.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax4.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('Receiver Operating Characteristic')
    ax4.legend(loc="lower right")
    st.pyplot(fig4)

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Depression Detection App", layout="wide")
    st.title("🧠 Depression Detection App")

    # Navigation Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Home", "Analytics"])

    # Load model and scaler
    model = joblib.load('depression_model.pkl')
    scaler = joblib.load('scaler.pkl')

    if page == "Home":
        st.subheader("Upload an Image for Mood Detection")
        
        uploaded_file = st.file_uploader("Drag and drop your image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Load image
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True)

            # Extract features from the uploaded image
            features = extract_features_from_path(uploaded_file)

            if features is not None:
                # Make prediction
                label, prob = predict(features, model, scaler)
                
                mood = "Happy" if label == 0 else "Depressed"
                st.write(f"Detected Mood: {mood}")
                st.write(f"Depression Probability: {prob[1] * 100:.2f}%")
            else:
                st.error("Error processing the image.")
    
    elif page == "Analytics":
        plot_performance_metrics()

if __name__ == "__main__":
    main()
