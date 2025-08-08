import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('payment_fraud_detection_model.h5')

# Initialize prediction history
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=["type", "Amount", "Oldbalance Org", 
                                                        "Newbalance Orig", "Fraud Prediction"])

# Helper function to make predictions
def predict_fraud(data):
    prediction = model.predict(data)
    return prediction[0][0]

# Streamlit App Layout and Pages
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS to center align all content and tabs
    st.markdown(
        """
        <style>
        /* Center all main content */
        .main-header, .sub-header, .stText, .stButton, .stDownloadButton, .stSelectbox, .stNumberInput, .stFileUploader, .stImage, .stExpander {
            display: flex;
            justify-content: center;
            text-align: center;
            font-weight: bold;
        }
        /* Center and style the main headings */
        .main-header {
            font-size: 3rem;
            color: #FF4B4B;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #4BFF4B;
        }
        /* Style each page tab */
        .css-1e5imcs {
            display: flex;
            justify-content: center;
            font-size: 1.25rem;
            font-weight: bold;
            color: #FF4B4B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create centered navigation tabs
    tabs = st.tabs(["🏠 Home", "📊 Single Prediction", "📂 Batch Prediction", "🗂️ Prediction History", "ℹ️ About"])

    # Home Page
    with tabs[0]:
        st.markdown('<div class="main-header">Welcome to the Fraud Detection System 🔒</div>', unsafe_allow_html=True)
        st.write("Our system ensures your transactions remain secure by detecting fraudulent activities in real time.")

        try:
            st.image("C:/Users/chara/Downloads/Payment Fraud Detection/p2.jpeg", 
                     caption="Fraud Detection System", use_column_width=True)
        except FileNotFoundError:
            st.error("⚠ The specified image file could not be found. Please ensure the path is correct.")

        st.markdown("<div class='sub-header'>🌟 Key Features</div>", unsafe_allow_html=True)
        with st.expander("🔑 Why Choose Us?"):
            st.markdown(
                """
                - **Real-Time Detection**: Instantly analyze transactions to detect potential fraud.
                - **AI-Powered Model**: Leveraging machine learning for accurate fraud prediction.
                - **User-Friendly Interface**: Simple, interactive interface for individual and batch predictions.
                - **Historical Insights**: Review past predictions and export them as needed.
                """
            )

    # Single Prediction Page
    with tabs[1]:
        st.markdown('<div class="main-header">Single Transaction Prediction 📊</div>', unsafe_allow_html=True)
        st.write("Enter the transaction details below to predict its status.")

        # Input fields for transaction details
        col1, col2 = st.columns(2)

        with col1:
            transaction_type = st.selectbox("Transaction Type", ["Cash-Out", "Payment", "Transfer", "Cash-In", "Debit"])
            amount = st.number_input("Transaction Amount (₹)", min_value=0.0, step=0.01)

        with col2:
            oldbalance_org = st.number_input("Old Balance Original (₹)", min_value=0.0, step=0.01)
            newbalance_orig = st.number_input("New Balance Original (₹)", min_value=0.0, step=0.01)

        if st.button("🔮 Predict"):
            # Preprocess and predict
            data = pd.DataFrame(
                [[transaction_type, amount, oldbalance_org, newbalance_orig]],
                columns=["type", "Amount", "Oldbalance Org", "Newbalance Orig"]
            )
            data = pd.get_dummies(data, columns=["type"], drop_first=True)

            model_input = np.zeros((1, model.input_shape[1]))
            model_input[0, :data.shape[1]] = data.values
            prediction = predict_fraud(model_input)

            # Display result
            result = "Fraudulent" if prediction >= 0.5 else "Legitimate"
            if prediction >= 0.5:
                st.error(f"⚠ **The transaction is predicted to be {result}.**")
            else:
                st.success(f"✅ **The transaction is predicted to be {result}.**")

            # Save to history
            data["Fraud Prediction"] = result
            st.session_state['history'] = pd.concat([st.session_state['history'], data], ignore_index=True)

    # Batch Prediction Page
    with tabs[2]:
        st.markdown('<div class="main-header">Batch Transaction Prediction 📂</div>', unsafe_allow_html=True)
        st.write("Upload a CSV file containing transaction details to predict fraud in bulk.")

        uploaded_file = st.file_uploader("📤 Upload a CSV File", type="csv")
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            if "type" not in batch_data.columns:
                st.error("⚠ The uploaded file must contain a 'type' column.")
            else:
                # Preprocess and predict
                batch_data = pd.get_dummies(batch_data, columns=["type"], drop_first=True)
                model_input = np.zeros((batch_data.shape[0], model.input_shape[1]))
                model_input[:, :batch_data.shape[1]] = batch_data.values[:, :model.input_shape[1]]
                predictions = model.predict(model_input)

                batch_data["Fraud Prediction"] = ["Fraudulent" if pred >= 0.5 else "Legitimate" for pred in predictions]

                # Save to history
                st.session_state['history'] = pd.concat([st.session_state['history'], batch_data], ignore_index=True)

                # Display results
                st.write(batch_data)
                csv = batch_data.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions as CSV", csv, "batch_predictions.csv", "text/csv")

    # Prediction History Page
    with tabs[3]:
        st.markdown('<div class="main-header">Prediction History 🗂️</div>', unsafe_allow_html=True)
        if st.session_state['history'].empty:
            st.info("No predictions have been made yet.")
        else:
            st.write("Here is the history of all predictions made so far:")
            st.write(st.session_state['history'])

            # Download history
            csv = st.session_state['history'].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download History as CSV", csv, "prediction_history.csv", "text/csv")

    # About Page
    with tabs[4]:
        st.markdown('<div class="main-header">About Us ℹ</div>', unsafe_allow_html=True)
        st.write("This fraud detection system ensures secure online transactions using state-of-the-art machine learning.")

# Run the app
if __name__ == "__main__":
    main()
