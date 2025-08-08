# 🔒 Payment Fraud Detection using Machine Learning

This project is a machine learning-based web application that detects fraudulent transactions in real-time using a trained model. It provides a user-friendly dashboard built with Streamlit and supports both single and batch transaction predictions.

---

## 🚀 Features

- 🔮 Real-time single transaction fraud prediction
- 📂 Batch prediction via CSV uploads
- 🗂️ Prediction history tracking and export
- 📊 AI-powered fraud classification
- ✅ Clean, interactive Streamlit interface

---

## 🧠 Model

The model was trained using TensorFlow/Keras on transaction data to classify whether a given transaction is fraudulent. It uses key features such as:

- `Transaction Type`
- `Transaction Amount`
- `Old Balance (Origin)`
- `New Balance (Origin)`

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit 🖥️
- TensorFlow/Keras 🤖
- Pandas & NumPy 📊

---

## 📁 Project Structure

```

Payment\_Fraud\_Detection\_using\_ML/
│
├── app.py                          # Streamlit dashboard app
├── Training.ipynb                  # Jupyter notebook for training the model
├── payment\_fraud\_detection\_model.h5  # Trained Keras model (place this in root)
├── README.md                       # Project documentation

````

---

## ▶️ Getting Started

### ✅ Prerequisites

Make sure you have Python 3.8+ installed.

Install required packages:

```bash
pip install -r requirements.txt
````

`requirements.txt`:

```
streamlit
tensorflow
pandas
numpy
```

### ⚙️ Running the App

```bash
streamlit run app.py
```

> Make sure the model file `payment_fraud_detection_model.h5` is in the same directory as `app.py`.

---

## 📌 Usage

### 🔹 Single Prediction

* Choose transaction type
* Enter amount and balances
* Click **Predict**
* View the result as **Fraudulent** or **Legitimate**

### 🔹 Batch Prediction

* Upload a `.csv` with the following headers:

  ```
  type,Amount,Oldbalance Org,Newbalance Orig
  ```
* Get results instantly
* Download the prediction file

### 🔹 History

* All predictions (single + batch) are saved in session
* Export full prediction history as CSV

---

## 🧾 Sample CSV Format

```csv
type,Amount,Oldbalance Org,Newbalance Orig
Transfer,1200.50,5000.00,3800.00
Cash-Out,2000.00,3000.00,1000.00
```

---




- A badge for accuracy or model performance.
```
