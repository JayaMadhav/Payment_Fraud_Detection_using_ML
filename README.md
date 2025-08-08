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

### 📈 Model Performance

We trained and evaluated multiple machine learning models for fraud detection. Below are the performance metrics based on classification scores and evaluation datasets.

#### ✅ Random Forest (Best Performing Model)

* **Best Parameters**:
  `max_depth=None`, `min_samples_split=7`, `n_estimators=141`
* **Accuracy**: `0.9975`
* **Precision**: `0.9974`
* **Recall**: `0.9975`
* **F1 Score**: `0.9974`

#### 🔹 Logistic Regression

* **Best Parameters**:
  `C=10.0435`, `penalty='l2'`
* **Accuracy**: `0.9951`
* **Precision**: `0.9951`
* **Recall**: `0.9951`
* **F1 Score**: `0.9932`

#### 🔹 Support Vector Machine (SVM)

* **Best Parameters**:
  `C=9.9050`, `kernel='linear'`
* **Accuracy**: `0.9951`
* **Precision**: `0.9951`
* **Recall**: `0.9951`
* **F1 Score**: `0.9932`

---

### 🧪 Final Evaluation on Test Set

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| **Random Forest**   | 0.9982   | 0.9231    | 0.7500 | 0.8276   |
| Logistic Regression | 0.9951   | 1.0000    | 0.1250 | 0.2222   |
| SVM                 | 0.9944   | 0.0000    | 0.0000 | 0.0000   |
| Decision Tree       | 0.9979   | 0.8125    | 0.8125 | 0.8125   |

---

### 🔍 Additional Insights

* **PCA (Dimensionality Reduction)**

  * Explained Variance Ratio: `60.56%`

* **Clustering (Silhouette Score)**

  * Silhouette Score: `0.8041` (indicates good clustering structure)

* **Imbalanced Class Support**:

  ```
  Class 0: Support = 2833 (legitimate)
  Class 1: Support = 16 (fraudulent)
  Macro F1 Score: 0.61 | Weighted F1 Score: 0.99
  ```

---

### 🏆 Summary

> **Random Forest** achieved the best performance in terms of **F1-score** and is selected as the final model for deployment.


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
