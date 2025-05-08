# Hybrid-ANN-PCA-MLP-Model-for-Accurate-and-Efficient-Credit-Card-Fraud-Detection


This project combines advanced machine learning techniques, such as **Principal Component Analysis (PCA)** and **Multi-Layer Perceptron (MLP)** models, to create a highly efficient and accurate **Credit Card Fraud Detection System**. The system uses PCA to reduce the dimensionality of the dataset, improving the performance and training time of the **MLP (Artificial Neural Network)** model, which then classifies transactions as either fraudulent or legitimate.

---

## 🚀 Overview

The goal of this project is to detect fraudulent credit card transactions in a dataset with highly imbalanced classes (fraudulent vs legitimate transactions). By leveraging PCA for dimensionality reduction and MLP for classification, the model is trained to accurately predict fraud with minimal computational cost.

---

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - **Scikit-learn**: For building and evaluating the machine learning models
  - **TensorFlow / Keras**: For implementing the Multi-Layer Perceptron (MLP) model
  - **Pandas**: For data manipulation and preprocessing
  - **NumPy**: For numerical operations
  - **Matplotlib / Seaborn**: For data visualization and analysis
  - **Imbalanced-learn**: For handling class imbalance issues (Optional)

---

## 📚 Dataset

This project uses the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle, which contains anonymized features of credit card transactions and a binary target variable indicating whether a transaction is fraudulent or not. The dataset includes:

- 284,807 transactions
- 31 features, with the first 30 being anonymized, and one column representing the transaction amount.
- **Class imbalance**: The dataset is highly imbalanced, with only about 0.17% fraudulent transactions.

---

## ⚙️ Approach

### 1. **Data Preprocessing**
   - Load and clean the dataset
   - Handle missing values (if any)
   - Scale the data using **StandardScaler** to normalize features for better model performance
   - **Split the data** into training and testing sets (80/20 split)

### 2. **Principal Component Analysis (PCA)**
   - Perform **PCA** on the features to reduce the dimensionality of the dataset
   - Choose the number of components that retain at least 95% of the original variance
   - PCA helps to simplify the model, reduce overfitting, and improve the training time

### 3. **Building the MLP Model**
   - Use **Multi-Layer Perceptron (MLP)**, a feedforward neural network model, for classification
   - The MLP consists of:
     - **Input layer**: Matches the number of PCA components
     - **Hidden layers**: Multiple layers with **ReLU** activation function
     - **Output layer**: A single unit with **sigmoid** activation for binary classification (fraud or not)

### 4. **Model Training & Evaluation**
   - Train the MLP model using the training data and evaluate its performance on the testing set
   - Utilize **Accuracy**, **Precision**, **Recall**, and **F1-Score** as metrics for evaluation, especially considering the class imbalance
   - Implement techniques like **Early Stopping** to prevent overfitting

### 5. **Handling Class Imbalance (Optional)**
   - Use methods like **SMOTE (Synthetic Minority Over-sampling Technique)** or **Class Weights** in the MLP to address the class imbalance issue and improve model sensitivity towards fraudulent transactions

---

## ✨ Features

- **Dimensionality Reduction** using PCA to improve model efficiency and reduce overfitting
- **Accurate Fraud Detection** using an MLP model for binary classification
- **Class Imbalance Handling** with techniques like SMOTE (optional) to enhance model performance
- **Performance Metrics** like accuracy, precision, recall, and F1-score for model evaluation
- **Visualizations** for model performance (e.g., confusion matrix, ROC curve)

---

## 📁 Project Structure

```plaintext
credit-card-fraud-detection/
├── data/                       # Contains the raw and preprocessed datasets
├── models/                      # Contains saved model files (e.g., MLP model)
├── notebooks/                   # Jupyter notebooks for data exploration and analysis
├── src/                         # Source code for preprocessing, PCA, MLP model, etc.
│   ├── preprocessing.py         # Data cleaning and preprocessing steps
│   ├── pca.py                   # PCA implementation for dimensionality reduction
│   ├── mlp_model.py             # MLP model definition, training, and evaluation
│   ├── utils.py                 # Utility functions (e.g., data splitting)
│   └── evaluate.py              # Model evaluation and metrics calculation
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
