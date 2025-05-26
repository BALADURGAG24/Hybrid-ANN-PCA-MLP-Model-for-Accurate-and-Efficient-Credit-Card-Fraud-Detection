import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

def load_and_preprocess_data(filepath, n_components=20, save_dir="models"):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Separate features and labels
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Save the PCA transformer
    joblib.dump(pca, os.path.join(save_dir, "pca_transformer.pkl"))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
