import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from models.anomaly_model import LSTMAutoencoder
from src.data_preprocessing import DataPreprocessor

class AnomalyDetector:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.iso_forest = IsolationForest(contamination=0.05)
    
    def detect(self, X):
        # Get reconstruction errors
        reconstructions = self.model.reconstruct(X)
        mse = np.mean(np.square(X - reconstructions), axis=(1, 2))
        
        # Train Isolation Forest on errors
        self.iso_forest.fit(mse.reshape(-1, 1))
        scores = -self.iso_forest.score_samples(mse.reshape(-1, 1))
        
        return scores
    
    def optimize_threshold(self, scores, y_true):
        thresholds = np.linspace(min(scores), max(scores), 100)
        best_f1 = 0
        best_thresh = 0
        
        for thresh in thresholds:
            preds = (scores > thresh).astype(int)
            current_f1 = f1_score(y_true, preds)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_thresh = thresh
                
        return best_thresh
