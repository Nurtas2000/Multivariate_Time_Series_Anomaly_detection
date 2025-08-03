import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self):
        df = pd.read_csv(self.config['data_path'], 
                        parse_dates=['timestamp'])
        return df
    
    def create_sequences(self, data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    
    def preprocess(self):
        # Load and prepare data
        df = self.load_data()
        features = df[self.config['features']]
        labels = df[self.config['target']]
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X_seq = self.create_sequences(X_scaled, self.config['seq_length'])
        y_seq = labels[self.config['seq_length']-1:]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state'],
            stratify=y_seq
        )
        
        return X_train, X_test, y_train, y_test, self.scaler
