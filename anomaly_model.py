import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

class LSTMAutoencoder(tf.keras.Model):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        # Encoder
        self.encoder = Sequential([
            LSTM(embedding_dim, activation='relu', 
                 input_shape=(seq_len, n_features)),
        ])
        
        # Decoder
        self.decoder = Sequential([
            RepeatVector(seq_len),
            LSTM(embedding_dim, activation='relu', 
                 return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def reconstruct(self, x):
        return self.predict(x)
