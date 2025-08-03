import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.anomaly_model import LSTMAutoencoder
from src.data_preprocessing import DataPreprocessor
from src.utils import setup_logging

def train(config_path='configs/config.yaml'):
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logging()
    
    # Preprocess data
    preprocessor = DataPreprocessor(config)
    X_train, X_test, y_train, y_test, scaler = preprocessor.preprocess()
    
    # Initialize model
    model = LSTMAutoencoder(
        seq_len=config['seq_length'],
        n_features=len(config['features']),
        embedding_dim=config['embedding_dim']
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
                  loss='mse')
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=config['patience'], restore_best_weights=True),
        ModelCheckpoint(
            filepath=config['model_save_path'],
            save_best_only=True,
            monitor='val_loss'
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training completed successfully")
    return model, history, scaler

if __name__ == "__main__":
    train()
