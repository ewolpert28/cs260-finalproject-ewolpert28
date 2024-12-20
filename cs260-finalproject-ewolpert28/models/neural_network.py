from utils.preprocess import load_and_preprocess_data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import mean_squared_error, r2_score
from utils.hyperparameters import THRESHOLD
def run_neural_network(file_path):
    print("Running Neural Network...")
    
    # Unpack all returned values
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)
    
    # Define learning rate schedule
    def lr_schedule(epoch, lr):
        return lr * 0.5 if epoch > 30 else lr

    # Build Neural Network
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, lr_scheduler], verbose=1)

    # Predict and evaluate
    predictions = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    accuracy = sum(abs(y_test - predictions) <= 2) / len(y_test) * 100

    print("Neural Network Results:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print(f"Accuracy (within ±2): {accuracy:.2f}%")
    
    return mse, r2, accuracy
