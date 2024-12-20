from utils.preprocess import load_and_preprocess_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from utils.hyperparameters import THRESHOLD

def run_linear_regression(file_path):
    print("Running Linear Regression...")
    
    # Unpack all returned values, even if feature_names isn't used
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    accuracy = sum(abs(y_test - predictions) <= THRESHOLD) / len(y_test) * 100

    print("Linear Regression Results:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
    print(f"Accuracy (within ±2): {accuracy:.2f}%")
    
    return mse, r2, accuracy
