from models.linear_regression import run_linear_regression
from models.svm import run_svm
from models.neural_network import run_neural_network


def compare_models(file_path):
    print("Evaluating Linear Regression...")
    lr_mse, lr_r2, lr_accuracy = run_linear_regression(file_path)

    print("Evaluating SVM...")
    svm_mse, svm_r2, svm_accuracy = run_svm(file_path)

    print("Evaluating Neural Network...")
    nn_mse, nn_r2, nn_accuracy = run_neural_network(file_path)

    print("\nComparison of Models:")
    print(f"Linear Regression: MSE={lr_mse:.2f}, R2={lr_r2:.2f}, Accuracy (±1): {lr_accuracy:.2f}%")
    print(f"SVM: MSE={svm_mse:.2f}, R2={svm_r2:.2f}, Accuracy (±1): {svm_accuracy:.2f}%")
    print(f"Neural Network: MSE={nn_mse:.2f}, R2={nn_r2:.2f}, Accuracy (±1): {nn_accuracy:.2f}%")


if __name__ == "__main__":
    compare_models('data/StudentPerformanceFactors.csv')
