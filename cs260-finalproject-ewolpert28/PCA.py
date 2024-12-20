import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from utils.preprocess import load_and_preprocess_data

def calculate_feature_importance_logistic(file_path):
    print("Calculating Feature Importance using Logistic Regression...")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)

    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)

    # Extract feature coefficients
    coefficients = model.coef_[0]

    # Create a DataFrame for coefficient-based importance
    coefficient_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values(by='Abs_Coefficient', ascending=False)

    print("\nFeature Importance from Coefficients:")
    print(coefficient_importance)

    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    permutation_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importance from Permutation:")
    print(permutation_importance_df)

    # Combine coefficient and permutation importance for comparison
    combined_importance = pd.merge(
        coefficient_importance,
        permutation_importance_df,
        on='Feature',
        how='outer'
    ).sort_values(by='Abs_Coefficient', ascending=False)

    print("\nCombined Feature Importance:")
    print(combined_importance)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(combined_importance['Feature'], combined_importance['Abs_Coefficient'], color='skyblue', label='Coefficient Importance')
    plt.barh(combined_importance['Feature'], combined_importance['Importance'], color='orange', alpha=0.7, label='Permutation Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance from Logistic Regression Coefficients and Permutation')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.legend()
    plt.tight_layout()
    plt.show()

    return combined_importance

if __name__ == "__main__":
    file_path = 'data/StudentPerformanceFactors.csv'
    feature_importance = calculate_feature_importance_logistic(file_path)
