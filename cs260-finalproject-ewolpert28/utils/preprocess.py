import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(file_path):
    print("Loading data from:", file_path)
    
    # Load dataset
    try:
        data = pd.read_csv(file_path, sep=',')
    except FileNotFoundError as e:
        print(f"File not found: {file_path}")
        raise e

    print("Dataset preview:\n", data.head())

    # Validate columns
    required_columns = [
        'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
        'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
        'Internet_Access', 'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Exam_Score'
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"The following required columns are missing: {missing_columns}")

    # Drop Gender column
    if 'Gender' in data.columns:
        print("Dropping the Gender column...")
        data = data.drop(columns=['Gender'])

    # Separate features and target
    if 'Exam_Score' not in data.columns:
        raise KeyError("Target column 'Exam_Score' is missing from the dataset.")
    
    X = data.drop(columns=['Exam_Score'])  # Features
    y = data['Exam_Score']  # Target variable

    # Identify feature types
    numerical_features = [
        'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity'
    ]
    categorical_features = [
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level',
        'Distance_from_Home'
    ]

    # Encode and scale features
    ct = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),  # Scale numerical features
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)  # One-hot encode categorical
    ], remainder='drop')

    X_transformed = ct.fit_transform(X)

    # Combine feature names for transformed features
    feature_names = numerical_features + list(
        ct.named_transformers_['cat'].get_feature_names_out(categorical_features)
    )

    print("Transformed feature matrix (X) preview:\n", X_transformed[:5])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    print(f"Train set size: {X_train.shape}, {y_train.shape}")
    print(f"Test set size: {X_test.shape}, {y_test.shape}")

    return X_train, X_test, y_train, y_test, feature_names

if __name__ == "__main__":
    file_path = 'data/StudentPerformanceFactors.csv'
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(file_path)
