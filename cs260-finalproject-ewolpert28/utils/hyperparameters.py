# Hyperparameters for all models

# Shared hyperparameters
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
PATIENCE = 10  # For EarlyStopping
THRESHOLD = 2  # For accuracy calculation

# Neural Network specific
L2_REGULARIZATION = 0.01
DROPOUT_RATE = .1

# SVM specific
SVM_C = 10
SVM_KERNEL = 'rbf'
SVM_GAMMA = 0.1

# Linear Regression specific
# Add any specific settings for Linear Regression here
