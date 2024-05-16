# Import the necessary libraries
import numpy as np                              # For numerical computations
import pandas as pd                             # For handling datasets
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
from sklearn.feature_extraction.text import CountVectorizer # For converting text to numerical features
import scipy.sparse                             # For working with sparse matrices
from keras.models import Sequential             # For creating a neural network model
from keras.layers import Dense                  # For adding layers to the neural network
from sklearn.metrics import classification_report
import joblib                                   # For saving the model

# Load the dataset
dataset = pd.read_csv('fraud_email_.csv')

# Replace NaN values with empty strings
dataset = dataset.fillna('')
X = dataset['Text'].values                       # Extract the text from the dataset
y = dataset['Class'].values                      # Extract the target variable (label) from the dataset

# Convert the text to numerical features using bag-of-words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Total number of samples
n_samples = X.shape[0]
n_train = int(n_samples * 0.8) # Use 80% of samples for training

# Create the sparse matrix for the training and testing data
X_train = scipy.sparse.csr_matrix(X[:n_train, :])
X_test = scipy.sparse.csr_matrix(X[n_train:, :])

# Sort the indices of the sparse matrix
X_train.sort_indices()
X_test.sort_indices()

# Split the dataset into training and testing sets, with 80% for training and 20% for testing
y_train = y[:n_train]
y_test = y[n_train:]

# Create the neural network model
model = Sequential()                            # Create an instance of the sequential model
model.add(Dense(units=16, activation='relu', input_dim=X_train.shape[1])) # Add a dense layer with 16 units and ReLU activation function as the input layer
model.add(Dense(units=8, activation='relu'))    # Add a dense layer with 8 units and ReLU activation function
model.add(Dense(units=1, activation='sigmoid')) # Add a dense layer with 1 unit and sigmoid activation function as the output layer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile the model with the Adam optimizer, binary cross-entropy loss function, and accuracy metric

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test)) # Train the model on the training data with a batch size of 32, 100 epochs, and validation data for evaluating the model's performance during training

# Save the model
joblib.dump(model, 'model.pkl')