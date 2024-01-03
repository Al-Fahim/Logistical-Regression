import os
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer


# Function to tokenize text by transforming to lowercase and removing punctuation
def tokenize(text):
    text = text.lower()  # Convert to lowercase
    return text.translate(str.maketrans('', '', string.punctuation)).split()  # Remove punctuation and split to tokens


# Function to preprocess data by reading files from a directory and tokenizing the text
def preprocess(directory):
    files = []
    for filename in os.listdir(directory):  # Iterate over each file in the directory
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:  # Open each file
                text = file.read()  # Read the text from file
                files.append(' '.join(tokenize(text)))  # Tokenize the text and add to data
    return files


# Function to extract features from the data using bag of words representation

vectorizer = CountVectorizer(max_features=1000, stop_words='english')  # Initialize a CountVectorizer


def extract_features(files):
    X = vectorizer.transform(files)  # Fit the vectorizer to the data and transform the data
    return X.toarray()  # Return the feature


# ---------------------------------------------------------------------
# Training Model
print("Begining Training")

# Process the positive and negative files
positive_files = preprocess('train/pos')
negative_files = preprocess('train/neg')

# Assign labels to the processed files
positive_labels = [1] * len(positive_files)
negative_labels = [0] * len(negative_files)

# Combine the processed files and their labels
file_train = positive_files + negative_files
label_train = positive_labels + negative_labels

# Fit the vectorizer on the training data
vectorizer.fit(file_train)

# Store the values to save
X_train = extract_features(file_train)
Y_train = np.array(label_train)

print("Training Ended")
# ---------------------------------------------------------------------
# Testing Model

print("Begining Testing")
# Process the positive and negative files
positive_testfile = preprocess('test/pos')
negative_testfile = preprocess('test/neg')

# Assign labels to the processed files
positive_testlabels = [1] * len(positive_testfile)
negative_testlabels = [0] * len(negative_testfile)

# Combine the processed files and their labels
file_test = positive_testfile + negative_testfile
label_test = positive_testlabels + negative_testlabels

# Store the values
X_test = extract_features(file_test)
Y_test = np.array(label_test)

print("Testing Ended")
# Save your findings

print("Creating Saves")

np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)
np.save('X_test.npy', X_test)
np.save('Y_test.npy', Y_test)

print("Saves Made")
