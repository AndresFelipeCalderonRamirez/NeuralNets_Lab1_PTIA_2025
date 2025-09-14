
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from keras.src.utils import to_categorical

# Fetch the Wine dataset from UCI repository
wine = fetch_ucirepo(id=109)

# Extract features and targets from the dataset
X = wine.data.features
y = wine.data.targets

# Normalization of the feature data (scaling the features to have zero mean and unit variance)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# One-hot encoding of categorical target variable (y), adjusting for zero-indexing
y_category = to_categorical(y.values.ravel()-1, num_classes=3)


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_category, test_size=0.2, random_state=42, stratify=y.values.ravel())

# Split the training set into training and validation sets (85% train, 15% validation)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42
)

# Initialize the Keras Sequential model
model = keras.Sequential()

# Add a Dense layer with 16 units and 'relu' activation for the first hidden layer
model.add(layers.Dense(16, activation="relu",input_shape=(X_train.shape[1],)))

# Add another Dense layer with 8 units and 'relu' activation for the second hidden layer
model.add(layers.Dense(8, activation="relu"))

# Add the output layer with 3 units and 'softmax' activation for multi-class classification
model.add(layers.Dense(3,  activation="softmax"))

# Compile the model with the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

# Train the model using the training data and validate it using the validation set (50 epochs, batch size 16)
training = model.fit( X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)

# Evaluate the trained model on the test set to assess its performance
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

# Print the test loss and accuracy
print("Pérdida en test:", test_loss)
print("Precisión en test:", test_acc)


