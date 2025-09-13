
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras import layers
from keras.src.utils import to_categorical

wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets

#Normalizar los datos
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Codificacion de variables categoricas
y_category = to_categorical(y.values.ravel()-1, num_classes=3)


# Separar train y test
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_category, test_size=0.2, random_state=42, stratify=y.values.ravel())

# Separar validación del train
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42
)

#Inicio modelo Keras, tiene units que son la cantidad de neuronas y una actiavacion
model = keras.Sequential()
model.add(layers.Dense(16, activation="relu",input_shape=(X_train.shape[1],)))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(3,  activation="softmax"))

#Compilar el modelo
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

#Entrenar el modelo
training = model.fit( X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)

#Evaluar el modelo
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print("Pérdida en test:", test_loss)
print("Precisión en test:", test_acc)


