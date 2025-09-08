import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical





wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets

#Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Codificacion de variables categoricas
y_categorical = to_categorical(y.values.ravel()-1, num_classes=3)


# Separar train y test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y.values.ravel())

# Separar validación del train
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42
)

print("\nTamaños de los conjuntos:")
print("Entrenamiento:", X_train.shape)
print("Validación:", X_val.shape)
print("Prueba:", X_test.shape)