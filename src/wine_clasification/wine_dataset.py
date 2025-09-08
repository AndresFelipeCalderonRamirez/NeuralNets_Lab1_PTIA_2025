from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns
# fetch dataset
wine = fetch_ucirepo(id=109)


# data (as pandas dataframes)
X = wine.data.features
y = wine.data.targets

# Informacion de las medidas quimicas de los vinos
print(X.info())

# Distribucion de las clases de vino
print(y.value_counts())
y.value_counts().plot(kind="bar", title="Distribución de clases")
plt.show()

# Verificación de datos nulos
print(X.isnull().values.any())
print(y.isnull().values.any())

# Matriz de correlación de los datos

corr_matrix = X.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matriz de correlación de variables")
plt.show()