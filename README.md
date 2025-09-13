# REDES NEURONALES

**ESCUELA COLOMBIANA DE INGENIERÍA**

**PRINCIPIOS Y TECNOLOGÍAS IA 2025-2**

## Integrantes
- Andres Felipe Calderon Ramirez - [andrescalderonr](https://github.com/andrescalderonr)
- Santiago Botero Garcia - [LePeanutButter](https://github.com/LePeanutButter)

## PARTE I. IMPLEMENTACIÓN DE RED NEURONAL

Para obtener las derivadas de las funciones utilizadas en esta implementación, se utilizó [Wolfram Alpha](https://www.wolframalpha.com/), una herramienta computacional avanzada que facilita el cálculo simbólico y la verificación de expresiones matemáticas. Esto permitió asegurar la precisión en el desarrollo de las derivadas, garantizando resultados correctos para la implementación de la red neuronal.

### Derivada función Sigmoide:
Se utilizó el siguiente comando en Wolfram Alpha para calcular la derivada de la función sigmoide: `derivative of 1 / (1 + (e ^ (-z)))`

![](img/sigmoid-derivative.png)

Como resultado, se obtuvo que la derivada de la función

$\frac{d}{dz} \left( \frac{1}{1 + e^{-z}} \right) = \frac{e^{-z}}{(1 + e^{-z})^2}$

### Derivada función ReLU:

Para hacer la derivada lo que hicimos fue que en el programa Wolfram Alpha colocamos el comando `derivative of max(0,z)`, el resultado fue el siguiente:

![](img/relu.png)

### Derivada función de costo: Entropia Cruzada:

Para poder hacer la derivada de la función de Entropia Cruzada, entramos a WolframAplha y usamos el comando `- D[Sum[f[x] * Log[g[x]], {i, 1, n}], x]`

![](img/Entropy.png)


# Parte 2:

## Paso 1:

El problema es que debemos hacer una clasificación de clases, debemos predecir a que clase pertenece un vino segun sus caracteristicas quimicas utilizando el dataset de vinos, cave tener en cuenta que cada vino solo hace parte de una sola clase.

La metrica que se selecciono fue la de exactitud al permitir medir la cantidad de vinos que fueron clasificados de forma correcta sobre el total de muestras.

## Paso 2:

### Exploración del dataset:

![](img/chimical_elements.png)

![](img/wine_classes.png)

Tenemos un total de 3 clases diferentes de vinos y un total de 13 diferentes medidas de elementos quimicos para los vinos, tenemos un total de 178 vinos en el dataset y tampoco tenemos

![](img/correlation.png)

Como se puede ver en la imagen si hay correlación entre los diferentes elementos quimicos, un ejemplo es que total_phenols y flavanoids tienen una correlacion de 0.86, indicando que si hay alguno de los 2 elementos quimicos, el otro aumentara en proporciones parecidas.  Tambien hay correlaciones negativas como flavanoids y nonflavanoid_phenols con una correlacion de -0.44 indicando que si hay un elemento el otr disminuira. Por ultimo hay correlaciones nulas como puede ser magnesium y hue con una correlacion de 0.02 indicando que hay independencia entre ambos elementos.

### Desarrollando la red

Primer lo que se hizo es importar el dataset de vinos con :

```` python 
wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets
````
Luego lo que se hizo fue normalizar los features que son las 13 medidas quimicas con:

```` python
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
````

Luego codificamos las variables categoricas que son las 3 categorias de vinos
```` python
y_category = to_categorical(y.values.ravel()-1, num_classes=3)
````
Esto se hace principalmente porque keras recibe datos que empiezen desde 0 y se pasan a vectores binarios al ser un problema de clasificación de clases, lo estamos haciendo con categorical_crossentropy.

Despues dividimos train, test y validation para el modelo

```` python
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_category, test_size=0.2, random_state=42, stratify=y.values.ravel())

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
````

Luego iniciamos el modelo y colocamos 3 capas, cada una con una cantidad de neuronas diferentes, una activación y en la primera capa un tamaño de entrada.

```` python
model = keras.Sequential()
model.add(layers.Dense(16, activation="relu",input_shape=(X_train.shape[1],)))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(3,  activation="softmax"))
````

Luego compilamos el modelo, el optimizador es el algoritmo para reducir el margen de error, utilizamos adam al ser muy usado por dar buenos resultados. 

La función de perdida, esto mide el margen de error, al usar vectores binarios y claisficación de clases usamos categorical_crossentropy.

La metrica que es la medida para evaluar el rendimiento, utilizamos accuracy que mide la proporcion de predicciones logradas.

````  python
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
````
Luego entrenamos el modelo X_train son los datos quimicos de los vinos ya normalizado y Y_train son las tres clases en vectores binarios.

Validation data se encarga de evaluar datos que no halla visto.

Epoch es la cantidad de vese que la red pasa por todo el conjunto de entrenamiento.

El batch_size que es la cantidad de muestras que la red procesa.

```` python
training = model.fit( X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)
````
Ya por ultimo evaluamos el modelo.

Tenemos el test_loss que mide la perdida de la prueba, test_acc que mide la precision del modelo.

x_test y y_test son los datos de prueba

```` python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
````

## Conclusiones

Al probar el modelo se hicieron un total de 50 epochs, al inicioa el accuracy o precision inician de forma baja debido al inicio del aprendisaje del modelo,tras un rato la perdida va bajando mas ya que el modelo al ir aprendiendo mejor a como clasificar los vinos.


