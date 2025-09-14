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


# PARTE 2. IMPLEMENTACIÓN DEL FRAMEWORK KERAS

## 1. Descripción del Proyecto

El objetivo de este proyecto es realizar una clasificación de vinos en función de sus características químicas utilizando el dataset de vinos. Cada vino pertenece a una única clase, y el desafío consiste en predecir correctamente a qué clase corresponde un vino basándose en sus propiedades químicas.

La métrica seleccionada para evaluar el modelo es la **exactitud (accuracy)**, que mide la proporción de vinos correctamente clasificados respecto al total de muestras.

## 2. Exploración del Dataset

El dataset de vinos contiene información sobre tres clases diferentes de vino y 13 características químicas para cada vino. En total, el conjunto de datos cuenta con 178 muestras.

Las clases corresponden a tipos de vino, y las características incluyen mediciones de componentes químicos en el vino. A continuación se presentan algunos gráficos clave que ayudan a entender la distribución y correlación de las características.

### Gráficos descriptivos

![](img/chimical_elements.png)

![](img/wine_classes.png)

### Correlación entre características

![](img/correlation.png)

En la matriz de correlación, se observa que algunas características tienen una fuerte correlación positiva, como total_phenols y flavonoids (correlación de 0.86), lo que indica que si uno de estos componentes aumenta, el otro tiende a aumentar también. Por otro lado, también existen correlaciones negativas, como entre flavonoids y nonflavanoid_phenols (correlación de -0.44), lo que sugiere que si uno de estos componentes aumenta, el otro disminuirá. Además, algunos elementos muestran correlaciones cercanas a cero, como magnesium y hue (correlación de 0.02), lo que indica independencia entre estas características.

## 3. Desarrollo del Modelo

### 3.1. Carga y Preprocesamiento del Dataset

Se comienza importando el dataset de vinos utilizando la función `fetch_ucirepo`:

```` python 
wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets
````
A continuación, normalizamos las características del conjunto de datos para asegurar que todas las medidas químicas estén en la misma escala, utilizando `StandardScaler`:

```` python
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
````

Luego, las etiquetas de clase se codifican en vectores binarios utilizando `to_categorical`:

```` python
y_category = to_categorical(y.values.ravel()-1, num_classes=3)
````

Este paso es necesario, ya que Keras requiere que las etiquetas sean codificadas como vectores binarios en problemas de clasificación de múltiples clases.

### 3.2. División del Conjunto de Datos

El conjunto de datos se divide en tres subconjuntos: entrenamiento, validación y prueba, utilizando la función `train_test_split`:

```` python
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_category, test_size=0.2, random_state=42, stratify=y.values.ravel())

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
````

### 3.3. Diseño de la Red Neuronal

Se define una red neuronal de tipo secuencial con tres capas densas:

```` python
model = keras.Sequential()
model.add(layers.Dense(16, activation="relu",input_shape=(X_train.shape[1],)))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(3,  activation="softmax"))
````
- **Capa de entrada:** 16 neuronas con activación `ReLU`.
- **Capa oculta:** 8 neuronas con activación `ReLU`.
- **Capa de salida:** 3 neuronas con activación `softmax` para la clasificación de tres clases.

### 3.4. Compilación del Modelo

El modelo se compila utilizando el optimizador Adam (ampliamente utilizado por su rendimiento y eficiencia), la función de pérdida categorical_crossentropy (adecuada para problemas de clasificación multiclase), y la métrica accuracy (para medir la precisión del modelo):

````  python
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
````

### 3.5. Entrenamiento del Modelo

El modelo se entrena con los datos de entrenamiento durante 50 épocas, utilizando un tamaño de lote de 16 muestras y validando el rendimiento con el conjunto de validación:

```` python
training = model.fit( X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)
````

### 3.6. Evaluación del Modelo

Finalmente, el modelo se evalúa utilizando el conjunto de prueba:

```` python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
````

- **test_loss:** Pérdida en el conjunto de prueba.
- **test_acc:** Precisión en el conjunto de prueba.

## 4. Conclusiones

Después de entrenar el modelo durante 50 épocas, se observó que la precisión (accuracy) inicial era baja, lo cual es esperado en las primeras etapas del aprendizaje. Con el tiempo, a medida que el modelo mejora su capacidad para clasificar correctamente los vinos, la pérdida disminuye y la precisión aumenta. Este comportamiento es característico de los procesos de entrenamiento de redes neuronales, donde el modelo va ajustando sus parámetros para mejorar su desempeño en el conjunto de datos de prueba.

Este enfoque proporciona una solución efectiva para la clasificación de vinos basándose en sus características químicas, y los resultados obtenidos muestran un desempeño prometedor para este tipo de problemas de clasificación multiclase.

## RETROSPECTIVA

### **1. ¿Cuál fue el tiempo total invertido en el laboratorio por cada uno de ustedes? (Horas/Hombre)**

El tiempo total invertido en el laboratorio fue de aproximadamente **6 horas por persona**. Los miembros del equipo fueron **Andrés Felipe Calderón** y **Santiago Botero**, quienes colaboraron en la implementación y resolución de los distintos aspectos del proyecto.

---

### **2. ¿Cuál es el estado actual del laboratorio? ¿Por qué?**

El estado actual del laboratorio es **completo**, con una implementación funcional del modelo de red neuronal utilizando Keras. Sin embargo, consideramos que aún hay **posibilidad de mejora** en términos de comprensión y optimización de ciertos conceptos, como el **gradiente** y el funcionamiento general de la **red neuronal** como un sistema completo, en lugar de solo centrarse en las neuronas individuales y su conectividad.

---

### **3. ¿Cuál consideran fue el mayor logro? ¿Por qué?**

El mayor logro fue la **implementación exitosa de la capa densa** en el modelo de red neuronal. Esta fue una parte fundamental del proceso, ya que permitió estructurar adecuadamente las conexiones entre las neuronas y mejorar la capacidad del modelo para aprender patrones complejos en los datos. La adición de esta capa marcó un hito en el progreso del proyecto, ya que nos permitió avanzar hacia un modelo funcional con una arquitectura adecuada para el problema de clasificación multiclase.

---

### **4. ¿Cuál consideran que fue el mayor problema técnico? ¿Qué hicieron para resolverlo?**

El mayor problema técnico fue **entender el concepto de gradiente** y cómo se aplica en el contexto de las redes neuronales, especialmente en la capa densa. El gradiente es esencial para el algoritmo de optimización, y nuestra comprensión inicial fue limitada. Para resolver esto, recurrimos a diversas fuentes de documentación y tutoriales, y realizamos experimentos prácticos para observar cómo el gradiente afectaba la actualización de los pesos de la red durante el entrenamiento. Además, profundizamos en la teoría detrás del **descenso de gradiente** para entender mejor su aplicación en el ajuste de los parámetros del modelo.

---

### **5. ¿Qué hicieron bien como equipo? ¿Qué se comprometen a hacer para mejorar los resultados?**

Como equipo, nos **distribuimos bien las tareas**, gestionamos adecuadamente el control de versiones mediante **GitHub**, y usamos **ramas que siguen el estándar de desarrollo**, asegurando que el trabajo fuera colaborativo y sin conflictos. Además, nos comprometimos a **retroalimentarnos constantemente**, manteniendo una comunicación clara sobre avances, dudas y problemas.

Para mejorar los resultados en el futuro, nos comprometemos a **profundizar más en los conceptos teóricos** de las redes neuronales y el descenso de gradiente, y a aplicar más técnicas de **optimización** y **regularización** para mejorar la precisión del modelo.

---

### **6. ¿Qué referencias usaron? ¿Cuál fue la más útil? Incluya citas con los estándares adecuados.**

Las principales referencias que utilizamos fueron:

1. **Keras Functional API Guide**: Esta guía fue crucial para entender cómo estructurar y definir el modelo de red neuronal de manera flexible utilizando la API funcional de Keras.  
   Cita: *Keras. "Functional API." Keras Documentation. https://keras.io/guides/functional_api/*

2. **ChatGPT**: Utilizamos ChatGPT para resolver dudas sobre corrección de código, estructura de documentación y explicación de conceptos relacionados con redes neuronales.

3. **Dragon Warrior’s Notes on Gradient Descent**: Esta fuente fue extremadamente útil para entender cómo funciona el **descenso de gradiente** en redes neuronales, especialmente en capas densas.  
   Cita: *Dragon Warrior. "Gradient Descent and Layers." Statistical Learning Notes. https://dragonwarrior15.github.io/statistical-learning-notes/notes/deep_learning/chapters/gradient_descent/gradient_layers.html*

De estas, la más útil fue **la guía de la API funcional de Keras**, ya que nos permitió definir y organizar el modelo de manera eficiente, a la vez que nos dio una base sólida para la implementación de redes neuronales más complejas.
