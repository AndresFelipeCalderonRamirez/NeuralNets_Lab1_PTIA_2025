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

Al utilizar el comando `derivative of max(0,z)`, en Wolfram Alpha, se obtiene que su derivada es:

![](img/relu-derivative.png)

$$
f'(z) =
\begin{cases}
0 & \text{si } z < 0 \\
1 & \text{si } z > 0 \\
\text{indefinida} & \text{si } z = 0
\end{cases}
$$


### Derivada función de costo: Entropia Cruzada:

La función de Entropía Cruzada se utiliza comúnmente en problemas de clasificación, utilizando el comando `- D[Sum[f[x] * Log[g[x]], {i, 1, n}], x]`, en Wolfram Alpha, se obtiene que su derivada es:

![](img/cross-entropy-alternative-form.png)

![](img/cross-entropy-derivative.png)

$\frac{n \left( g(x)^2 f''(x) \log(g(x)) + g(x) \left[ 2 f'(x) g'(x) + f(x) g''(x) \right] - f(x) g'(x)^2 \right)}{g(x)^2}$
