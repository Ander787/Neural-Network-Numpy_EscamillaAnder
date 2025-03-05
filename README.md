# README - Red Neuronal Numpy

## 🤓 Información del Proyecto  
**Materia:** Sistemas de Visión Artificial  
**Tarea:** Tarea 2.2 - Implementación y Entrenamiento de Red Neuronal  
**Estudiante:** Ander Heinrich Escamilla Wong  

**Fecha:** 03/03/2025  

---

## ✏️ Descripción General  
Este repositorio contiene una implementación en Python de una red neuronal artificial capaz de realizar clasificación binaria utilizando funciones de activación sigmoide y ReLU. La red neuronal se entrena utilizando el algoritmo de retropropagación y gradiente descendente. Los datos utilizados para el entrenamiento se generan aleatoriamente con una distribución gaussiana. El proyecto visualiza tanto los datos de entrenamiento como la predicción de los datos de prueba.

---

## 🧠 Estructura de la Red Neuronal  
La red neuronal implementada consta de tres capas ocultas y una capa de salida. Las funciones de activación utilizadas son:

- **ReLU (Rectified Linear Unit)** para las capas ocultas.
- **Sigmoide** para la capa de salida.

El modelo utiliza el **error cuadrático medio (MSE)** como función de pérdida, optimizando los parámetros a través de gradiente descendente.

---

# Proyecto de Red Neuronal

## ☝️ Requisitos Previos
Antes de ejecutar el código, asegúrate de tener instaladas las siguientes bibliotecas en tu entorno de Python:

```sh
pip install numpy matplotlib scikit-learn
```
## 📋 Estructura del Repositorio
```
proyecto_red_neuronal/
│── neural_network.py           # Código principal de la red neuronal
│── README.md                  # Este documento
```
### 1. `generate_data(N)`
Genera datos de clasificación con distribución gaussiana. Utiliza la función `make_gaussian_quantiles` de `scikit-learn` para crear dos clases de datos en un espacio bidimensional.

```python
def generate_data(N):
    # Genera datos de clasificación con distribución gaussiana
    gaussian_quantiles = make_gaussian_quantiles(
        mean=None, cov=0.1, n_samples=N, n_features=2, n_classes=2,
        shuffle=True, random_state=None)
    X, Y = gaussian_quantiles
    Y = Y[:, np.newaxis]  # Ajusta la dimensión de Y
    return X, Y
```
### 2. Funciones de activacion sigmoide
Función de activación sigmoide y su derivada. Si derivate=True, retorna la derivada de la función sigmoide. La función sigmoide es útil en redes neuronales porque introduce no linealidad y es diferenciable.
```python
def sigmoid(x, derivate=False):
    # Función de activación sigmoide y su derivada
    return np.exp(-x) / (np.exp(-x) + 1) ** 2 if derivate else 1 / (1 + np.exp(-x))
```
### 3. Funcion relu
Función de activación ReLU (Rectified Linear Unit) y su derivada. Si derivate=True, retorna la derivada de ReLU, que es 0 para valores negativos y 1 para valores positivos. ReLU es común en redes neuronales debido a su simplicidad y efectividad.
```python
def relu(x, derivate=False):
    # Función de activación ReLU y su derivada
    if derivate:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    return np.maximum(0, x)
```
### 4. mse(y, y_hat, derivate=False)
Función de error cuadrático medio (MSE) y su derivada. Si derivate=True, retorna la derivada de la función de pérdida, útil para la retropropagación. MSE es ampliamente utilizado para problemas de regresión.
```python
def mse(y, y_hat, derivate=False):
    # Función de error cuadrático medio y su derivada
    return (y_hat - y) if derivate else np.mean((y_hat - y) ** 2)
```
### 5. initialize_parameters_deep(layers_dims)
Inicializa los parámetros de la red neuronal (pesos y sesgos) de acuerdo con la estructura especificada en layers_dims. Los pesos se inicializan aleatoriamente para evitar que todos los neuronas tengan los mismos valores.
```python
def initialize_parameters_deep(layers_dims):
    # Inicializa los parámetros (pesos y sesgos) de la red neuronal
    parameters = {}
    for l in range(len(layers_dims) - 1):
        parameters[f'W{l+1}'] = (np.random.rand(layers_dims[l], layers_dims[l+1]) * 2) - 1
        parameters[f'b{l+1}'] = (np.random.rand(1, layers_dims[l+1]) * 2) - 1
    return parameters
```
### 6. train(x_data, y_data, learning_rate, params, training=True)
Entrena la red neuronal mediante propagación hacia adelante y retropropagación. Si training=True, realiza la actualización de los pesos y sesgos utilizando gradiente descendente. La propagación hacia adelante calcula las activaciones y la retropropagación ajusta los parámetros de la red.
```python
def train(x_data, y_data, learning_rate, params, training=True):
    # Propagación hacia adelante
    params['A0'] = x_data
    params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']
    params['A1'] = relu(params['Z1'])

    params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']
    params['A2'] = relu(params['Z2'])

    params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']
    params['A3'] = sigmoid(params['Z3'])

    output = params['A3']

    if training:
        # Retropropagación
        params['dZ3'] = mse(y_data, output, True) * sigmoid(params['A3'], True)
        params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])
        params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['A2'], True)
        params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])
        params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['A1'], True)
        params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])

        # Actualización de pesos y sesgos con gradiente descendente
        params['W3'] -= params['dW3'] * learning_rate
        params['W2'] -= params['dW2'] * learning_rate
        params['W1'] -= params['dW1'] * learning_rate
        params['b3'] -= np.mean(params['dW3'], axis=0, keepdims=True) * learning_rate
        params['b2'] -= np.mean(params['dW2'], axis=0, keepdims=True) * learning_rate
        params['b1'] -= np.mean(params['dW1'], axis=0, keepdims=True) * learning_rate

    return output
```
### 7. run_neural_network(iterations=50000, learning_rate=0.001, N=1000)
Función principal que genera los datos, inicializa los parámetros y entrena la red neuronal. Al finalizar el entrenamiento, visualiza los datos de entrenamiento y las predicciones realizadas sobre un conjunto de datos de prueba.
```python
def run_neural_network(iterations=50000, learning_rate=0.001, N=1000):
    # Inicializar y entrenar el modelo
    X, Y = generate_data(N)
    layers_dims = [2, 6, 10, 1]  # Estructura de la red neuronal
    params = initialize_parameters_deep(layers_dims)
    error = []

    for _ in range(iterations):
        output = train(X, Y, learning_rate, params)
        if _ % 50 == 0:
            print(mse(Y, output))  # Muestra el error cada 50 iteraciones
            error.append(mse(Y, output))

    # Visualización de los datos de entrenamiento
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)

    # Generación de datos de prueba y evaluación del modelo
    data_test_x = (np.random.rand(1000, 2) * 2) - 1
    data_test_y = train(data_test_x, X, 0.0001, params, training=False)

    y = np.where(data_test_y > 0.5, 1, 0)  # Clasificación de los datos de prueba

    plt.scatter(data_test_x[:, 0], data_test_x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
```
## 📖 Uso
Para ejecutar la red neuronal, usa el siguiente comando:

```sh
python main.py
```

Esto generará un gráfico de dispersión mostrando los datos de entrenamiento y otro con las predicciones realizadas por la red neuronal sobre un conjunto de datos de prueba en `src/`.

## 📊 Resultados Esperados

El programa generará:
-El gráfico de dispersión de los datos de entrenamiento, con puntos coloreados según su clase (0 o 1), generados aleatoriamente con distribución gaussiana.
-El gráfico de dispersión de los datos de prueba clasificados por la red neuronal, mostrando cómo la red ha aprendido a separar las dos clases.

## 🔍 Conclusión

El código demuestra cómo entrenar una red neuronal simple para clasificación binaria utilizando funciones de activación y gradiente descendente, visualizando los resultados de la clasificación sobre datos generados aleatoriamente.


## ⭐ ¡Dale una estrella al repo si te fue útil!
