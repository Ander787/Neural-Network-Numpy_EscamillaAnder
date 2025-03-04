import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

def run_neural_network(iterations=50000, learning_rate=0.001, N=1000):
    def generate_data(N):
        # Genera datos de clasificación con distribución gaussiana
        gaussian_quantiles = make_gaussian_quantiles(
            mean=None, cov=0.1, n_samples=N, n_features=2, n_classes=2,
            shuffle=True, random_state=None)
        X, Y = gaussian_quantiles
        Y = Y[:, np.newaxis]  # Ajusta la dimensión de Y
        return X, Y

    def sigmoid(x, derivate=False):
        # Función de activación sigmoide y su derivada
        return np.exp(-x) / (np.exp(-x) + 1) ** 2 if derivate else 1 / (1 + np.exp(-x))

    def relu(x, derivate=False):
        # Función de activación ReLU y su derivada
        if derivate:
            x[x <= 0] = 0
            x[x > 0] = 1
            return x
        return np.maximum(0, x)

    def mse(y, y_hat, derivate=False):
        # Función de error cuadrático medio y su derivada
        return (y_hat - y) if derivate else np.mean((y_hat - y) ** 2)

    def initialize_parameters_deep(layers_dims):
        # Inicializa los parámetros (pesos y sesgos) de la red neuronal
        parameters = {}
        for l in range(len(layers_dims) - 1):
            parameters[f'W{l+1}'] = (np.random.rand(layers_dims[l], layers_dims[l+1]) * 2) - 1
            parameters[f'b{l+1}'] = (np.random.rand(1, layers_dims[l+1]) * 2) - 1
        return parameters

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
    data_test_x = (np.random.rand(1000,2)*2)-1
    data_test_y = train(data_test_x, X, 0.0001, params, training=False)

    y = np.where(data_test_y > 0.5, 1, 0)  # Clasificación de los datos de prueba

    plt.scatter(data_test_x[:, 0], data_test_x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    
if __name__ == "__main__":
    run_neural_network()  # Ejecuta la red neuronal

