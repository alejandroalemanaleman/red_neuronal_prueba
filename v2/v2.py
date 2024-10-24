import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Función de activación ReLU
def relu(x):
    return np.maximum(0, x)

# Derivada de ReLU para backpropagation
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Función softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Evita overflow numérico
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Inicializamos los pesos
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2


# Implementamos Adam Optimizer
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        if not self.m:
            # Initialize moments for each parameter
            for key in params.keys():
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.t += 1
        updates = {}
        for key in params.keys():
            grad_key = 'd' + key  # Make sure the grad key matches the correct gradient, like 'dW1' for 'W1'

            if grad_key not in grads:
                raise KeyError(f"Gradient key {grad_key} not found in grads.")

            # Update moments for each parameter
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[grad_key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[grad_key] ** 2)

            # Correct bias
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters
            updates[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updates


# Función de forward propagation con softmax en la salida
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)  # Salida con softmax
    return Z1, A1, Z2, A2


# Función de costo con entropía cruzada categórica
def compute_cost(A2, Y):
    m = Y.shape[0]
    cost = -np.sum(Y * np.log(A2)) / m
    return cost


# Función de backpropagation
def backpropagation(X, Y, W1, b1, W2, b2, Z1, A1, Z2, A2):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = 1. / m * np.dot(A1.T, dZ2)
    db2 = 1. / m * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = 1. / m * np.dot(X.T, dZ1)
    db1 = 1. / m * np.sum(dZ1, axis=0, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# Función de predicción
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=1)  # Seleccionamos la clase con mayor probabilidad
    return predictions


# Carga el dataset Iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encoding de las etiquetas
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)

# Dividimos en training (60%), validation (20%) y test (20%)
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.25, random_state=42)

# Inicializamos pesos
input_size = X_train.shape[1]
hidden_size = 10  # Este es un hiperparámetro que puedes ajustar
output_size = Y_train.shape[1]
W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)

# Parámetros de entrenamiento
epochs = 1000
adam_optimizer = AdamOptimizer(learning_rate=0.01)  # Tasa de aprendizaje es un hiperparámetro

# Entrenamos la red neuronal usando conjunto de validación para ajustar hiperparámetros
best_val_loss = float('inf')
best_params = None

for i in range(epochs):
    # Forward propagation
    Z1, A1, Z2, A2 = forward_propagation(X_train, W1, b1, W2, b2)

    # Computamos el costo para el conjunto de entrenamiento
    cost = compute_cost(A2, Y_train)

    # Backpropagation
    grads = backpropagation(X_train, Y_train, W1, b1, W2, b2, Z1, A1, Z2, A2)

    # Actualizamos los pesos usando Adam
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    params = adam_optimizer.update(params, grads)
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    # Validación
    _, _, _, A2_val = forward_propagation(X_val, W1, b1, W2, b2)
    val_loss = compute_cost(A2_val, Y_val)

    # Guardamos el mejor conjunto de hiperparámetros basado en la validación
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = params.copy()

    # Imprimimos el costo cada 100 épocas
    if i % 100 == 0:
        print(f"Epoch {i}, Training Cost: {cost}, Validation Loss: {val_loss}")

# Usamos los mejores hiperparámetros
W1, b1, W2, b2 = best_params["W1"], best_params["b1"], best_params["W2"], best_params["b2"]

# Predicciones en el conjunto de prueba
predictions = predict(X_test, W1, b1, W2, b2)
accuracy = np.mean(predictions == np.argmax(Y_test, axis=1))
print(f"Test Accuracy: {accuracy * 100}%")