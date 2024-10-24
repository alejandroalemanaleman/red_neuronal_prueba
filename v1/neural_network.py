
import numpy as np
from activations import relu, relu_derivada, softmax

class RedNeuronal:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización de pesos y sesgos
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        # Parámetros para Adam
        self.mW1, self.mb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.mW2, self.mb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.vW1, self.vb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vW2, self.vb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)

    def forward(self, X):
        # Capa oculta
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)

        # Capa de salida
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = softmax(self.Z2)

        return self.A2

    def retropropagacion(self, X, Y):
        # Cálculo de errores
        m = X.shape[0]
        dZ2 = self.A2 - Y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivada(self.Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def actualizar_pesos_adam(self, dW1, db1, dW2, db2, lr, beta1, beta2, epsilon, t):
        # Actualizar con Adam
        self.mW1 = beta1 * self.mW1 + (1 - beta1) * dW1
        self.mb1 = beta1 * self.mb1 + (1 - beta1) * db1
        self.mW2 = beta1 * self.mW2 + (1 - beta1) * dW2
        self.mb2 = beta1 * self.mb2 + (1 - beta1) * db2

        self.vW1 = beta2 * self.vW1 + (1 - beta2) * np.square(dW1)
        self.vb1 = beta2 * self.vb1 + (1 - beta2) * np.square(db1)
        self.vW2 = beta2 * self.vW2 + (1 - beta2) * np.square(dW2)
        self.vb2 = beta2 * self.vb2 + (1 - beta2) * np.square(db2)

        mW1_corr = self.mW1 / (1 - beta1 ** t)
        mb1_corr = self.mb1 / (1 - beta1 ** t)
        mW2_corr = self.mW2 / (1 - beta1 ** t)
        mb2_corr = self.mb2 / (1 - beta1 ** t)

        vW1_corr = self.vW1 / (1 - beta2 ** t)
        vb1_corr = self.vb1 / (1 - beta2 ** t)
        vW2_corr = self.vW2 / (1 - beta2 ** t)
        vb2_corr = self.vb2 / (1 - beta2 ** t)

        self.W1 -= lr * mW1_corr / (np.sqrt(vW1_corr) + epsilon)
        self.b1 -= lr * mb1_corr / (np.sqrt(vb1_corr) + epsilon)
        self.W2 -= lr * mW2_corr / (np.sqrt(vW2_corr) + epsilon)
        self.b2 -= lr * mb2_corr / (np.sqrt(vb2_corr) + epsilon)
