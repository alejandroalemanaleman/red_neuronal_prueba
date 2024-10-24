import numpy as np
from neural_network import RedNeuronal
from activations import cross_entropy_loss, accuracy

def entrenar(modelo, X, Y, epochs, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for epoch in range(epochs):
        salida = modelo.forward(X)
        dW1, db1, dW2, db2 = modelo.retropropagacion(X, Y)

        modelo.actualizar_pesos_adam(dW1, db1, dW2, db2, lr, beta1, beta2, epsilon, epoch + 1)

        if epoch % 100 == 0:
            loss = cross_entropy_loss(salida, Y)
            acc = accuracy(salida, Y)
            print(f"Epoch {epoch} - Pérdida: {loss:.4f}, Precisión: {acc:.4f}")

# Ejemplo de uso con el dataset Iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Cargar el dataset de Iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Convertir las etiquetas en one-hot encoding
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# VALIDATION DATA para hiper params.
# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir el modelo
input_size = 4  # Número de características en el dataset Iris
hidden_size = 10
output_size = 3  # Número de clases

modelo = RedNeuronal(input_size, hidden_size, output_size)

# Entrenar el modelo
entrenar(modelo, X_train, y_train, epochs=1000)
