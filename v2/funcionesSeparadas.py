def forward_propagation(X, weights, biases, activations):
    """
    Realiza la propagación hacia adelante en una red de múltiples capas.

    Parameters:
        X : np.array
            Entrada de la red.
        weights : list of np.array
            Lista de matrices de pesos para cada capa.
        biases : list of np.array
            Lista de vectores de bias para cada capa.
        activations : list of functions
            Funciones de activación para cada capa.

    Returns:
        activations_list : list of np.array
            Lista de activaciones para cada capa.
        zs : list of np.array
            Lista de valores z (pre-activación) para cada capa.
    """
    activations_list = [X]
    zs = []
    activation = X

    # Propagación hacia adelante para todas las capas menos la última
    for i in range(len(weights) - 1):
        z = np.dot(activation, weights[i]) + biases[i]
        zs.append(z)
        activation = activations[i](z)
        activations_list.append(activation)

    # Última capa con softmax
    z = np.dot(activation, weights[-1]) + biases[-1]
    zs.append(z)
    activation = softmax(z)
    activations_list.append(activation)

    return activations_list, zs


def backward_propagation(Y, weights, biases, activations_list, zs, activation_derivatives):
    """
    Realiza la retropropagación en una red de múltiples capas para calcular los gradientes.

    Parameters:
        Y : np.array
            Etiquetas en formato one-hot.
        weights : list of np.array
            Lista de matrices de pesos para cada capa.
        biases : list of np.array
            Lista de vectores de bias para cada capa.
        activations_list : list of np.array
            Lista de activaciones para cada capa.
        zs : list of np.array
            Lista de valores z (pre-activación) para cada capa.
        activation_derivatives : list of functions
            Derivadas de las funciones de activación para cada capa.

    Returns:
        nabla_b : list of np.array
            Gradientes de los biases para cada capa.
        nabla_w : list of np.array
            Gradientes de los pesos para cada capa.
    """
    num_layers = len(weights)
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]

    # Cálculo del error en la capa de salida
    delta = activations_list[-1] - Y
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(activations_list[-2].T, delta) # Creo que hay que quitarlo / Y.shape[0]

    # Retropropagación a través de las capas ocultas
    for l in range(2, num_layers + 1):
        z = zs[-l]
        sp = activation_derivatives[-l](z)
        delta = np.dot(delta, weights[-l + 1].T) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(activations_list[-l - 1].T, delta) # Esto también / Y.shape[0]

    return nabla_b, nabla_w

def compute_cost(Y_pred, Y_true):
    """
    Calcula el costo usando la entropía cruzada categórica.

    Parameters:
        Y_pred : np.array
            Salidas predichas de la red (después de softmax).
        Y_true : np.array
            Etiquetas en formato one-hot.

    Returns:
        cost : float
            El valor del costo (entropía cruzada categórica).
    """
    m = Y_true.shape[0]
    cost = -np.sum(Y_true * np.log(Y_pred)) / m
    return cost


def train_network(X, Y, weights, biases, activations, activation_derivatives, epochs, learning_rate):
    """
    Entrena la red neuronal.

    Parameters:
        X : np.array
            Entrada de la red.
        Y : np.array
            Etiquetas en formato one-hot.
        weights : list of np.array
            Lista de matrices de pesos para cada capa.
        biases : list of np.array
            Lista de vectores de bias para cada capa.
        activations : list of functions
            Funciones de activación para cada capa.
        activation_derivatives : list of functions
            Derivadas de las funciones de activación para cada capa.
        epochs : int
            Número de épocas para el entrenamiento.
        learning_rate : float
            Tasa de aprendizaje.

    Returns:
        weights, biases : list of np.array
            Pesos y biases ajustados después del entrenamiento.
    """
    for epoch in range(epochs):
        # Forward propagation
        activations_list, zs = forward_propagation(X, weights, biases, activations)

        # Cálculo del costo
        cost = compute_cost(activations_list[-1], Y)

        # Backward propagation
        nabla_b, nabla_w = backward_propagation(Y, weights, biases, activations_list, zs, activation_derivatives)

        # Actualización de pesos y biases
        for i in range(len(weights)):
            weights[i] -= learning_rate * nabla_w[i]
            biases[i] -= learning_rate * nabla_b[i]

        # Opcional: imprimir el costo cada cierto número de épocas
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")

    return weights, biases

