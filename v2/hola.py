def initialize_weights(layer_sizes):
    """
    Inicializa los pesos y biases para una red de múltiples capas.

    Parameters:
        layer_sizes : list of int
            Lista donde cada elemento es el número de neuronas de cada capa (incluyendo capa de entrada y de salida).

    Returns:
        weights : list of np.array
            Lista de matrices de pesos para cada capa.
        biases : list of np.array
            Lista de vectores de bias para cada capa.
    """
    weights = []
    biases = []

    # Inicializamos los pesos y biases entre cada capa de la red
    for i in range(1, len(layer_sizes)):
        W = np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * 0.01
        b = np.zeros((1, layer_sizes[i]))
        weights.append(W)
        biases.append(b)

    return weights, biases


def backprop(self, x, y):
    """Return a tuple "(nabla_b, nabla_w)" representing the
    gradient for the cost function C_x.  "nabla_b" and
    "nabla_w" are layer-by-layer lists of numpy arrays, similar
    to "self.biases" and "self.weights"."""
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = x
    activations = [x]  # list to store all the activations, layer by layer
    zs = []  # list to store all the z vectors, layer by layer
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    # backward pass
    delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in xrange(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
    return (nabla_b, nabla_w)


...


def cost_derivative(self, output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return (output_activations - y)