import tensorflow.compat.v1 as tf
from external_optimizer import ScipyOptimizerInterface
import numpy as np

# The activation function is Softplus by default
# for non-negative stock price

# Neural network class
class NeuralNetwork:
    def __init__(self, input_dimension, hidden_layers, activation_function = tf.nn.softplus,
                 seed=None):
        self.layers = [input_dimension] + hidden_layers + [1]
        self.activation_function = activation_function

        if seed is not None:
            tf.set_random_seed(seed)
            np.random.seed(seed)

        # Network parameters
        self.weights, self.biases = self.network_params_generator()

        # Input placeholders
        self.x_int = []
        self.x_bound = []
        self.x_int_validate = []
        self.x_bound_validate = []
        for i in range(input_dimension):
            self.x_int.append(tf.placeholder(tf.float64, shape=[None, 1], name="x_int" + str(i)))

            self.x_bound.append(tf.placeholder(tf.float64, shape=[None, 1], name="x_bound" + str(i)))
            self.x_int_validate.append(
                tf.placeholder(tf.float64, shape=[None, 1], name="x_int_validate" + str(i)))
            self.x_bound_validate.append(
                tf.placeholder(tf.float64, shape=[None, 1], name="x_bound_validate" + str(i)))

        # Outputs
        self.y_int = self.build_graph(self.x_int)
        self.y_bound = self.build_graph(self.x_bound)
        self.y_int_validate = self.build_graph(self.x_int_validate)
        self.y_bound_validate = self.build_graph(self.x_bound_validate)

        # Boundary condition & Source functions
        self.boundary_condition = tf.placeholder(tf.float64, shape=[None, 1], name="BoundaryCondition")
        self.boundary_condition_validate = tf.placeholder(tf.float64, shape=[None, 1],
                                                          name="BoundaryConditionValidate")

        self.saver = tf.train.Saver()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def network_params_generator(self):
        weights = []
        biases = []

        initializer = tf.initializers.he_normal()

        for i in range(len(self.layers) - 1):
            x = tf.cast(initializer((self.layers[i], self.layers[i + 1])), tf.float64)
            print(x)
            w = tf.Variable(x, dtype=tf.float64)
            b = tf.Variable(tf.zeros([1, self.layers[i + 1]], dtype=tf.float64), dtype=tf.float64)

            weights.append(w)
            biases.append(b)

        return weights, biases

    def build_graph(self, x):
        y = tf.concat(x, axis=1)
        for i in range(len(self.layers) - 2):
            w = self.weights[i]
            b = self.biases[i]

            y = self.activation_function(tf.add(tf.matmul(y, w), b))

        w = self.weights[-1]
        b = self.biases[-1]
        return tf.add(tf.matmul(y, w), b)

    def NN_train(self, loss_function, iterations, feed_dict, fetch_list, callback):
        optimizer = ScipyOptimizerInterface(tf.log(loss_function),
                                            method='L-BFGS-B',
                                            options={'maxiter': iterations,
                                                     'maxfun': iterations,
                                                     'maxcor': 50,
                                                     'maxls': 50,
                                                     'ftol': 1.0 * np.finfo(
                                                         np.float64).eps,
                                                     'gtol': 0.000001})

        optimizer.minimize(self.session, feed_dict=feed_dict, fetches=fetch_list, loss_callback=callback)

    def predict(self, x):
        feed_dict = dict()
        for i in range(len(x)):
            feed_dict[self.x_int[i]] = x[i]
        return self.session.run(self.y_int, feed_dict=feed_dict)

    def save_weights(self, path="Autosave"):
        self.saver.save(self.session, path)

    def load_weights(self, path="Autosave"):
        self.saver.restore(self.session, path)

    def cleanup(self):
        self.session.close()
        tf.reset_default_graph()