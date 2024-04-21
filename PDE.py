import tensorflow.compat.v1 as tf
import numpy as np
import time

from NeuralNetwork import NeuralNetwork
from model_saver import Model_Save


# Base neural network PDE solver class
class PDENeuralNetwork:

    def __init__(
        self,
        domain=None,
        network=NeuralNetwork,
        interior_point_count=1024,
        boundary_point_count=1024,
        ratio_domains=True,
    ):
        self.loss_weight = tf.placeholder(tf.float64, name="lossWeight")

        self.domain = domain.copy()
        self.network = network

        # Compute domain sizes
        self.interior_domain_size = 1
        for axis in domain:
            self.interior_domain_size = self.interior_domain_size * (axis[1] - axis[0])

        self.boundary_domain_size = []
        self.total_boundary_domain_size = 0
        for i in range(len(domain)):
            self.boundary_domain_size.append(1)
            if ratio_domains:
                for j in range(len(domain)):
                    if j != i:
                        self.boundary_domain_size[i] = self.boundary_domain_size[i] * (
                            self.domain[j][1] - self.domain[j][0]
                        )
            self.total_boundary_domain_size += 2 * self.boundary_domain_size[i]

        # Initialize loss variables
        self.default_loss = None
        self.optimal_loss = None
        self.magnitude_loss = None

        self.default_loss_validate = None
        self.optimal_loss_validate = None
        self.magnitude_loss_validate = None

        self.fetch_list = None
        self.fetch_list_validate = None
        self.iteration = 0
        self.best_loss = np.infty
        self.best_weights = []
        self.best_biases = []
        self.start_time = 0

        self.analytical_interior = None
        self.analytical_boundary = None
        self.analytical_interior_magnitude = None
        self.analytical_boundary_magnitude = None

        self.interior_point_count = interior_point_count
        self.boundary_point_count = boundary_point_count

        self.history = []

    @staticmethod
    def partial_derivative(tensor, variable, order=1):
        for _ in range(order):
            if tensor is not None:
                tensor = tf.gradients(tensor, variable)[0]
        return tensor

    ###################################### Sampling ###################################
    # region Collocation Points
    def sample_data(
        self,
        interior_point_count,
        boundary_point_count,
        validate=False,
        loss_weight=None,
    ):
        feed_dict = dict()

        x_int = self.interior_sampling(interior_point_count)
        for i in range(len(self.domain)):
            feed_dict[self.network.x_int[i]] = x_int[i]

        x_bound = self.boundary_sampling(boundary_point_count)
        for i in range(len(self.domain)):
            feed_dict[self.network.x_bound[i]] = x_bound[i]

        boundary_condition = self.boundary_condition(x_bound)
        if boundary_condition is not None:
            feed_dict[self.network.boundary_condition] = boundary_condition

        if validate:
            x_int = self.interior_sampling(interior_point_count)
            for i in range(len(self.domain)):
                feed_dict[self.network.x_int_validate[i]] = x_int[i]

            x_bound = self.boundary_sampling(boundary_point_count)
            for i in range(len(self.domain)):
                feed_dict[self.network.x_bound_validate[i]] = x_bound[i]

            boundary_condition = self.boundary_condition(x_bound)
            if boundary_condition is not None:
                feed_dict[self.network.boundary_condition_validate] = boundary_condition

        if loss_weight is not None:
            feed_dict[self.loss_weight] = loss_weight

        return feed_dict

    # Sample uniform collocation points in the interior of the domain
    def interior_sampling(self, point_count):
        if point_count < 1:
            point_count = 1

        x_int = []
        for i in range(len(self.domain)):
            x_int.append(
                np.random.uniform(
                    self.domain[i][0], self.domain[i][1], (point_count, 1)
                )
            )
        return x_int

    # Sample uniform collocation points on the boundary of the domain
    def boundary_sampling(self, point_count):
        if point_count < 2 * len(self.domain):
            point_count = 2 * len(self.domain)

        x_bound = [np.empty((0, 1), dtype=np.float64) for _ in range(len(self.domain))]
        # Iterate over dimensions
        for i in range(len(self.domain)):
            # Iterate over boundaries
            for j in range(len(self.domain)):
                for bound in self.domain[j]:
                    new_points = max(
                        int(
                            point_count
                            * self.boundary_domain_size[i]
                            / self.total_boundary_domain_size
                        ),
                        1,
                    )

                    if j == i:
                        new_x = np.full((new_points, 1), bound, dtype=np.float64)
                    else:
                        new_x = np.random.uniform(
                            self.domain[j][0], self.domain[j][1], (new_points, 1)
                        )
                    x_bound[j] = np.concatenate((x_bound[j], new_x))
        return x_bound

    ######################################## Training #################################
    def train(
        self, train_mode, iterations=10000, loss_weight=None, custom_callback=None
    ):
        self.start_time = time.time()
        
        if train_mode == "Adaptive":
            self.__train_validate(
                loss_function=self.default_loss,
                iterations=iterations,
                fetch_list=self.fetch_list_validate,
                custom_callback=custom_callback,
            )
        else:
            raise TypeError("train_mode should be Adaptive - Check your spelling!")

    def __train_validate(
        self,
        loss_function,
        iterations,
        loss_weight=None,
        fetch_list=None,
        custom_callback=None,
    ):
        if custom_callback is None:
            callback = self.default_callback_validate
            fetch_list = fetch_list.copy()
            # print(fetch_list)
            fetch_list.insert(0, loss_function)
            for i in range(len(self.network.weights)):
                fetch_list.append(self.network.weights[i])

            for i in range(len(self.network.biases)):
                fetch_list.append(self.network.biases[i])
        else:
            callback = custom_callback

        # Stop the algorithm if the collocation point counts exceed 200,000 points
        while (
            self.interior_point_count < 200000
            and self.boundary_point_count < 200000
            and self.iteration < iterations
        ):
            try:
                feed_dict = self.sample_data(
                    interior_point_count=self.interior_point_count,
                    boundary_point_count=self.boundary_point_count,
                    validate=True,
                    loss_weight=loss_weight,
                )

                self.best_loss = np.infty
                self.network.NN_train(
                    loss_function,
                    iterations - self.iteration,
                    feed_dict,
                    fetch_list,
                    callback,
                )

            except OverfitError as e:
                # Raise the number of points
                if e.raise_int:
                    self.interior_point_count *= 2
                    print("Interior point count raised to ", self.interior_point_count)
                if e.raise_bound:
                    self.boundary_point_count *= 2
                    print("Boundary point count raised to ", self.boundary_point_count)

                if len(self.best_weights) > 0:
                    for i in range(len(self.network.weights)):
                        self.network.weights[i].load(
                            self.best_weights[i], self.network.session
                        )

                    for i in range(len(self.network.biases)):
                        self.network.biases[i].load(
                            self.best_biases[i], self.network.session
                        )

    def default_callback_validate(
        self, loss, loss_int, loss_bound, loss_int_validate, loss_bound_validate, *args
    ):
        self.iteration += 1
        self.history.append(
            Model_Save(
                self.iteration,
                loss_int,
                loss_bound,
                loss_int_validate,
                loss_bound_validate,
                self.interior_point_count,
                self.boundary_point_count,
                time.time() - self.start_time,
            )
        )
        print(
            "Iteration: ",
            self.iteration,
            ": Interior loss: ",
            "{:.4E}".format(loss_int),
            ": Interior validation loss: ",
            "{:.4E}".format(loss_int_validate),
            ", Boundary loss: ",
            "{:.4E}".format(loss_bound),
            ": Boundary validation loss: ",
            "{:.4E}".format(loss_bound_validate),
            ", Interior point count: ",
            self.interior_point_count,
            ", Boundary point count: ",
            self.boundary_point_count,
            " Time elapsed: ",
            "{:.2f}".format(time.time() - self.start_time),
            "s",
        )

        if loss < self.best_loss:
            self.best_loss = loss
            self.store_weights(*args)

        if loss_int < loss_int_validate / 5 or loss_bound < loss_bound_validate / 5:
            raise OverfitError(
                loss_int < loss_int_validate / 5, loss_bound < loss_bound_validate / 5
            )

    def store_weights(self, *args):
        self.best_weights = []
        self.best_biases = []

        counter = 0
        for _ in range(len(self.network.weights)):
            self.best_weights.append(args[counter].copy())
            counter += 1

        for _ in range(len(self.network.biases)):
            self.best_biases.append(args[counter].copy())
            counter += 1

    ################################# Start Plotting ############################

    # Wrapper function to get interior plot data
    def get_interior_plot_data(self, point_count, tensor, x, option_name = None):
        axis_count = 0
        for axis in x:
            if isinstance(axis, tuple):
                axis_count += 1

        if axis_count == 1:
            return self.get_interior_plot_data1d(point_count, tensor, x)

        elif axis_count == 2:
            return self.get_interior_plot_data2d(point_count, tensor, x, option_name)

        raise Exception("Bad plot domain: domain must be 1- or 2-dimensional")

    # Get interior plot data for 1d plots
    def get_interior_plot_data1d(self, point_count, tensor, x):
        x_int = self.regular_grid_points(point_count, x)
        feed_dict = self.get_feed_dict(x_int=x_int)
        y = self.network.session.run(tensor, feed_dict)

        x_plot = []
        for i in range(len(x)):
            if isinstance(x[i], tuple):
                x_plot.append(x_int[i])

        return x_plot[0], y

    # Get interior plot data for 2d plots
    def get_interior_plot_data2d(self, point_count, tensor, x, option_name):

        point_count = max(1, int(np.sqrt(point_count)))

        x_int = self.regular_grid_points(point_count, x)
        feed_dict = self.get_feed_dict(x_int=x_int)
        y = self.network.session.run(tensor, feed_dict)

        x_plot = []
        for i in range(len(x)):
            if isinstance(x[i], tuple):
                x_plot.append(x_int[i].reshape(point_count, point_count))

        if option_name == 'American':
            return x_plot[0], x_plot[1], y.reshape(point_count, point_count)
        else:
            analytical = self.network.session.run(self.analytical_interior, feed_dict)  
            return x_plot[0], x_plot[1], y.reshape(point_count, point_count), analytical.reshape(point_count, point_count)
    
    # Get a regular grid of points, formatted into column vectors
    @staticmethod
    def regular_grid_points(point_count, x):
        x_int = []
        for i in range(len(x)):
            if isinstance(x[i], tuple):
                x_int.append(
                    np.linspace(x[i][0], x[i][1], point_count, dtype=np.float64)
                )

            else:
                x_int.append(np.full((1, 1), x[i], dtype=np.float64))

        x_int_mesh = np.meshgrid(*x_int)
        return [x.reshape(-1, 1) for x in x_int_mesh]

    # Get uniform input points in the boundary of the domain, formatted into lists of column vectors
    def get_boundary_plot_data(self, point_count, tensor, x):
        # Compute overlap of x and the domain of the PDE, and store the axes of the plot
        x_overlap = []
        axes = []
        for i in range(len(x)):
            if isinstance(x[i], tuple):
                axes.append(i)
                if x[i][0] > self.domain[i][1] or x[i][1] < self.domain[i][0]:
                    raise Exception("Requested domain does not contain the boundary")

                x_min = max(x[i][0], self.domain[i][0])
                x_max = min(x[i][1], self.domain[i][1])
                if x_min == x_max:
                    x_overlap.append(x_min)
                else:
                    x_overlap.append((x_min, x_max))

            else:
                if x[i] < self.domain[i][0] or x[i] > self.domain[i][1]:
                    raise Exception("Requested domain does not contain the boundary")

                else:
                    x_overlap.append(x[i])

        if len(axes) > 2 or len(axes) < 1:
            raise Exception("Bad plot domain: domain must be 1- or 2-dimensional")

        # Find which boundaries are included in the requested domain and store the corresponding domain
        bound_included = [-1, -1] * len(x)
        for i in range(len(x_overlap)):
            if isinstance(x_overlap[i], tuple):
                if x_overlap[i][0] == self.domain[i][0]:
                    dim = 0
                    for j in range(len(x_overlap)):
                        if j != i and isinstance(x_overlap[j], tuple):
                            dim += 1
                    bound_included[2 * i] = dim

                if x_overlap[i][1] == self.domain[i][1]:
                    dim = 0
                    for j in range(len(x_overlap)):
                        if j != i and isinstance(x_overlap[j], tuple):
                            dim += 1
                    bound_included[2 * i + 1] = dim

            else:
                if x_overlap[i] == self.domain[i][0]:
                    dim = 0
                    for j in range(len(x_overlap)):
                        if j != i and isinstance(x_overlap[j], tuple):
                            dim += 1
                    bound_included[2 * i] = dim

                if x_overlap[i] == self.domain[i][1]:
                    dim = 0
                    for j in range(len(x_overlap)):
                        if j != i and isinstance(x_overlap[j], tuple):
                            dim += 1
                    bound_included[2 * i + 1] = dim

        boundary_dimension = max(bound_included)
        if boundary_dimension < 0:
            raise Exception("Requested domain does not contain a boundary")

        # Get boundary plot data
        if boundary_dimension == 0:
            return self.get_boundary_plot_data0d(
                tensor, x_overlap, bound_included, axes
            )

        elif boundary_dimension == 1:
            return self.get_boundary_plot_data1d(
                point_count, tensor, x_overlap, bound_included, axes
            )

        elif boundary_dimension == 2:
            return self.get_boundary_plot_data2d(point_count, tensor, x_overlap)

    # Get 0-d boundary plot data as 1-d arrays
    def get_boundary_plot_data0d(self, tensor, x, bound_included, axes):
        # Generate list with point coordinates
        x_bound = []
        for i in range(len(x)):
            x_bound.append([])

        for i in range(len(bound_included)):
            if bound_included[i] == 0:
                for j in range(len(x)):
                    if j == int(i / 2):
                        x_bound[j].append(self.domain[j][i % 2])

                    else:
                        x_bound[j].append(x[j])

        for i in range(len(x_bound)):
            x_bound[i] = np.array(x_bound[i]).reshape(-1, 1)

        feed_dict = self.get_feed_dict(x_bound=x_bound)
        y = self.network.session.run(tensor, feed_dict)
        output = [x_bound[axis].reshape(-1) for axis in axes] + [y.reshape(-1)]
        return tuple(output)

    # Get 1-d boundary plot data as lists of 1-d arrays
    def get_boundary_plot_data1d(self, point_count, tensor, x, bound_included, axes):
        # Compute number of included boundaries
        boundary_count = 0
        for bound in bound_included:
            if bound == 1:
                boundary_count += 1

        # Create point arrays
        x_bound = []
        for i in range(len(bound_included)):
            if bound_included[i] == 1:
                new_x = x.copy()
                bound_index = int(i / 2)
                new_x[bound_index] = self.domain[bound_index][i % 2]
                x_bound.append(
                    self.regular_grid_points(
                        max(point_count // boundary_count, 2), new_x
                    )
                )

        # Evaluate tensor
        y = []
        x_output = []
        for j in range(len(axes)):
            x_output.append([])

        for i in range(len(x_bound)):
            feed_dict = self.get_feed_dict(x_bound=x_bound[i])
            y.append(self.network.session.run(tensor, feed_dict).reshape(-1))

            for j in range(len(axes)):
                x_output[j].append(x_bound[i][axes[j]].reshape(-1))

        return tuple(x_output + [y])

    # Get 2-d boundary plot data
    def get_boundary_plot_data2d(self, point_count, tensor, x):
        point_count = max(1, int(np.sqrt(point_count)))

        x_bound = self.regular_grid_points(point_count, x)
        feed_dict = self.get_feed_dict(x_bound=x_bound)
        y = self.network.session.run(tensor, feed_dict)
        
        x_plot = []
        for i in range(len(x)):
            if isinstance(x[i], tuple):
                x_plot.append(x_bound[i].reshape(point_count, point_count))

        return x_plot[0], x_plot[1], y.reshape(point_count, point_count)

    ######################## End Plotting ########################

    # Override this to define boundary conditions
    def boundary_condition(self, x):
        return None

    def get_feed_dict(
        self,
        x_int=None,
        x_bound=None,
        x_int_validate=None,
        x_bound_validate=None,
        loss_weight=None,
    ):
        feed_dict = dict()

        if x_int is not None:
            for i in range(len(self.domain)):
                feed_dict[self.network.x_int[i]] = x_int[i]

        if x_bound is not None:
            for i in range(len(self.domain)):
                feed_dict[self.network.x_bound[i]] = x_bound[i]

            boundary_condition = self.boundary_condition(x_bound)
            if boundary_condition is not None:
                feed_dict[self.network.boundary_condition] = boundary_condition

        if x_int_validate is not None:
            for i in range(len(self.domain)):
                feed_dict[self.network.x_int_validate[i]] = x_int[i]

        if x_bound_validate is not None:
            for i in range(len(self.domain)):
                feed_dict[self.network.x_bound_validate[i]] = x_bound_validate[i]

            boundary_condition = self.boundary_condition(x_bound_validate)
            if boundary_condition is not None:
                feed_dict[self.network.boundary_condition_validate] = boundary_condition

        if loss_weight is not None:
            feed_dict[self.loss_weight] = loss_weight

        return feed_dict
    
    def analytical_solution(self, x):
        return None

    def compute_analytical_solution(self, feed_dict):
        if self.analytical_interior is not None:
            # European Options
            return self.network.session.run(self.analytical_interior, feed_dict)
        else:
            # American Options
            return np.array(
                [self.analytical_solution(self._get_x_int_from_feed_dict(feed_dict))]
            ).transpose()

    def _get_x_int_from_feed_dict(self, feed_dict):
        x_int = []
        for i in range(len(self.domain)):
            x_int.append(feed_dict[self.network.x_int[i]].transpose()[0])
        return x_int

    def compute_l2_error(self, relative=True, sample_points=5000):
        feed_dict = self.sample_data(
            interior_point_count=sample_points, boundary_point_count=sample_points
        )
        prediction = self.network.session.run(self.network.y_int, feed_dict)
        analytical = self.compute_analytical_solution(feed_dict)

        if relative:
            error = np.sqrt(np.sum((prediction - analytical) ** 2)) / np.sqrt(
                np.sum(analytical**2)
            )
            print("Relative L2 error: ", "{:.4E}".format(error))
            return error
        else:
            error = np.sqrt(np.sum((prediction - analytical) ** 2))
            print("L2 error: ", "{:.4E}".format(error))
            return error

    def compute_max_error(self, relative=True, sample_points=5000):
        feed_dict = self.sample_data(
            interior_point_count=sample_points, boundary_point_count=sample_points
        )
        prediction = self.network.session.run(self.network.y_int, feed_dict)
        analytical = self.compute_analytical_solution(feed_dict)

        if relative:
            error = np.max(prediction - analytical) / np.max(analytical)
            print("Relative L_Infinity error: ", "{:.4E}".format(error))
            return error
        else:
            error = np.max(prediction - analytical)
            print("L_Infinity error: ", "{:.4E}".format(error))
            return error


# Exception class used to terminate the scipy optimizer interface
class OverfitError(Exception):
    def __init__(self, raise_int, raise_bound):
        self.raise_int = raise_int
        self.raise_bound = raise_bound
