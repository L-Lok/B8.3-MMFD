from abc import abstractmethod

import tensorflow.compat.v1 as tf
import numpy as np
# from NeuralNetwork import NeuralNetwork
from PDE import PDENeuralNetwork
from tensorflow_probability import distributions as tfd


class ExchangeOptionBase(PDENeuralNetwork):
    def __init__(self, domain, network):
        PDENeuralNetwork.__init__(self, domain=domain, network=network)

        ############################# Regular Losses #########################
        loss_int, loss_bound = self.compute_loss_terms(
            self.network.x_int,
            self.network.y_int,
            self.network.y_bound,
            self.network.boundary_condition,
        )
        self.default_loss = tf.add(loss_int, loss_bound)
        ########################## Validation losses ##########################
        (loss_int_validate, loss_bound_validate) = self.compute_loss_terms(
            self.network.x_int_validate,
            self.network.y_int_validate,
            self.network.y_bound_validate,
            self.network.boundary_condition_validate,
        )

        ##################### default loss validation #####################
        self.default_loss_validate = tf.add(loss_int_validate, loss_bound_validate)

        # Create fetch lists
        self.fetch_list = [loss_int, loss_bound]
        self.fetch_list_validate = [
            loss_int,
            loss_bound,
            loss_int_validate,
            loss_bound_validate,
        ]

        # Analytical Solution
        self.analytical_interior = self.analytical_solution(self.network.x_int)
        self.analytical_boundary = self.analytical_solution(self.network.x_bound)

    def compute_loss_terms(self, x_int, y_int, y_bound, boundary_condition):
        dvdt = PDENeuralNetwork.partial_derivative(y_int, self.time_stamp(x_int), 1)

        dvds1 = PDENeuralNetwork.partial_derivative(y_int, self.stock_price_1(x_int), 1)

        dvds2 = PDENeuralNetwork.partial_derivative(y_int, self.stock_price_2(x_int), 1)

        d2vds12 = PDENeuralNetwork.partial_derivative(
            dvds1, self.stock_price_2(x_int), 1
        )

        d2vds1_2nd = PDENeuralNetwork.partial_derivative(
            y_int, self.stock_price_1(x_int), 2
        )

        d2vds2_2nd = PDENeuralNetwork.partial_derivative(
            y_int, self.stock_price_2(x_int), 2
        )

        L = (
            dvdt
            + 0.5
            * self.sigma_1(x_int) ** 2
            * tf.square(self.stock_price_1(x_int))
            * d2vds1_2nd
            + 0.5
            * self.sigma_2(x_int) ** 2
            * tf.square(self.stock_price_2(x_int))
            * d2vds2_2nd
            + self.rho(x_int)
            * self.sigma_2(x_int)
            * self.sigma_1(x_int)
            * self.stock_price_1(x_int)
            * self.stock_price_2(x_int)
            * d2vds12
            + (self.rf_rate(x_int) - self.dividend_1(x_int))
            * self.stock_price_1(x_int)
            * dvds1
            + (self.rf_rate(x_int) - self.dividend_2(x_int))
            * self.stock_price_2(x_int)
            * dvds2
            - self.rf_rate(x_int) * y_int
        )

        loss_int = tf.reduce_mean(tf.square(L))

        loss_bound = tf.reduce_mean(tf.square(y_bound - boundary_condition))

        return loss_int, loss_bound

    def analytical_solution(self, x):
        d1 = (
            tf.log(self.stock_price_1(x) / self.stock_price_2(x))
            + (self.dividend_2(x) - self.dividend_1(x) + (self.sigma(x) ** 2) / 2)
        ) / (self.sigma(x) * tf.sqrt(self.maturity(x) - self.time_stamp(x)))

        d2 = d1 - self.sigma(x) * tf.sqrt(self.maturity(x) - self.time_stamp(x))

        gaussian = tfd.Normal(loc=0.0, scale=1.0)

        cdf_d1 = tf.cast(gaussian.cdf(tf.cast(d1, tf.float32)), tf.float64)
        cdf_d2 = tf.cast(gaussian.cdf(tf.cast(d2, tf.float32)), tf.float64)

        return (
            tf.exp(-self.dividend_1(x) * (self.maturity(x) - self.time_stamp(x)))
            * self.stock_price_1(x)
            * cdf_d1
            - tf.exp(-self.dividend_2(x) * (self.maturity(x) - self.time_stamp(x)))
            * self.stock_price_2(x)
            * cdf_d2
        )

    # remove the points on the vertical line t = 0 since no initial conditions are needed
    def boundary_sampling(self, point_count):
        x_bound = super().boundary_sampling(point_count)
        return np.delete(x_bound, np.where(self.time_stamp(x_bound) == 0)[0], axis=1)
    
    def boundary_condition(self, x):
        bound = []
        for i in range(len(self.time_stamp(x))):
            if abs(self.time_stamp(x)[i] - self.maturity(x)) <= 1e-3:
                # print(self.stock_price_1(x)[i] - self.stock_price_2(x)[i]);exit()
                bound.append(np.maximum((self.stock_price_1(x)[i] - self.stock_price_2(x)[i]), 0))
            else:
                bound.append(self.analytical_solution([self.stock_price_1(x)[i], self.stock_price_2(x)[i], self.time_stamp(x)[i]]).eval(session = self.network.session))
        # return  np.maximum((self.stock_price_1(x) - self.stock_price_2(x)), 0)
        return bound
    
    def get_analytical_plot_data(self, domain):
        new_x = None
        inputs = []
        for x_ in domain:
            if isinstance(x_, tuple):
                if new_x is None:
                    new_x = np.linspace(x_[0], x_[1])
                    inputs.append(tf.constant(new_x))
                else:
                    raise ValueError("Can only provide one range in the domain.")
            elif isinstance(x_, int) or isinstance(x_, float):
                inputs.append(tf.constant(np.linspace(x_, x_)))
        result = self.analytical_solution(inputs)
        return new_x, result.eval(session=tf.Session())

    def get_predicted_plot_data(self, domain):
        new_x = None
        inputs = []
        for xx in domain:
            if isinstance(xx, tuple):
                if new_x is None:
                    new_x = np.array([np.linspace(xx[0], xx[1])]).transpose()
                    inputs.append(new_x)
                else:
                    raise ValueError("Can only provide one range in the domain.")
            elif isinstance(xx, int) or isinstance(xx, float):
                inputs.append(np.array([np.linspace(xx, xx)]).transpose())
        feed_dict = self.get_feed_dict(x_int=inputs)
        y = self.network.session.run(self.network.y_int, feed_dict)
        return new_x.transpose()[0], y.transpose()[0]

    @abstractmethod
    def stock_price_1(self, x):
        pass

    @abstractmethod
    def stock_price_2(self, x):
        pass

    @abstractmethod
    def sigma_1(self, x):
        pass

    @abstractmethod
    def sigma_2(self, x):
        pass

    @abstractmethod
    def sigma(self, x):
        pass

    @abstractmethod
    def dividend_1(self, x):
        pass

    @abstractmethod
    def dividend_2(self, x):
        pass

    @abstractmethod
    def rf_rate(self, x):
        pass

    @abstractmethod
    def maturity(self, x):
        pass

    @abstractmethod
    def time_stamp(self, x):
        pass

    @abstractmethod
    def rho(self, x):
        pass


class ExchangeOptionSt(ExchangeOptionBase):
    input_dim = 3

    def __init__(
        self,
        strike_price,
        rf_rate,
        dividend_1,
        dividend_2,
        sigma_1,
        sigma_2,
        rho,
        maturity,
        network,
    ):
        self._strike_price = float(strike_price)
        self._rf_rate = float(rf_rate)
        self._div_1 = float(dividend_1)
        self._div_2 = float(dividend_2)
        self._sigma_1 = float(sigma_1)
        self._sigma_2 = float(sigma_2)
        self._maturity = float(maturity)
        self._rho = float(rho)

        domain = [
            (0, 4 * self._strike_price),
            (0, 4 * self._strike_price),
            (0, self._maturity),
        ]
        ExchangeOptionBase.__init__(self, domain=domain, network=network)

    def stock_price_1(self, x):
        return x[0]

    def stock_price_2(self, x):
        return x[1]

    def time_stamp(self, x):
        return x[2]

    def sigma_1(self, x):
        return self._sigma_1

    def sigma_2(self, x):
        return self._sigma_2

    def sigma(self, x):
        return tf.cast(
            tf.sqrt(
                self.sigma_1(x) ** 2
                + self.sigma_2(x) ** 2
                - 2 * self.sigma_1(x) * self.sigma_2(x) * self.rho(x)
            ),
            tf.float64,
        )

    def dividend_1(self, x):
        return self._div_1

    def dividend_2(self, x):
        return self._div_2

    def rf_rate(self, x):
        return self._rf_rate

    def maturity(self, x):
        return self._maturity

    def rho(self, x):
        return self._rho
