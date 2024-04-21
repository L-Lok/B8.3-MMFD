from abc import abstractmethod

import tensorflow.compat.v1 as tf
import numpy as np
from PDE import PDENeuralNetwork
from tensorflow_probability import distributions as tfd



class EuropeanOptionsBase(PDENeuralNetwork):
    def __init__(self, domain, option_type, network=None):
        self.option_type = option_type

        PDENeuralNetwork.__init__(self, domain=domain, network=network)

        ############################## Regular losses ##########################
        loss_int, loss_bound = self.compute_loss_terms(
            self.network.x_int,
            self.network.y_int,
            self.network.y_bound,
            self.network.boundary_condition,
        )
        ################## default loss function #################
        self.default_loss = tf.add(loss_int, loss_bound)

        ########################## Validation losses ##########################
        (
            loss_int_validate,
            loss_bound_validate
        ) = self.compute_loss_terms(
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
        dvds = PDENeuralNetwork.partial_derivative(y_int, self.stock_price(x_int), 1)
        dvdt = PDENeuralNetwork.partial_derivative(y_int, self.time_stamp(x_int), 1)
        d2vds2 = PDENeuralNetwork.partial_derivative(y_int, self.stock_price(x_int), 2)
        
        loss_int = tf.reduce_mean(
            tf.square(
                dvdt
                + 0.5
                * (self.sigma(x_int) ** 2)
                * tf.square(self.stock_price(x_int))
                * d2vds2
                + (self.rf_rate(x_int) - self.dividend(x_int))
                * self.stock_price(x_int)
                * dvds
                - self.rf_rate(x_int) * y_int
            )
        )
        loss_bound = tf.reduce_mean(tf.square(y_bound - boundary_condition))

        return loss_int, loss_bound

    def analytical_solution(self, x):
        d1 = (
            tf.log(self.stock_price(x) / self.strike_price(x))
            + (self.rf_rate(x) - self.dividend(x) + 0.5 * self.sigma(x) ** 2)
            * (self.maturity(x) - self.time_stamp(x))
        ) / (self.sigma(x) * tf.sqrt(self.maturity(x) - self.time_stamp(x)))
        d2 = d1 - self.sigma(x) * tf.sqrt(self.maturity(x) - self.time_stamp(x))

        gaussian = tfd.Normal(loc=0.0, scale=1.0)
        cdf_d1 = tf.cast(gaussian.cdf(tf.cast(d1, tf.float32)), tf.float64)
        cdf_d2 = tf.cast(gaussian.cdf(tf.cast(d2, tf.float32)), tf.float64)
        if self.option_type == 'call':
            return (
                self.stock_price(x)
                * tf.exp(-self.dividend(x) * (self.maturity(x) - self.time_stamp(x)))
                * cdf_d1
                - self.strike_price(x)
                * tf.exp(-self.rf_rate(x) * (self.maturity(x) - self.time_stamp(x)))
                * cdf_d2
            )
        elif self.option_type == 'put':
            return self.strike_price(x) * tf.exp(
                -self.rf_rate(x) * (self.maturity(x) - self.time_stamp(x))
            ) * (1.0 - cdf_d2) - self.stock_price(x) * tf.exp(
                -self.dividend(x) * (self.maturity(x) - self.time_stamp(x))
            ) * (
                1.0 - cdf_d1
            )
        
    # remove the points on the vertical line t = 0 since no initial conditions are needed
    def boundary_sampling(self, point_count):
        x_bound = super().boundary_sampling(point_count)
        return np.delete(x_bound, np.where(self.time_stamp(x_bound) == 0)[0], axis=1)

    def boundary_condition(self, x):
        if self.option_type == "call":
            return np.maximum(
                self.stock_price(x)
                - self.strike_price(x)
                * np.exp(-self.rf_rate(x) * (self.maturity(x) - self.time_stamp(x))),
                0
            )
        elif self.option_type == "put":
            return np.maximum(
                self.strike_price(x)
                * np.exp(-self.rf_rate(x) * (self.maturity(x) - self.time_stamp(x)))
                - self.stock_price(x),
                0
            )

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
    def strike_price(self, x):
        pass

    @abstractmethod
    def rf_rate(self, x):
        pass

    @abstractmethod
    def dividend(self, x):
        pass

    @abstractmethod
    def sigma(self, x):
        pass

    @abstractmethod
    def maturity(self, x):
        pass

    @abstractmethod
    def stock_price(self, x):
        pass

    @abstractmethod
    def time_stamp(self, x):
        pass


class EuroSt(EuropeanOptionsBase):
    input_dim = 2

    def __init__(
        self, strike_price, rf_rate, dividend, sigma, maturity, option_type, network
    ):
        self._strike_price = float(strike_price)
        self._rf_rate = float(rf_rate)
        self._div_yield = float(dividend)
        self._sigma = float(sigma)
        self._maturity = float(maturity)
        self.option_type = option_type

        domain = [(0, 4 * self._strike_price), (0, self._maturity)]
        EuropeanOptionsBase.__init__(self, domain, self.option_type, network)

    def stock_price(self, x):
        return x[0]

    def time_stamp(self, x):
        return x[1]

    def strike_price(self, x):
        return self._strike_price

    def rf_rate(self, x):
        return self._rf_rate

    def dividend(self, x):
        return self._div_yield

    def sigma(self, x):
        return self._sigma

    def maturity(self, x):
        return self._maturity
    


class EuroStK(EuropeanOptionsBase):
    input_dim = 3  # no. of input variables

    def __init__(
        self,
        strike_price_min,
        strike_price_max,
        sigma,
        rf_rate,
        dividend,
        maturity,
        option_type,
        network,
    ):
        self._sigma = float(sigma)
        self._rf_rate = float(rf_rate)
        self._dividend = float(dividend)
        self._maturity = float(maturity)
        self.option_type = option_type
        domain = [
            (0, 4 * strike_price_max),
            (0, maturity),
            (strike_price_min, strike_price_max),
        ]
        EuropeanOptionsBase.__init__(self, domain,  self.option_type, network)

    def sigma(self, x):
        return self._sigma

    def rf_rate(self, x):
        return self._rf_rate

    def dividend(self, x):
        return self._dividend

    def strike_price(self, x):
        return x[2]

    def maturity(self, x):
        return self._maturity

    def stock_price(self, x):
        return x[0]

    def time_stamp(self, x):
        return x[1]

class EuroStSig(EuropeanOptionsBase):
    input_dim = 3

    def __init__(self, strike_price, rf_rate, dividend, maturity, option_type, network):
        self._strike_price = float(strike_price)
        self._rf_rate = float(rf_rate)
        self._dividend = float(dividend)
        self._maturity = float(maturity)
        self.option_type = option_type

        domain = [(0, 4 * self._strike_price), (0, self._maturity), (0.05, 0.75)]
        EuropeanOptionsBase.__init__(self, domain, self.option_type, network)

    def stock_price(self, x):
        return x[0]

    def time_stamp(self, x):
        return x[1]

    def strike_price(self, x):
        return self._strike_price

    def rf_rate(self, x):
        return self._rf_rate

    def dividend(self, x):
        return self._dividend

    def sigma(self, x):
        return x[2]

    def maturity(self, x):
        return self._maturity
