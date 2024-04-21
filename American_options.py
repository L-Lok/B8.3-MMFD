import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import QuantLib as ql
from abc import abstractmethod
from PDE import PDENeuralNetwork
from tqdm import tqdm


class Pricer:
    def price(self, df, use_tqdm=False):
        prices = np.zeros(df.shape[0])

        row_iterator = df.iterrows()
        if use_tqdm:
            row_iterator = tqdm(row_iterator, total=df.shape[0])

        for i, row in row_iterator:
            strike = row["strike_price"]
            sigma = row["sigma"]
            dividend = row["dividend"]
            days_to_maturity = row["days_to_maturity"]
            stock_price = row["stock_price"]
            rf_rate = row["rf_rate"]

            prices[i] = self.price_one(
                strike, sigma, dividend, days_to_maturity, stock_price, rf_rate
            )

        return prices

    def price_one(self, strike, sigma, div, maturity, underlying, rf_rate):
        pass


def bsm(stock_price, rf_rate, dividend, sigma, day_count_conv, calendar, current_date):
    # current price of the underlying asset
    current_price = ql.QuoteHandle(ql.SimpleQuote(stock_price))

    # risk free interest rate term structure
    # FlatForward means unchanged throughout
    rf_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(current_date, rf_rate, day_count_conv)
    )

    # dividend term structure
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(current_date, dividend, day_count_conv)
    )

    # volatility term structure
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(current_date, calendar, sigma, day_count_conv)
    )

    # Return the BSM process
    return ql.BlackScholesMertonProcess(current_price, div_ts, rf_ts, vol_ts)


class BinomialAmericanPricer(Pricer):
    def __init__(self, option_type, steps, current_date=ql.Date(6, 3, 2020)):
        # Pricer.__init__(use_tqdm)
        self.current_date = current_date
        self.option_type = option_type
        self.steps = steps

    def price_one(self, strike, sigma, dividend, maturity, stock_price, rf_rate):
        maturity_date = ql.Date(self.current_date.serialNumber() + int(maturity))

        day_count_conv = ql.Actual365Fixed()
        calendar = ql.UnitedStates(ql.UnitedStates.NYSE)  # UnitedStates

        # Determining the exercising time
        ql.Settings.instance().evaluationDate = self.current_date

        # Assume the underlying process is BSM
        process = bsm(
            stock_price,
            rf_rate,
            dividend,
            sigma,
            day_count_conv,
            calendar,
            self.current_date,
        )

        if self.option_type == "call":
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        elif self.option_type == "put":
            payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)

        exercise = ql.AmericanExercise(self.current_date, maturity_date)

        # Instantiate the VanillaOption
        option = ql.VanillaOption(payoff, exercise)

        binomial_engine = ql.BinomialVanillaEngine(process, "crr", self.steps)

        # set pricing engine
        option.setPricingEngine(binomial_engine)

        # return the theoretical price
        return option.NPV()


class AmericanOptionsBase(PDENeuralNetwork):
    def __init__(self, domain, pricer, option_type, network):
        self.pricer = pricer  # Binomial American Pricer
        self.option_type = option_type

        # initialize PDENeuralNetwork
        PDENeuralNetwork.__init__(self, domain=domain, network=network)

        ############################## Regular loss ########################
        loss_int, loss_bound = self.compute_loss_terms(
            self.network.x_int,
            self.network.y_int,
            self.network.y_bound,
            self.network.boundary_condition,
        )

        self.default_loss = tf.add(loss_int, loss_bound)
        # self.optimal_loss = (self.loss_weight * self.interior_domain_size) * loss_int + \
        #                     ((1 - self.loss_weight) * self.total_boundary_domain_size) * loss_bound
        # self.magnitude_loss = loss_int / magnitude_int + loss_bound / magnitude_bound

        ############################# Validation loss ##########################
        loss_int_validate, loss_bound_validate = self.compute_loss_terms(
            self.network.x_int_validate,
            self.network.y_int_validate,
            self.network.y_bound_validate,
            self.network.boundary_condition_validate,
        )

        self.default_loss_validate = tf.add(loss_int_validate, loss_bound_validate)
        # self.optimal_loss_validate = (self.loss_weight * self.interior_domain_size) * loss_int_validate + \
        #                              ((1 - self.loss_weight) * self.total_boundary_domain_size) * \
        #                              loss_bound_validate
        # self.magnitude_loss_validate = loss_int_validate / magnitude_int_validate \
        #                                + loss_bound_validate / magnitude_bound_validate

        # Create fetch lists
        self.fetch_list = [loss_int, loss_bound]
        self.fetch_list_validate = [
            loss_int,
            loss_bound,
            loss_int_validate,
            loss_bound_validate,
        ]

    def compute_loss_terms(self, x_int, y_int, y_bound, boundary_condition):
        dvds = PDENeuralNetwork.partial_derivative(y_int, self.stock_price(x_int), 1)

        dvdt = PDENeuralNetwork.partial_derivative(y_int, self.time_stamp(x_int), 1)
        d2vds2 = PDENeuralNetwork.partial_derivative(y_int, self.stock_price(x_int), 2)
        L = (
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

        if self.option_type == "call":
            loss_int = tf.reduce_mean(
                tf.square(
                    tf.minimum(
                        -L,
                        y_int
                        - tf.maximum(
                            self.stock_price(x_int) - self.strike_price(x_int), 0
                        ),
                    )
                )
            )
        elif self.option_type == "put":
            loss_int = tf.reduce_mean(
                tf.square(
                    tf.minimum(
                        -L,
                        y_int
                        - tf.maximum(
                            -self.stock_price(x_int) + self.strike_price(x_int), 0
                        ),
                    )
                )
            )
        else:
            loss_int = None

        loss_bound = tf.reduce_mean(tf.square(y_bound - boundary_condition))

        return loss_int, loss_bound

    def boundary_condition(self, x):
        if self.option_type == "call":
            return np.maximum(self.stock_price(x) - self.strike_price(x), 0)
        elif self.option_type == "put":
            return np.maximum(self.strike_price(x) - self.stock_price(x), 0)

    def analytical_solution(self, x):
        df = pd.DataFrame()
        df["stock_price"] = self.stock_price(x)
        df["strike_price"] = self.strike_price(x)
        df["rf_rate"] = self.rf_rate(x)
        df["days_to_maturity"] = self.maturity(x) * 365
        df["dividend"] = self.dividend(x)
        df["sigma"] = self.sigma(x)

        prices = self.pricer.price(df, use_tqdm=True)
        return prices

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


################################## Models ###############################
class AmericanSt(AmericanOptionsBase):
    input_dim = 2

    def __init__(
        self,
        strike_price,
        rf_rate,
        dividend,
        sigma,
        maturity,
        option_type,
        network,
        current_date=ql.Date(6, 3, 2020),
        steps=2000,
    ):
        self._strike_price = float(strike_price)
        self._rf_rate = float(rf_rate)
        self._dividend = float(dividend)
        self._sigma = float(sigma)
        self._maturity = float(maturity)
        self.option_type = option_type
        self.current_date = current_date

        domain = [(0, 4 * self._strike_price), (0, self._maturity)]
        binomial_pricer = BinomialAmericanPricer(
            current_date=self.current_date, option_type=self.option_type, steps=steps
        )
        AmericanOptionsBase.__init__(
            self,
            domain=domain,
            pricer=binomial_pricer,
            option_type=self.option_type,
            network=network,
        )

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
        return self._sigma

    def maturity(self, x):
        return self._maturity


class AmericanStK(AmericanOptionsBase):
    input_dim = 3

    def __init__(
        self,
        strike_price,
        rf_rate,
        dividend,
        sigma,
        maturity,
        option_type,
        network,
        current_date=ql.Date(6, 3, 2020),
        steps=2000,
    ):
        self._strike_price = float(strike_price)
        self._rf_rate = float(rf_rate)
        self._dividend = float(dividend)
        self._sigma = float(sigma)
        self._maturity = float(maturity)
        self.option_type = option_type
        self.current_date = current_date

        domain = [(0, 4 * self._strike_price), (0, self._maturity), (0, 100)]
        binomial_pricer = BinomialAmericanPricer(
            current_date=self.current_date, option_type=self.option_type, steps=steps
        )
        AmericanOptionsBase.__init__(
            self,
            domain=domain,
            pricer=binomial_pricer,
            option_type=self.option_type,
            network=network,
        )

    def stock_price(self, x):
        return x[0]

    def time_stamp(self, x):
        return x[1]

    def strike_price(self, x):
        return x[2]

    def rf_rate(self, x):
        return self._rf_rate

    def dividend(self, x):
        return self._dividend

    def sigma(self, x):
        return self._sigma

    def maturity(self, x):
        return self._maturity


class AmericanStSig(AmericanOptionsBase):
    input_dim = 3

    def __init__(
        self,
        strike_price,
        rf_rate,
        dividend,
        maturity,
        option_type,
        network,
        current_date=ql.Date(6, 3, 2020),
        steps=2000,
    ):
        self._strike_price = float(strike_price)
        self._rf_rate = float(rf_rate)
        self._dividend = float(dividend)
        self._maturity = float(maturity)
        self.option_type = option_type
        self.current_date = current_date

        domain = [(0, 4 * self._strike_price), (0, self._maturity), (0.05, 0.75)]
        
        binomial_pricer = BinomialAmericanPricer(
            current_date=self.current_date, option_type=self.option_type, steps=steps
        )
        AmericanOptionsBase.__init__(
            self,
            domain=domain,
            pricer=binomial_pricer,
            option_type=self.option_type,
            network=network,
        )

    def stock_price(self, x):
        return x[0]

    def time_stamp(self, x):
        return x[1]

    def sigma(self, x):
        return x[2]

    def strike_price(self, x):
        return self._strike_price

    def rf_rate(self, x):
        return self._rf_rate

    def dividend(self, x):
        return self._dividend

    def maturity(self, x):
        return self._maturity
