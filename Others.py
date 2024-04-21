import configparser
from European_options import EuroSt, EuroStK, EuroStSig
from American_options import AmericanSt, AmericanStK, AmericanStSig
from Two_underlying_asset import ExchangeOptionSt
from NeuralNetwork import NeuralNetwork
import ast

conf_file_ = "setup.conf"


def get_sections(model_name):
    return [
        f"{model_name}.TrainingParams",
        f"{model_name}.ModelParams",
        f"{model_name}.Ranges",
        f"{model_name}.GraphingParams3D",
        f"{model_name}.GraphingParams2D",
    ]


def get_params(model_name, file=conf_file_):
    config = configparser.ConfigParser()
    config.read(file)
    params_dict = {}
    for section in get_sections(model_name):
        params_dict[section] = {}
        for option in config.options(section):
            params_dict[section][option] = config.get(section, option)

    return params_dict


def make_title(model_name, dimension="2d"):
    params_dict = get_params(model_name=model_name)

    rf_rate = float(params_dict[f"{model_name}.ModelParams"]["rf_rate"])
    div_yield = float(params_dict[f"{model_name}.ModelParams"]["dividend"])
    maturity = float(params_dict[f"{model_name}.ModelParams"]["maturity"])
    strike_price = float(params_dict[f"{model_name}.ModelParams"]["strike_price"])
    sigma = float(params_dict[f"{model_name}.ModelParams"]["sigma"])
    option_type = params_dict[f"{model_name}.ModelParams"]["option_type"]
    iterations = int(params_dict[f"{model_name}.TrainingParams"]["iterations"])
    train_mode = params_dict[f"{model_name}.TrainingParams"]["train_mode"]

    title = None
    if model_name == "EuroStK":
        stock_price, _, strike_price = ast.literal_eval(
            params_dict[f"{model_name}.GraphingParams2D"]["domain_2d"]
        )
        ################################### fixed K ##################################
        if dimension == "2d" :
            if isinstance(stock_price, tuple):
                title = (
                    rf"{option_type} option, $t$=0.0, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                    + " \n"
                    + f" ({train_mode}, iterations={iterations})"
                )
            else:
                title = (
                rf"{option_type} option, $t$=0.0, $S$={stock_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                + " \n"
                + f" ({train_mode}, iterations={iterations})"
                )
        elif dimension == "3d":
            if isinstance(stock_price, tuple):
                title = (
                    rf"{option_type} option, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                    + "\n"
                    + f" {train_mode}, iterations={iterations}"
                )
            else:
                title = (
                rf"{option_type} option, $S$={stock_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                + "\n"
                + f" {train_mode}, iterations={iterations}"
            )
    ####################### Model EuroStSig ####################
    elif model_name == "EuroStSig":
        stock_price, _, sigma = ast.literal_eval(
            params_dict[f"{model_name}.GraphingParams2D"]["domain_2d"]
        )
        if dimension == "2d" :
            if isinstance(sigma, tuple):
                title = (
                    rf"{option_type} option, $t$=0.0, $S = K$ = {strike_price}, $r$={rf_rate}, $q$={div_yield}, $T$={maturity}"
                    + " \n"
                    + f" ({train_mode}, iterations={iterations})"
                )
            else:
                title = (
                rf"{option_type} option, $t$=0.0, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                + " \n"
                + f" ({train_mode}, iterations={iterations})"
                )
        elif dimension == "3d":
            if isinstance(stock_price, tuple):
                title = (
                    rf"{option_type} option, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                    + "\n"
                    + f" {train_mode}, iterations={iterations}"
                )
            else:
                title = (
                    rf"{option_type} option, $S = K $={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                    + "\n"
                    + f" {train_mode}, iterations={iterations}"
                )        
    ################################## model EuroSt ################################
    elif model_name == "EuroSt":
        if dimension == "2d":
            title = (
                rf"{option_type} option, $t$=0.0, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                + " \n"
                + f" ({train_mode}, iterations={iterations})"
            )
        elif dimension == "3d":
            title = (
                rf"{option_type} option, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                + "\n"
                + f" {train_mode}, iterations={iterations}"
            )
    ################################## model AmericanSt ##########################
    elif model_name == 'AmericanSt':
        if dimension == "2d":
            title = (
                rf"{option_type} option, $t$=0.0, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                + " \n"
                + f" ({train_mode}, iterations={iterations})"
            )
        elif dimension == "3d":
            title = (
                rf"{option_type} option, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                + "\n"
                + f" {train_mode}, iterations={iterations}"
            )
    ################################ model AmericanStK ###########################
    elif model_name == "AmericanStK":
        stock_price, _, strike_price = ast.literal_eval(
            params_dict[f"{model_name}.GraphingParams2D"]["prediction_ranges"]
        )
        if dimension == "2d":
            if isinstance(stock_price, str):
                title = (
                    rf"{option_type} option, $t$=0.0, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                    + " \n"
                    + f" ({train_mode}, iterations={iterations})"
                )
            # elif isinstance(strike_price, tuple):
            else:
                title = (
                    rf"{option_type} option, $t$=0.0, $S$={stock_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                    + " \n"
                    + f" ({train_mode}, iterations={iterations})"
                )
        elif dimension == "3d":
            if isinstance(stock_price, tuple):
                title = (
                    rf"{option_type} option, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                    + "\n"
                    + f" {train_mode}, iterations={iterations}"
                )
            else:
                title = (
                rf"{option_type} option, $S$={stock_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                + "\n"
                + f" {train_mode}, iterations={iterations}"
            )
    elif model_name == "AmericanStSig":
        stock_price, _, sigma = ast.literal_eval(
            params_dict[f"{model_name}.GraphingParams2D"]["prediction_ranges"]
        )
        if dimension == "2d":
            if isinstance(stock_price, str):
                title = (
                    rf"{option_type} option, $t$=0.0, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                    + " \n"
                    + f" ({train_mode}, iterations={iterations})"
                )
            # elif isinstance(strike_price, tuple):
            else:
                title = (
                    rf"{option_type} option, $t$=0.0, $S = K$ = {strike_price}, $r$={rf_rate}, $q$={div_yield}, $T$={maturity}"
                    + " \n"
                    + f" ({train_mode}, iterations={iterations})"
                )
        elif dimension == "3d":
            if isinstance(stock_price, tuple):
                title = (
                rf"{option_type} option, $S = K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $T$={maturity}"
                + "\n"
                + f" {train_mode}, iterations={iterations}"
            )
            else:
                title = (
                    rf"{option_type} option, $K$={strike_price}, $r$={rf_rate}, $q$={div_yield}, $\sigma$={sigma}, $T$={maturity}"
                    + "\n"
                    + f" {train_mode}, iterations={iterations}"
                )
    return title

def make_title_exchange(model_name, dimension="2d"):
    params_dict = get_params(model_name=model_name)

    rf_rate = float(params_dict[f"{model_name}.ModelParams"]["rf_rate"])
    q1= float(params_dict[f"{model_name}.ModelParams"]["dividend1"])
    q2 = float(params_dict[f"{model_name}.ModelParams"]["dividend2"])
    maturity = float(params_dict[f"{model_name}.ModelParams"]["maturity"])
    strike_price = float(params_dict[f"{model_name}.ModelParams"]["strike_price"])
    sigma1 = float(params_dict[f"{model_name}.ModelParams"]["sigma1"])
    sigma2 = float(params_dict[f"{model_name}.ModelParams"]["sigma2"])
    rho = params_dict[f"{model_name}.ModelParams"]["rho"]
    iterations = int(params_dict[f"{model_name}.TrainingParams"]["iterations"])
    train_mode = params_dict[f"{model_name}.TrainingParams"]["train_mode"]

    title = None
 
    S1, S2, _ = ast.literal_eval(
        params_dict[f"{model_name}.GraphingParams2D"]["domain_2d"]
    )
    if dimension == '2d':
        if isinstance(S1, tuple):
            title = (
                    rf"$K$={strike_price}, $r$={rf_rate}, $q$={q1}, $\sigma$={sigma1}, $T$={maturity}, $\rho$ = {rho}"
                    + "\n"
                    + f" $S_2$ = {S2}, {train_mode}, iterations={iterations}"
                )
        else:
            title = (
                    rf"$K$={strike_price}, $r$={rf_rate}, $q$={q1}, $\sigma$={sigma1}, $T$={maturity}, $\rho$ = {rho}"
                    + "\n"
                    + f" $S_1$ = {S1}, {train_mode}, iterations={iterations}"
                )
    elif dimension == '3d':
        title = (
                    rf"$K$={strike_price}, $r$={rf_rate}, $q$={q2}, $\sigma$={sigma1}, $T$={maturity}, $\rho$ = {rho}"
                    + "\n"
                    + f"{train_mode}, iterations={iterations}"
                )
    return title


#################################### Create Model #################################
def create_model(model_name):
    params_dict = get_params(model_name=model_name)

    if model_name == "EuroSt":
        input_dim = EuroSt.input_dim  # 2

        # Training Parameters
        nodes_per_layer = int(
            params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
        )
        layers = int(params_dict[f"{model_name}.TrainingParams"]["layers"])

        # Model Parameters
        rf_rate = float(params_dict[f"{model_name}.ModelParams"]["rf_rate"])
        div_yield = float(params_dict[f"{model_name}.ModelParams"]["dividend"])
        maturity = float(params_dict[f"{model_name}.ModelParams"]["maturity"])
        strike_price = float(params_dict[f"{model_name}.ModelParams"]["strike_price"])
        sigma = float(params_dict[f"{model_name}.ModelParams"]["sigma"])
        option_type = params_dict[f"{model_name}.ModelParams"]["option_type"]

        network = NeuralNetwork(
            input_dimension=input_dim, hidden_layers=[nodes_per_layer] * layers
        )

        return EuroSt(
            strike_price=strike_price,
            sigma=sigma,
            rf_rate=rf_rate,
            dividend=div_yield,
            maturity=maturity,
            option_type=option_type,
            network=network,
        )
    elif model_name == "EuroStK":
        input_dim = EuroStK.input_dim  # 3

        # Training Parameters
        nodes_per_layer = int(
            params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
        )
        layers = int(params_dict[f"{model_name}.TrainingParams"]["layers"])

        # Model Parameters
        rf_rate = float(params_dict[f"{model_name}.ModelParams"]["rf_rate"])
        div_yield = float(params_dict[f"{model_name}.ModelParams"]["dividend"])
        maturity = float(params_dict[f"{model_name}.ModelParams"]["maturity"])
        sigma = float(params_dict[f"{model_name}.ModelParams"]["sigma"])
        option_type = params_dict[f"{model_name}.ModelParams"]["option_type"]
        strike_price_min = float(
            params_dict[f"{model_name}.ModelParams"]["strike_price_min"]
        )
        strike_price = float(params_dict[f"{model_name}.ModelParams"]["strike_price"])

        network = NeuralNetwork(
            input_dimension=input_dim, hidden_layers=[nodes_per_layer] * layers
        )

        return EuroStK(
            strike_price_min=strike_price_min,
            strike_price_max=strike_price,
            sigma=sigma,
            rf_rate=rf_rate,
            dividend=div_yield,
            maturity=maturity,
            option_type=option_type,
            network=network,
        )
    
    elif model_name == "EuroStSig":
        input_dim = EuroStSig.input_dim  # 3

        # Training Parameters
        nodes_per_layer = int(
            params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
        )
        layers = int(params_dict[f"{model_name}.TrainingParams"]["layers"])

        # Model Parameters
        rf_rate = float(params_dict[f"{model_name}.ModelParams"]["rf_rate"])
        div_yield = float(params_dict[f"{model_name}.ModelParams"]["dividend"])
        maturity = float(params_dict[f"{model_name}.ModelParams"]["maturity"])
        option_type = params_dict[f"{model_name}.ModelParams"]["option_type"]

        strike_price = float(params_dict[f"{model_name}.ModelParams"]["strike_price"])

        network = NeuralNetwork(
            input_dimension=input_dim, hidden_layers=[nodes_per_layer] * layers
        )

        return EuroStSig(
            strike_price=strike_price,
            rf_rate=rf_rate,
            dividend=div_yield,
            maturity=maturity,
            option_type=option_type,
            network=network,
        )
    
    elif model_name == "AmericanSt":
        input_dim = AmericanSt.input_dim  # 2

        # Training Parameters
        nodes_per_layer = int(
            params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
        )
        layers = int(params_dict[f"{model_name}.TrainingParams"]["layers"])

        # Model Parameters
        rf_rate = float(params_dict[f"{model_name}.ModelParams"]["rf_rate"])
        div_yield = float(params_dict[f"{model_name}.ModelParams"]["dividend"])
        maturity = float(params_dict[f"{model_name}.ModelParams"]["maturity"])
        strike_price = float(params_dict[f"{model_name}.ModelParams"]["strike_price"])
        sigma = float(params_dict[f"{model_name}.ModelParams"]["sigma"])
        option_type = params_dict[f"{model_name}.ModelParams"]["option_type"]

        network = NeuralNetwork(
            input_dimension=input_dim, hidden_layers=[nodes_per_layer] * layers
        )

        return AmericanSt(
            strike_price=strike_price,
            sigma=sigma,
            rf_rate=rf_rate,
            dividend=div_yield,
            maturity=maturity,
            option_type=option_type,
            network=network,
        )
    
    elif model_name == "AmericanStSig":
        input_dim = AmericanStSig.input_dim  # 3

        # Training Parameters
        nodes_per_layer = int(
            params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
        )
        layers = int(params_dict[f"{model_name}.TrainingParams"]["layers"])

        # Model Parameters
        rf_rate = float(params_dict[f"{model_name}.ModelParams"]["rf_rate"])
        div_yield = float(params_dict[f"{model_name}.ModelParams"]["dividend"])
        maturity = float(params_dict[f"{model_name}.ModelParams"]["maturity"])
        strike_price = float(params_dict[f"{model_name}.ModelParams"]["strike_price"])
        
        option_type = params_dict[f"{model_name}.ModelParams"]["option_type"]

        network = NeuralNetwork(
            input_dimension=input_dim, hidden_layers=[nodes_per_layer] * layers
        )

        return AmericanStSig(
            strike_price=strike_price,
            rf_rate=rf_rate,
            dividend=div_yield,
            maturity=maturity,
            option_type=option_type,
            network=network,
        )
    elif model_name == "AmericanStK":
        input_dim = AmericanStK.input_dim  # 2

        # Training Parameters
        nodes_per_layer = int(
            params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
        )
        layers = int(params_dict[f"{model_name}.TrainingParams"]["layers"])

        # Model Parameters
        rf_rate = float(params_dict[f"{model_name}.ModelParams"]["rf_rate"])
        div_yield = float(params_dict[f"{model_name}.ModelParams"]["dividend"])
        maturity = float(params_dict[f"{model_name}.ModelParams"]["maturity"])
        strike_price = float(params_dict[f"{model_name}.ModelParams"]["strike_price"])
        sigma = float(params_dict[f"{model_name}.ModelParams"]["sigma"])
        option_type = params_dict[f"{model_name}.ModelParams"]["option_type"]

        network = NeuralNetwork(
            input_dimension=input_dim, hidden_layers=[nodes_per_layer] * layers
        )

        return AmericanStK(
            strike_price=strike_price,
            sigma=sigma,
            rf_rate=rf_rate,
            dividend=div_yield,
            maturity=maturity,
            option_type=option_type,
            network=network,
        )
        
    elif model_name == 'ExchangeOptionSt':
        input_dim = ExchangeOptionSt.input_dim  # 2

        # Training Parameters
        nodes_per_layer = int(
            params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
        )
        layers = int(params_dict[f"{model_name}.TrainingParams"]["layers"])

        # Model Parameters
        rf_rate = float(params_dict[f"{model_name}.ModelParams"]["rf_rate"])
        div_yield1 = float(params_dict[f"{model_name}.ModelParams"]["dividend1"])
        div_yield2 = float(params_dict[f"{model_name}.ModelParams"]["dividend2"])
        maturity = float(params_dict[f"{model_name}.ModelParams"]["maturity"])
        strike_price = float(params_dict[f"{model_name}.ModelParams"]["strike_price"])
        sigma1 = float(params_dict[f"{model_name}.ModelParams"]["sigma1"])
        sigma2 = float(params_dict[f"{model_name}.ModelParams"]["sigma2"])
        rho = float(params_dict[f"{model_name}.ModelParams"]["rho"])

        network = NeuralNetwork(
            input_dimension=input_dim, hidden_layers=[nodes_per_layer] * layers
        )

        return ExchangeOptionSt(strike_price=strike_price,
                                rf_rate=rf_rate,
                                dividend_1=div_yield1,
                                dividend_2=div_yield2,
                                sigma_1=sigma1,
                                sigma_2=sigma2,
                                rho=rho,
                                maturity=maturity,
                                network=network)



