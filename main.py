import ast
import numpy as np
from model_saver import Model_Save, make_path
from Others import get_params, create_model, make_title, make_title_exchange
from graphing import (
    Euro_plot,
    create_prices_df,
    get_prediction_date,
    American_plot,
    Exchange_plot,
    surface_plot,
    avg_losses_against_iterations,
)
from American_options import BinomialAmericanPricer
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()


########################## Training ##########################
def start_training(model_name, option_name="European"):

    params_dict = get_params(model_name=model_name)

    iterations = int(params_dict[f"{model_name}.TrainingParams"]["iterations"])
    trainMode = params_dict[f"{model_name}.TrainingParams"]["train_mode"]
    nodes_per_layers = params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
    layers = params_dict[f"{model_name}.TrainingParams"]["layers"]
    option_type = None
    if option_name != 'ExchangeOption':
        option_type = get_params(model_name)[f"{model_name}.ModelParams"]["option_type"]

    model = create_model(model_name)

    model.train(train_mode=trainMode, iterations=iterations)

    # Create folders
    make_path(option_name=option_name, model_name=model_name)

    # save the weights
    model.network.save_weights(
        f"{option_name}/{model_name}/weights NPL = {nodes_per_layers}X{layers}/"
    )

    l2_error = model.compute_l2_error()
    max_error = model.compute_max_error()

    # save the history (losses, etc.)
    if option_name == 'ExchangeOption':
        Model_Save.save_history(
            model.history,
            l2_error,
            max_error,
            f"{option_name}/{model_name}/histories/{model_name}{iterations} NPL = {nodes_per_layers}X{layers}.csv",
        )
    else:
        Model_Save.save_history(
        model.history,
        l2_error,
        max_error,
        f"{option_name}/{model_name}/histories/{model_name}{option_type}{iterations} NPL = {nodes_per_layers}X{layers}.csv",
    )
    model.network.cleanup()


################################## Plotting ########################################
# losses
def Plot_loss(option_name, model_name):
    params_dict = get_params(model_name)

    # training params
    nodes_per_layers = params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
    layers = params_dict[f"{model_name}.TrainingParams"]["layers"]

    # Graphing Parameters from configuration file
    option_type = ''
    if option_name == 'ExchangeOption':
        option_type = None
    else:
        option_type = get_params(model_name)[f"{model_name}.ModelParams"]["option_type"]
    
    iterations = get_params(model_name)[f"{model_name}.TrainingParams"]["iterations"]

    avg_losses_against_iterations(
        option_name=option_name,
        model_name=model_name,
        option_type=option_type,
        iterations=iterations,
        npl=nodes_per_layers,
        Layers=layers,
        window= 200
    )


# European Plots
def Graphing_Euro(option_name, model_name):
    # Config file
    params_dict = get_params(model_name)

    # training params
    nodes_per_layers = params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
    layers = params_dict[f"{model_name}.TrainingParams"]["layers"]

    model = create_model(model_name)
    model.network.load_weights(
        f"{option_name}/{model_name}/weights NPL = {nodes_per_layers}X{layers}/"
    )

    # Graphing Parameters from configuration file
    option_type = get_params(model_name)[f"{model_name}.ModelParams"]["option_type"]
    iterations = get_params(model_name)[f"{model_name}.TrainingParams"]["iterations"]
    ########################## 2D params #######################
    title_2d = make_title(model_name=model_name, dimension="2d")
    x_label_2d = params_dict[f"{model_name}.GraphingParams2D"]["x_label_2d"]
    domain_2d = ast.literal_eval(
        params_dict[f"{model_name}.GraphingParams2D"]["domain_2d"]
    )
    ######################## 3D params ##########################
    domain_3d = ast.literal_eval(
        params_dict[f"{model_name}.GraphingParams3D"]["domain_3d"]
    )
    x_label_3d = params_dict[f"{model_name}.GraphingParams3D"]["x_label_3d"]
    y_label_3d = params_dict[f"{model_name}.GraphingParams3D"]["y_label_3d"]
    title_3d = make_title(model_name=model_name, dimension="3d")

    Euro_plot(
        model_name=model_name,
        option_name=option_name,
        model=model,
        domain=domain_2d,
        option_type=option_type,
        iterations=iterations,
        title=title_2d,
        x_label=x_label_2d,
    )

    surface_plot(
        model_name=model_name,
        option_name=option_name,
        model=model,
        domain=domain_3d,
        option_type=option_type,
        iterations=iterations,
        x_label=x_label_3d,
        y_label=y_label_3d,
        title=title_3d,
    )

def American_BTM_Prices(model_name):
    # Config file
    params_dict = get_params(model_name)

    # set up the params DataFrame for prices
    strike_price_range = ast.literal_eval(
        params_dict[f"{model_name}.Ranges"]["strike_price_range"]
    )
    stock_price_range = ast.literal_eval(
        params_dict[f"{model_name}.Ranges"]["stock_price_range"]
    )
    rf_rate_range = ast.literal_eval(
        params_dict[f"{model_name}.Ranges"]["rf_rate_range"]
    )
    dividend_range = ast.literal_eval(
        params_dict[f"{model_name}.Ranges"]["dividend_range"]
    )
    sigma_range = ast.literal_eval(params_dict[f"{model_name}.Ranges"]["sigma_range"])

    samples = 70

    steps = ast.literal_eval(params_dict[f"{model_name}.Ranges"]["steps"])

    prices = []
    t = np.linspace(0.0, 365, samples)
    for t_ in t:
        ranges = [
            strike_price_range,
            stock_price_range,
            rf_rate_range,
            365 - t_,
            dividend_range,
            sigma_range,
        ]
        prices_df = create_prices_df(ranges, samples)
        # Graphing Parameters from configuration file
        option_type = params_dict[f"{model_name}.ModelParams"]["option_type"]
        # Use Binominal Tree Model to obtain prices numerically
        numerical_prices = BinomialAmericanPricer(
            option_type=option_type, steps=steps
        ).price(df=prices_df, use_tqdm=True)
        prices.append(numerical_prices)
    return prices

# American Plots
def Graphing_American(option_name, model_name):
    # Config file
    params_dict = get_params(model_name)

    # set up the params DataFrame for prices
    strike_price_range = ast.literal_eval(
        params_dict[f"{model_name}.Ranges"]["strike_price_range"]
    )
    stock_price_range = ast.literal_eval(
        params_dict[f"{model_name}.Ranges"]["stock_price_range"]
    )
    rf_rate_range = ast.literal_eval(
        params_dict[f"{model_name}.Ranges"]["rf_rate_range"]
    )
    days_to_maturity = ast.literal_eval(
        params_dict[f"{model_name}.Ranges"]["days_to_maturity_range"]
    )
    dividend_range = ast.literal_eval(
        params_dict[f"{model_name}.Ranges"]["dividend_range"]
    )
    sigma_range = ast.literal_eval(params_dict[f"{model_name}.Ranges"]["sigma_range"])

    samples = ast.literal_eval(params_dict[f"{model_name}.Ranges"]["samples"])
    prediction_ranges = ast.literal_eval(
        params_dict[f"{model_name}.GraphingParams2D"]["prediction_ranges"]
    )
    steps = ast.literal_eval(params_dict[f"{model_name}.Ranges"]["steps"])

    ranges = [
        strike_price_range,
        stock_price_range,
        rf_rate_range,
        days_to_maturity,
        dividend_range,
        sigma_range,
    ]

    prices_df = create_prices_df(ranges, samples)

    data, index = get_prediction_date(prices_df, prediction_ranges, samples)

    # training params
    nodes_per_layers = params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
    layers = params_dict[f"{model_name}.TrainingParams"]["layers"]

    model = create_model(model_name)
    model.network.load_weights(
        f"{option_name}/{model_name}/weights NPL = {nodes_per_layers}X{layers}/"
    )

    # Graphing Parameters from configuration file
    option_type = params_dict[f"{model_name}.ModelParams"]["option_type"]
    iterations = params_dict[f"{model_name}.TrainingParams"]["iterations"]
    title_2d = make_title(model_name=model_name, dimension="2d")
    x_label_2d = params_dict[f"{model_name}.GraphingParams2D"]["x_label_2d"]
    domain_2d = prediction_ranges

    # Use Binominal Tree Model to obtain prices numerically
    numerical_prices = BinomialAmericanPricer(
        option_type=option_type, steps=steps
    ).price(df=prices_df, use_tqdm=True)

    American_plot(
        model_name=model_name,
        option_name=option_name,
        model=model,
        data=data,
        index=index,
        numerical_prices=numerical_prices,
        domain=domain_2d,
        option_type=option_type,
        iterations=iterations,
        title=title_2d,
        x_label=x_label_2d,
    )

    ################################# 3D params ######################
    title_3d = make_title(model_name=model_name, dimension="3d")
    x_label_3d = params_dict[f"{model_name}.GraphingParams3D"]["x_label_3d"]
    y_label_3d = params_dict[f"{model_name}.GraphingParams3D"]["y_label_3d"]
    domain_3d = ast.literal_eval(params_dict[f"{model_name}.GraphingParams3D"]["domain_3d"])

    surface_plot(
        model_name=model_name,
        option_name=option_name,
        model=model,
        domain=domain_3d,
        option_type=option_type,
        iterations=iterations,
        x_label = x_label_3d,
        y_label=y_label_3d,
        title=title_3d,
        American_BTM_Prices=American_BTM_Prices(model_name),
        is_difference=False
    )

#################################### Exchange Plot ##############################


def Graphing_Exchange(option_name, model_name):
    # Config file
    params_dict = get_params(model_name)

    # training params
    nodes_per_layers = params_dict[f"{model_name}.TrainingParams"]["nodes_per_layer"]
    layers = params_dict[f"{model_name}.TrainingParams"]["layers"]

    model = create_model(model_name)
    model.network.load_weights(
        f"{option_name}/{model_name}/weights NPL = {nodes_per_layers}X{layers}/"
    )

    # Graphing Parameters from configuration file
    iterations = get_params(model_name)[f"{model_name}.TrainingParams"]["iterations"]
    ############################### 2D params #########################
    title_2d = make_title_exchange(model_name=model_name, dimension="2d")
    x_label_2d = params_dict[f"{model_name}.GraphingParams2D"]["x_label_2d"]
    domain_2d = ast.literal_eval(
        params_dict[f"{model_name}.GraphingParams2D"]["domain_2d"]
    )

    Exchange_plot(
        model_name=model_name,
        option_name=option_name,
        model=model,
        domain=domain_2d,
        iterations=iterations,
        title=title_2d,
        x_label=x_label_2d,
    )
    ######################## 3D params ##########################
    domain_3d = ast.literal_eval(
        params_dict[f"{model_name}.GraphingParams3D"]["domain_3d"]
    )
    x_label_3d = params_dict[f"{model_name}.GraphingParams3D"]["x_label_3d"]
    y_label_3d = params_dict[f"{model_name}.GraphingParams3D"]["y_label_3d"]
    title_3d = make_title_exchange(model_name=model_name, dimension="3d")

    surface_plot(
        model_name=model_name,
        option_name=option_name,
        model=model,
        domain=domain_3d,
        option_type=None,
        iterations=iterations,
        x_label=x_label_3d,
        y_label=y_label_3d,
        title=title_3d,
    )


if __name__ == "__main__":
    # option_name = "ExchangeOption"
    # model_name = "ExchangeOptionSt"

    option_name = "European"
    model_name = "EuroSt"

    # option_name = "American"
    # model_name = "AmericanStSig"

    ############################ Start Training ##################################
    start_training(model_name, option_name)

    ########################### Loss Plot ########################
    # Plot_loss(option_name=option_name,model_name=model_name)

    ############################ Plotting ########################################
    # Graphing_Exchange(option_name=option_name, model_name=model_name)
    # Graphing_Euro(option_name=option_name, model_name=model_name)
    # Graphing_American(option_name=option_name, model_name=model_name)
