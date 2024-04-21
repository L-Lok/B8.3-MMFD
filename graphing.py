import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


############################## Losses Plots ###################################
def avg_losses_against_iterations(
    option_name, model_name, option_type, iterations, npl, Layers, window=200
):
    df = None
    if option_name == "ExchangeOption":
        df = pd.read_csv(
            f"{option_name}/{model_name}/histories/{model_name}{iterations} NPL = {npl}X{Layers}.csv"
        )
    else:
        df = pd.read_csv(
            f"{option_name}/{model_name}/histories/{model_name}{option_type}{iterations} NPL = {npl}X{Layers}.csv"
        )

    # avg losses per 200 time steps
    # interior losses - regular and validaiton
    df["avg_int_losses"] = df["interior_loss"].rolling(window=window).mean()
    df["avg_int_validation_losses"] = (
        df["interior_loss_validation"].rolling(window=window).mean()
    )
    # boudnary lossses - interior and validation
    df["avg_boundary_losses"] = df["boundary_loss"].rolling(window=window).mean()
    df["avg_boundary_validation_losses"] = (
        df["boundary_loss_validation"].rolling(window=window).mean()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    ax1.plot(
        df["iterations"],
        df["avg_int_losses"],
        label="Interior Losses",
        color="royalblue",
    )
    ax1.plot(
        df["iterations"],
        df["avg_int_validation_losses"],
        label="Validation loss",
        color="darkorange",
    )
    ax1.set_title("Interior Losses", fontsize=15)
    ax1.set_xlabel("Iteration", fontsize=15)
    ax1.set_ylabel("Log-scaled Interior Loss", fontsize=15)
    ax1.set_yscale("log")
    ax1.grid()
    ax1.legend(fontsize=15)

    ax2.plot(
        df["iterations"],
        df["avg_boundary_losses"],
        label="Boundary Losses",
        color="royalblue",
    )
    ax2.plot(
        df["iterations"],
        df["avg_boundary_validation_losses"],
        label="Validation loss",
        color="darkorange",
    )
    ax2.set_title("Boundary Losses", fontsize=15)
    ax2.set_xlabel("Iteration", fontsize=15)
    ax2.set_ylabel("Log-scaled Boundary Loss", fontsize=15)
    ax2.set_yscale("log")
    ax2.grid()
    ax2.legend(fontsize=15)

    fig.suptitle(
        f"Average Loss per {window} Iterations"
        + "\n"
        + f"Neural Network = {npl}X{Layers}",
        fontsize=20,
    )
    plt.tight_layout()
    if option_name == "ExchangeOption":
        plt.savefig(
            f"{option_name}/{model_name}/NPL = {npl}X{Layers} avg_losses_{window} in {iterations}.jpg"
        )
    else:
        plt.savefig(
            f"{option_name}/{model_name}/NPL = {npl}X{Layers} {option_type} avg_losses_{window} in {iterations}.jpg"
        )
    plt.show()


############################## Surface Plot ###################################
def surface_plot(
    model_name,
    option_name,
    model,
    domain,
    option_type,
    iterations,
    x_label,
    y_label,
    title,
    elevation=18,  # 18
    azim_angle=-53,  # -53
    boundary_plot=False,
    is_difference=False,
    American_BTM_Prices=None,
):  # elevation = 10, azim_angle = 38 for exchange option

    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax = fig.add_subplot(projection="3d")
    x, y, v, v_ana, p = None, None, None, None, None
    if American_BTM_Prices is not None:
        x, y, v = model.get_interior_plot_data(
            point_count=5000,
            tensor=model.network.y_int,
            x=domain,
            option_name=option_name,
        )

        # diff = abs(v -v_ana)
        # num = len(diff[-1])
        # diff[-1] = np.ones(num)*0.001

        if is_difference == True:
            v_ana = np.array([American_BTM_Prices]).reshape(70, 70)
            p = ax.plot_surface(x, y, abs(v - v_ana), cmap="viridis")
        else:
            p = ax.plot_surface(x, y, abs(v), cmap="viridis")
    elif American_BTM_Prices is None:
        x, y, v, v_ana = model.get_interior_plot_data(
            point_count=5000,
            tensor=model.network.y_int,
            x=domain,
            option_name=option_name,
        )

        if is_difference == True:
            p = ax.plot_surface(x, y, abs(v - v_ana), cmap="viridis")
        else:
            p = ax.plot_surface(x, y, abs(v), cmap="viridis")

    if boundary_plot:
        x, y, v = model.get_boundary_plot_data(
            point_count=1000, tensor=model.network.boundary_condition, x=domain
        )
        ax.plot3D(x[0], y[0], v[0], color="orange", label="Boundary Condition")
        for i in range(1, len(x)):
            if np.any(y[i] != 0):
                ax.plot3D(x[i], y[i], v[i], color="orange")
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.set_zlabel("V", fontsize=15)
    ax.set_title(title, fontsize=15)
    ax.view_init(elev=elevation, azim=azim_angle)
    cax = fig.add_axes([0.35, 0.10, 0.40, 0.03])
    fig.add_axes(cax)

    fig.colorbar(p, ax=ax, orientation="horizontal", cax=cax)
    if model_name == "ExchangeOptionSt":
        plt.savefig(f"{option_name}/{model_name}/3d iterations = {iterations}.jpg")
    else:
        plt.savefig(
            f"{option_name}/{model_name}/{option_type} 3d iterations = {iterations}.jpg"
        )

    plt.show()


#################################### 2D Plots ##################################


# European Options
def Euro_plot(
    model_name,
    option_name,
    model,
    domain,
    option_type,
    iterations,
    title,
    x_label,
    y_label=rf"$V$",
):
    # figures
    model_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x_pre, v_pre = model.get_predicted_plot_data(domain)
    x_ana, v_ana = model.get_analytical_plot_data(domain)
    ax1.plot(x_pre, v_pre, label="Predicted Solution", color="blue")
    ax1.plot(
        x_ana, v_ana, label="Analytical Solution", linestyle="dashed", color="orange"
    )
    # if model_name != 'EuroStSig':
    S, t, K = None, None, None
    if model_name == "EuroStK":
        S, t, K = domain

    intrinsic_val = None
    ##################### fixed K ######################
    if isinstance(S, tuple):
        if option_type == "put":
            intrinsic_val = np.maximum(K - x_pre, 0)
        elif option_type == "call":
            intrinsic_val = np.maximum(x_pre - K, 0)
    #################### fixed S ########################
    elif isinstance(K, tuple):
        if option_type == "put":
            intrinsic_val = np.maximum(x_pre - S, 0)
        elif option_type == "call":
            intrinsic_val = np.maximum(S - x_pre, 0)
    ################### Other Models ######################
    else:
        if option_type == "put":
            intrinsic_val = np.maximum(model.strike_price(None) - x_pre, 0)
        elif option_type == "call":
            intrinsic_val = np.maximum(x_pre - model.strike_price(None), 0)

    ax1.plot(
        x_pre,
        intrinsic_val,
        label="Payoff",
        linestyle="dotted",
        color="green",
    )
    ax1.set_title(rf"{title}", fontsize=15)
    ax1.set_xlabel(rf"{x_label}", fontsize=15)
    ax1.set_ylabel(y_label, fontsize=15)
    ax1.grid(True)
    ax1.legend(fontsize=20)

    ax2.plot(x_ana, np.abs(v_ana - v_pre), color="blue")
    ax2.set_title("Differences From the Analytical Solution", fontsize=17)
    ax2.set_xlabel(rf"{x_label}", fontsize=15)
    ax2.set_ylabel(y_label, fontsize=15)
    ax2.grid(True)

    # model_fig.suptitle(f"Model {model_name}")

    plt.show()

    model_fig.savefig(
        f"{option_name}/{model_name}/{option_type} iterations = {iterations}.jpg"
    )


######################### Prepare Graphing American Options #####################
def create_prices_df(values, samples, columns=None, seed=42):
    if columns is None:
        # columns = [STRIKE_COL, UNDERLYING_COL, RF_RATE_COL, DAYS_TO_MATURITY_COL, DIV_COL, SIGMA_COL]
        columns = [
            "strike_price",
            "stock_price",
            "rf_rate",
            "days_to_maturity",
            "dividend",
            "sigma",
        ]
    df = pd.DataFrame()
    for val_range, col in zip(values, columns):
        np.random.seed(seed + 1)
        if isinstance(val_range, tuple):
            df[col] = np.linspace(*val_range, num=samples)
        else:
            df[col] = [val_range] * samples

    return df


def get_prediction_date(prices_df, prediction_ranges, samples):
    """
    prices_df:      the DataFrame that contains lists of values of each params
    prediction_ranges: values of params for plotting
    samples:        the number of data points in the range
    """
    data = []
    idx = 0
    for i, item in enumerate(prediction_ranges):
        if isinstance(item, str):
            data.append(np.array([prices_df[item]]).transpose())
            idx = i
        else:
            data.append(np.full((samples, 1), item))
    return np.array(data), idx


############################### American Options Plotting #########################


def American_plot(
    model_name,
    option_name,
    model,
    data,
    index,
    numerical_prices,
    domain,
    option_type,
    iterations,
    title,
    x_label,
    y_label=rf"$V$",
):

    # figures
    model_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    v_pre = model.network.predict(data)
    x_pre = data[index].transpose()[0]
    x_ana = np.copy(x_pre)
    ax1.plot(x_pre, v_pre, label="Predicted Solution", color="blue")
    ax1.plot(
        x_ana,
        numerical_prices,
        label="Binomial Tree Model Solution",
        linestyle="dashed",
        color="orange",
    )

    S, t, K = None, None, None
    if model_name == "AmericanStK":
        S, t, K = domain

    intrinsic_val = None
    ##################### fixed K ######################
    if isinstance(S, str):
        if option_type == "put":
            intrinsic_val = np.maximum(K - x_pre, 0)
        elif option_type == "call":
            intrinsic_val = np.maximum(x_pre - K, 0)
    #################### fixed S ########################
    elif isinstance(K, str):
        if option_type == "put":
            intrinsic_val = np.maximum(x_pre - S, 0)
        elif option_type == "call":
            intrinsic_val = np.maximum(S - x_pre, 0)
    ################### Other Models ######################
    else:
        if option_type == "put":
            intrinsic_val = np.maximum(model.strike_price(None) - x_pre, 0)
        elif option_type == "call":
            intrinsic_val = np.maximum(x_pre - model.strike_price(None), 0)

    ax1.plot(
        x_pre,
        intrinsic_val,
        label="Payoff",
        linestyle="dotted",
        color="green",
    )
    ax1.set_title(rf"{title}", fontsize=15)
    ax1.set_xlabel(rf"{x_label}", fontsize=15)
    ax1.set_ylabel(y_label, fontsize=15)
    ax1.grid(True)
    ax1.legend(fontsize=15)

    ax2.plot(x_ana, abs(v_pre.transpose()[0] - numerical_prices), color="royalblue")
    ax2.set_title("Price Differences from the Binomial Tree Model", fontsize=15)
    ax2.set_xlabel(rf"{x_label}", fontsize=15)
    ax2.set_ylabel(y_label, fontsize=15)
    ax2.grid(True)

    # model_fig.suptitle(f"Model {model_name}")

    plt.show()

    model_fig.savefig(
        f"{option_name}/{model_name}/{option_type} iterations = {iterations}.jpg"
    )


################################## Exchange Plot ################################


def Exchange_plot(
    model_name,
    option_name,
    model,
    domain,
    iterations,
    title,
    x_label,
    y_label=rf"$V$",
):
    # figures
    model_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    x_pre, v_pre = model.get_predicted_plot_data(domain)
    x_ana, v_ana = model.get_analytical_plot_data(domain)
    ax1.plot(x_pre, v_pre, label="Predicted Solution", color="blue")
    ax1.plot(
        x_ana, v_ana, label="Analytical Solution", linestyle="dashed", color="orange"
    )

    intrinsic_val = None
    intrinsic_val = np.maximum(x_pre - model.stock_price_2(domain), 0)

    ax1.plot(
        x_pre,
        intrinsic_val,
        label="Payoff",
        linestyle="dotted",
        color="green",
    )
    ax1.set_title(rf"{title}", fontsize=20)
    ax1.set_xlabel(rf"{x_label}", fontsize=15)
    ax1.set_ylabel(y_label, fontsize=15)
    ax1.grid(True)
    ax1.legend(fontsize=15)

    ax2.plot(x_ana, np.abs(v_ana - v_pre), color="blue")
    ax2.set_title("Differences From the Analytical Solution", fontsize=17)
    ax2.set_xlabel(rf"{x_label}", fontsize=15)
    ax2.set_ylabel(y_label, fontsize=15)
    ax2.grid(True)

    # model_fig.suptitle(f"Model {model_name}")

    plt.show()

    model_fig.savefig(
        f"{option_name}/{model_name}/exchange iterations = {iterations}.jpg"
    )
