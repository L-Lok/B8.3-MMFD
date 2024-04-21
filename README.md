# Pricing Options in the Black-Scholes Model using Deep Neural Networks
This repository contains the Oxford B8.3 Mathematical Models of Financial Derivatives Special Topic code. The code has been tested in the environment:
* Python $3.10.11$
* TensorFlow-gpu $2.10.0$
* Matplotlib $3.8.3$
* Numpy $1.26.4$
* Pandas $2.2.1$
* QuantLib $1.33$
* Quantsbin $1.0.3$
* scikit-learn $1.4.1$
* scipy $1.12.0$
* tqdm $4.66.2$
# Guidance
This project utilized the unsupervised deep learning approach to price European and American options in the Black-Scholes Model with up to two underlying assets. The code consists of three main parts:
*    `NeuralNetwork.py` provides the foundational neural network elements, such as activation functions and structures.
*    The general PDE setup based on the neural networks in `NeuralNetwork.py` can be found in `PDE.py`, where loss and validation variables are defined.
*    Specific PDEs and associated loss functions are then constructed, given `NeuralNetwork.py` and `PDE.py` -- `European_options.py`, `American_options.py`, and `Two_underlying_asset.py`.
For the other scripts,
*    The models are saved by `model_saver.py`, where the interior and boundary training and validation losses per iteration are tracked.
*    The script `graphing.py` contains the functions for graphing the 2D and 3D model results.
*    The script `Others.py` contains the functions that generate option models such as EuroSt and AmericanStSig and information for plots.
*    The script `add.py` plot the model performance, $L_2$ and $L_{\infty}$ errors, given different numbers of layers.
*    The configuration file `setup.conf` contains the model training and graphing parameters.
In this project, the optimisation algorithm used is L-BFGS. `external_optimizer.py` calls Scipy in the TensorFlow environment to perform optimisation.

# How to Run
The training can be initialised using the terminal command `python main.py`. One can change the model parameters in the `setup.conf` to generate various training results.

# Example Results
The example results are in the `American`, `European`, and `ExchangeOption` folders.

# References
[1] van der Meer, R., Oosterlee, C., & Borovykh, A. (2020). Optimally weighted loss functions for solving pdes with neural networks. arXiv preprint arXiv:2002.06269.
https://github.com/remcovandermeer/Optimally-Weighted-PINNs
[2] https://github.com/pooyasf/DGM/tree/main
[3] Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations
https://github.com/maziarraissi/PINNs
