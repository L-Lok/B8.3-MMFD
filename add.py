import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.figure(figsize = (10, 8))
    txt = np.loadtxt('layers.txt')
    L2 = txt[:, 1]
    L_inf = txt[:, -1]
    layers = txt[:,0]
    plt.plot(layers, L2, label = r"$L_2^{rel}$ error")
    plt.plot(layers, L_inf, label = r"$L_{\inf}^{rel}$ error")
    plt.xlabel('Number of Layers', fontsize = 10)
    plt.ylabel('Error Metrics', fontsize = 10)
    plt.title(r'$L_2^{rel}$ and $L_{\inf}^{rel}$ Errors Against Number of Layers'+ '\n' + 'Neurons Per Layer = 20', fontsize = 13)
    plt.legend(fontsize = 10)
    plt.grid()
    plt.savefig('number of layers.jpg')
    plt.show()
    