# This is the script we used for generating "Figure 2: Common mechanism vs. tuned DHC algorithm as
# privacy budget 𝜌 varies" in the paper.

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14

def make_plot(name):

    if name == 'T10':
        x = [ round(1/(2*s**2),3) for s in [2,4,6,8,10,12]]
        xi = list(range(len(x)))

        plt.plot(xi, [97.71, 94.59, 89.03, 87.02, 87.57, 88.81], marker = '.', label = 'DHC with noiseless threshold', linewidth=3)
        # plt.plot(xi, [97.71, 94.50, 88.91, 86.86, 87.18, 88.30], marker = '.', label = 'LR1', linewidth=2)
        # plt.plot(xi, [97.68, 94.16, 88.06, 85.60, 85.02, 85.17], marker = '.', label = 'LR2', linewidth=2)
        # plt.plot(xi, [97.63, 91.70, 83.17, 76.70, 73.59, 71.63], marker = '.', label = 'LR3', linewidth=2)
        plt.plot(xi, [99.80, 98.74, 97.32, 96.09, 95.88, 95.75], marker = '.', label = 'CM Accuracy', linewidth=3)
        plt.legend(prop={'size': 18}, loc = 3)
        plt.xticks(xi, x)
        plt.ylim(60,100)
        plt.title('(x,y) = (0.5,20) with different ' r'$\rho$',fontsize = 20)
        plt.xlabel(r'$\rho$',fontsize = 14)
        plt.ylabel('Accuracy',fontsize = 20)

        plt.savefig('Table11.pdf')
        plt.show()

    # elif name == 'T11':
    #
    #     x = [2,7,12,17,22,27]
    #     xi = list(range(len(x)))
    #
    #     plt.plot(xi, [99.17, 96.17, 92.23, 86.51, 87.23, 87.90], marker = '.', label = 'DHC with noiseless threshold', linewidth=2)
    #     # plt.plot(xi, [99.07, 96.17, 91.98, 86.36, 87.05, 87.67], marker = '.', label = 'LR1', linewidth=2)
    #     # plt.plot(xi, [99.03, 95.88, 90.64, 85.10, 85.47, 86.23], marker = '.', label = 'LR2', linewidth=2)
    #     # plt.plot(xi, [98.67, 93.32, 84.62, 78.41, 76.46, 77.54], marker = '.', label = 'LR3', linewidth=2)
    #     plt.plot(xi, [99.61, 98.80, 97.23, 96.19, 96.24, 96.76], marker = '.', label = 'CM', linewidth=2)
    #     plt.legend(prop={'size': 13}, loc = 1)
    #     plt.xticks(xi, x)
    #     plt.title('x=0.5, 'r'$\rho$''=1/128 with different y',fontsize = 15)
    #     plt.xlabel('y',fontsize = 14)
    #     plt.ylabel('Accuracy',fontsize = 16)
    #
    #     plt.savefig('Table11.pdf')
    #     plt.show()
    #
    # elif name == 'T12':
    #
    #     x = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    #     xi = list(range(len(x)))
    #
    #     plt.plot(xi, [92.83, 89.37, 88.26, 87.02, 86.10, 86.11, 73.20], marker = '.', label = 'DHC with noiseless threshold', linewidth=2)
    #     # plt.plot(xi, [92.52, 89.23, 88.09, 86.86, 85.82, 85.85, 73.19], marker = '.', label = 'LR1', linewidth=2)
    #     # plt.plot(xi, [90.69, 87.66, 86.37, 85.60, 84.41, 84.73, 72.74], marker = '.', label = 'LR2', linewidth=2)
    #     # plt.plot(xi, [80.18, 78.14, 76.43, 76.70, 75.66, 77.93, 69.83], marker = '.', label = 'LR3', linewidth=2)
    #     plt.plot(xi, [97.43, 96.79, 96.27, 96.09, 95.88, 96.25, 95.32], marker = '.', label = 'CM', linewidth=2)
    #     plt.legend(prop={'size': 13}, loc = 1)
    #     plt.xticks(xi, x)
    #     plt.title('y=20, 'r'$\rho$''=1/128 with different x',fontsize = 15)
    #     plt.xlabel('x',fontsize = 16)
    #     plt.ylabel('Accuracy',fontsize = 16)
    #
    #     plt.savefig('Table12.pdf')
    #     plt.show()

make_plot('T10')