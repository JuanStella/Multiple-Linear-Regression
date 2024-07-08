import math, copy
import numpy as np
import matplotlib.pyplot as plt

class algorithm:

    def __init__(self, alfa, it_max):
        self.alfa = alfa
        self.it_max = it_max
        self.b = 0
        self.w = None
        self.x = None
        self.y = None
        self.data = None
        self.b_history = []  # Para almacenar los valores de b
        self.w1_history = []  # Para almacenar los valores de w
        self.w2_history = []
        self.cost_history = []  # Para almacenar los valores de J(w,b)
        self.it_history = []  # Para almacenar los valores de las iteraciones   

    def read_data(self, file):
        self.data = np.loadtxt(file, delimiter=',', skiprows=1)
        return None

    def model(self, data):
        self.x = np.zeros((2, len(data)))
        self.x[0, :] = data[:, 0]
        self.x[1, :] = data[:, 1]

        self.y = data[:, 2]
        self.y = np.array(self.y)

        self.y = self.y / np.max(self.y)

        n, m = self.x.shape  # (number of features, number of examples)
        self.w = np.zeros((n,))
        self.b = 0

        it = 0

        while it < self.it_max :
            
            for j in range(n):
                self.w[j] = self.w[j] - self.alfa * 1 / m * np.sum((np.dot(self.w, self.x) + self.b - self.y) * self.x[j, :])
            self.b = self.b - self.alfa * 1 / m * np.sum(np.dot(self.w, self.x) + self.b - self.y)
            self.cost_history.append(1 / (2 * m) * np.sum((self.y - (np.dot(self.w[0], self.x[0,:]) + self.b)) ** 2))
            self.it_history.append(it)
            dif = self.cost_history[it] - self.cost_history[it - 1]
            print(abs(dif))
            if (abs(dif) < 0.0001) and it > 0:
                break
            it += 1
            
        
        return np.dot(self.w, self.x),self.cost_history, self.it_history
    
def main():
    algo = algorithm(0.00001, 500)
    algo.read_data('D:\\ML\\MyProgress\\Multiple Linear Regression\\archive\\data.csv')
    f , cost_hist, it_hist  = algo.model(algo.data)

    
    algo.y = algo.y * np.max(algo.data[:, 2])
    f = f * np.max(algo.data[:, 2])
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(algo.x[0, :], algo.x[1,:], algo.y, color='b', label='Data Points')
    ax.plot_trisurf(algo.x[0, :], algo.x[1, :], f, color='r', alpha=0.5, label='Regression Plane')

    # Añadir una barra de color
    scatter = ax.scatter(algo.x[0, :], algo.x[1, :], algo.y, c=algo.y, cmap='viridis', label='Data Points')
    fig.colorbar(scatter, label='Income', shrink=0.5, aspect=5)
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Yrs of Exp')
    ax.set_zlabel('Income')
    ax.legend()

    plt.show()
    

    # Plotting the cost history
    plt.figure()
    plt.plot(it_hist, cost_hist, color='b', label='Cost Function')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()


if __name__ == '__main__':
    main()
