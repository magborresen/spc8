import numpy as np
import matplotlib.pyplot as plt
import random

class System():
    
    def __init__(self, timestep : float, region : list):
        self.T = timestep
        self.region = region
        self.F = np.array([[1, 0, self.T, 0],
                           [0, 1, 0, self.T],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.G = np.array([[((self.T**2)/2), 0],
                           [0, ((self.T**2)/2)],
                           [self.T, 0],
                           [0, self.T]])

    def random_acceleration(self, k_tot : int):
        ax = [np.cos(np.linspace(0, 2*np.pi*random.random(), k_tot))]
        ay = [np.sin(np.linspace(0, 2*np.pi*random.random(), k_tot))]
        n = np.vstack((ax,ay))
        return n

    def generate_states(self, k_tot, init_state=np.zeros((4,1))):
        n = self.random_acceleration(k_tot)
        state = []
        state.append(init_state)
        for i in range(1,k_tot):
            state.append(np.dot(self.F, state[i-1]) + np.dot(self.G, np.array([[n[0,i]],[n[1,i]]])))
        return np.array(state)
  
    def plot_trajectory(self, states):
        fig, ax = plt.subplots()
        ax.scatter(states[:,0],states[:,1])
        ax.set_aspect(1)
        plt.xlim([0, self.region[0]])
        plt.ylim([0, self.region[1]])
        plt.title('Trajectory')
        plt.ylabel('y [m]')
        plt.xlabel('x [m]')
        plt.show()
        
if __name__ == '__main__':
    model = System(1, (2000,2000))
    state = model.generate_states(75)
    model.plot_trajectory(state)
    n = model.random_acceleration(10)