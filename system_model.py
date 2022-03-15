import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal

class System():
    
    def __init__(self, timestep : float, region : list):
        self.T = timestep
        self.region = region
        self.F = np.block([[np.eye(2), self.T*np.eye(2)],
                           [np.zeros((2,2)), np.eye(2)]])
        self.G = np.block([[(self.T**2/2)*np.eye(2)],
                           [self.T*np.eye(2)]])

    def random_acceleration(self, k_tot : int):
        ax = [np.cos(np.linspace(0, 2*np.pi, k_tot))]
        ay = [np.sin(np.linspace(0, 2*np.pi, k_tot))]
        #ax = np.zeros((1,k_tot))
        #ay = np.zeros((1,k_tot))
        #ax[0,60] = 1
        #ay[0,60] = -1
        acc = np.vstack((ax,ay))        
        return acc

    # def generate_states(self, k_tot, init_state=14*np.ones((4,1))):
    #     acc = self.random_acceleration(k_tot)
    #     state = [init_state] 
    #     states = [(state.append(np.dot(self.F, state[-1]) + np.dot(self.G, np.array([[acc[0,i]],[acc[1,i]]]))), state[-1])[1] for i in range(1,k_tot)]
    #     return np.array(states)
    def generate_states(self, k_tot, init_state=14*np.ones((4,1))):
        acc = self.random_acceleration(k_tot)
        state = [init_state] 
        for i in range(1,k_tot):
            state.append(np.dot(self.F, state[i-1]) 
                         + np.dot(self.G, np.array([[acc[0,i]],[acc[1,i]]])))
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
        
    def velocity(self, states):
        vel = np.sqrt(states[:,2]**2 + states[:,3]**2)
        return vel
        
    
if __name__ == '__main__':
    model = System(1, (5000,5000))
    state = model.generate_states(100, np.array([[0],[0],[10],[10]]))
    print(model.velocity(state))
    model.plot_trajectory(state)
    