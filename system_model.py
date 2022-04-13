"""
    This module creates a system model for the estimator to use.
"""
import numpy as np
import matplotlib.pyplot as plt

class System():
    """
        Class to represent an oberservation
    """
    def __init__(self, timestep : float):
        self.T = timestep
        self.F = np.block([[np.eye(2), self.T*np.eye(2)],
                           [np.zeros((2,2)), np.eye(2)]])
        self.G = np.block([[(self.T**2/2)*np.eye(2)],
                           [self.T*np.eye(2)]])

    def acceleration(self, k_tot : int):
        """ Generate k input vectors for the system.
            Right now this function sets all acceleration to 0!

            Args:
                k_tot (int): Observations

            Returns:
                acc (list): Acceleration for each observation
        """
        #ax = [np.cos(np.linspace(0, 2*np.pi, k_tot))]
        #ay = [np.sin(np.linspace(0, 2*np.pi, k_tot))]
        ax = np.zeros((1,k_tot))
        ay = np.zeros((1,k_tot))
        acc = np.vstack((ax,ay))        
        return acc

    def generate_states(self, k_tot, init_state=np.array([[200], [200], [14], [14]])):
        """ Generate k states, based on acceleration.

            Args:
                k_tot (int): Observations
                init_state (list): Initial position and velocities

            Returns:
                state (list): List of k states
        """
        acc = self.acceleration(k_tot)
        state = [init_state] 
        for i in range(1,k_tot):
            state.append(np.dot(self.F, state[i-1]) 
                         + np.dot(self.G, np.array([[acc[0,i]],[acc[1,i]]])))
        return np.array(state)

        
    def velocity(self, states):
        """ Calculate velocity for all states.

            Args:
                states (list): States to plot

            Returns:
                vel (list): List of velocities per observation
        """
        vel = np.sqrt(states[:,2]**2 + states[:,3]**2)
        return vel
        
    
if __name__ == '__main__':
    model = System(1, (5000,5000))
    state = model.generate_states(100, np.array([[0],[0],[10],[10]]))
    print(model.generate_states(100))
    