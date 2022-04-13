"""
    This module will generate a trajectory for the target
"""
import numpy as np
import matplotlib.pyplot as plt

class Target():
    """
        Class to generate target states
    """
    def __init__(self, timestep : float):
        self.T = timestep
        self.F = np.block([[np.eye(2), self.T*np.eye(2)],
                           [np.zeros((2,2)), np.eye(2)]])
        self.G = np.block([[(self.T**2/2)*np.eye(2)],
                           [self.T*np.eye(2)]])

    def acceleration(self, k_tot : int, method):
        """ 
            Generate target accelerations for all observations.
        
            A method can be defined, depending on what kind of movement pattern
            is desired for the target.

            Args:
                k_tot (int): Observations
                method (str): Acceleration pattern, which can be...
                    'spiral': A spiralling motion from region center
                    'linear': A linear motion, going towards origin
                    None: A linear motion, going away from origin

            Returns:
                acc (list): Acceleration for each observation
        """

        # Spiral from region center and out:
        if method == 'spiral':
            print("doing spiral")
            self.init_state = np.array([[1000], [1000], 
                                        [0], [0]])
            k = int(k_tot/2)
            
            ax = np.block([np.cos(np.linspace(0, 2*np.pi, k)),
                           -np.cos(np.linspace(0, np.pi, k_tot - k))])
            ay = np.block([-np.sin(np.linspace(0, 2*np.pi, k)),
                           np.sin(np.linspace(0, np.pi, k_tot - k))])

        # Linear, away from origin:
        elif method == 'linear':
            print("doing linear")
            self.init_state = np.array([[0], [0], 
                                        [100], [100]])
            ax = np.zeros((1,k_tot))
            ay = np.zeros((1,k_tot))

        # Linear, towards origin:
        else:
            print("doing default")
            self.init_state = np.array([[2000], [2000], 
                                        [-100], [-100]])
            ax = np.zeros((1,k_tot))
            ay = np.zeros((1,k_tot))

        acc = np.vstack((ax,ay))
        return acc

    def generate_states(self, k_tot, method=None):
        """ Generate k states, based on acceleration.

            Args:
                k_tot (int): Observations
                method (str): Acceleration pattern

            Returns:
                state (list): List of k states
        """
        acc = self.acceleration(k_tot, method)
        state = [self.init_state] 
        for i in range(1,k_tot):
            state.append(np.dot(self.F, state[i-1]) 
                         + np.dot(self.G, np.array([[acc[0,i]],[acc[1,i]]])))
        return np.array(state) 