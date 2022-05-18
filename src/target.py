"""
    This module will generate a trajectory for the target
"""
import numpy as np
import matplotlib.pyplot as plt

class Target():
    """
        Class to generate target states
    """
    def __init__(self, timestep : float, region=2000, velocity=16):
        self.T = timestep
        self.F = np.block([[np.eye(2), self.T*np.eye(2)],
                           [np.zeros((2,2)), np.eye(2)]])
        self.G = np.block([[(self.T**2/2)*np.eye(2)],
                           [self.T*np.eye(2)]])
        self.region = region
        self.velocity = velocity
        self.init_state = None

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
        if method == 'random':
            # Random velocity
            vel_x = np.sqrt(np.random.uniform(0, self.velocity**2))
            vel_y = np.sqrt(self.velocity**2 - vel_x**2)

            direction = np.random.randint(4)
            
            if direction == 0: # From the left
                self.init_state = np.array([[0], 
                                            [np.random.uniform(0, 0.9 * self.region*vel_y/self.velocity)],
                                            [vel_x], [vel_y]])
            elif direction == 1: # From the right
                self.init_state = np.array([[self.region], 
                                            [np.random.uniform(0, 0.9 * self.region*vel_y/self.velocity)],
                                            [-vel_x], [vel_y]])
            else: # From the top
                self.init_state = np.array([[np.random.uniform(0, self.region)], 
                                            [self.region],
                                            [vel_x], [-vel_y]])
        # Linear, away from origin:
        elif method == 'linear_away':
            self.init_state = np.array([[1000], [100],
                                        [0], [self.velocity]])
        # Linear, towards origin:
        else:
            self.init_state = np.array([[1000], [1000],
                                        [0], [-self.velocity]])
        ax = np.random.normal(0, 0.1, k_tot)#np.zeros((1,k_tot))
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
    
    def plot_region(self, states, zoom=False):
        """
            Plots the observation region with antenna locations and trajectory

            Args:
                states (np.ndarray): Collection of all states
                zoom (bool): Show only trajectory region if true

            Returns:
                no value
        """
        _, ax = plt.subplots()
        ax.scatter(states[:,0], states[:,1], label="Trajectory")
        ax.set_aspect(1)
        if zoom:
            ax.set_xlim(min(states[:,0]), max(states[:,0]))
            ax.set_ylim(min(states[:,1]), max(states[:,1]))
        else:
            ax.set_xlim(0, self.region)
            ax.set_ylim(0, self.region)
        plt.title('Observation region with antennas and trajectory')
        plt.ylabel('y [m]')
        plt.xlabel('x [m]')
        plt.legend()
        plt.show()

    def get_velocities(self, target_states):
        
        vel = []
        for idx, state in enumerate(target_states):
            vel.append(np.linalg.norm((state[2], state[3])))
            
        vel = np.array(vel)
        
        print(f'Minimum: {np.min(vel)}\nMaximum: {np.max(vel)}\nMean: {np.mean(vel)}')
        
if __name__ == '__main__':
    t_obs = 1.0003
    target = Target(t_obs, 2000, velocity=60)
    states = target.generate_states(100, method='random')
    target.plot_region(states)
    
    target.get_velocities(states)