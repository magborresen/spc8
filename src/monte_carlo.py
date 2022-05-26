"""
    Class file for class to run monte carlo simulations
"""
import logging
import os
from timeit import default_timer as timer
import numpy as np
import pandas as pd
from tqdm import tqdm
from transmitter import Transmitter
from receiver import Receiver
from radar import Radar
from particle_filter import ParticleFilter
from simulator import Simulator
from target import Target

_LOG = logging.getLogger(__name__)

class MonteCarlo:
    """
        Class to run Monte Carlo simulations through the simulator
    """

    def __init__(self, simulator, iterations=1000):
        self.simulator = simulator
        self.iterations = iterations
        self.k_obs_tot = self.simulator.k_obs_tot
        self.dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def range_theta_rmse(self):
        """
            Find the Root Mean Square Error between the observed range
            and the true range

            Args:
                None

            Returns:
                theta_rmse (np.ndarray): Trajectory error
                range_rmse (np.ndarray): Range error
        """
        # Setup lists to save data
        theta_error = []
        range_error = []
        range_rmse = []
        save_file = os.path.join(self.dirname, "sim_results/range_traj_rmse.csv")
        if os.path.exists(save_file):
            os.remove(save_file)
        cnt_k = 0
        header = True
        # Start simulating
        while cnt_k < self.k_obs_tot-1:
            _LOG.info('k=%s/%s', str(cnt_k), str(self.k_obs_tot-1))
            theta = self.simulator.target.generate_states(200, method='random')
            theta_est = []
            for k in range(theta.shape[0]):
                # Calculate distances from all antennas to target
                dists = self.simulator.radar.get_true_dist(theta[k])

                # Stop if target is out of region
                if (np.array(dists) > 2400).any() or (np.array(dists) < 1).any() or cnt_k > self.k_obs_tot-1:
                    _LOG.warning('Stopped - Target out of region, k+=%s', str(k))
                    break

                # Generate radar observation
                range_est, _ = self.simulator.radar.observation(k, theta[k], add_noise=True)

                # Range error and RMSE
                range_error.append(np.array([dists[n]-range_est[n]
                                             for n in range(self.simulator.radar.n_channels)]))
                range_rmse = np.sqrt(1/(cnt_k+1) * np.sum(np.square(range_error)))

                # Estimate target trajectory
                estimate = theta[k] + np.random.normal(0,0.1,4)
                theta_est.append(estimate)

                # Target trajectory error and RMSE
                theta_error.append(theta[k] - theta_est[k])
                theta_rmse = np.sqrt(1/(cnt_k+1) * np.sum(np.square(theta_error)))
                df = pd.DataFrame({'Range Error': range_error[k][0],
                                   'Range RMSE': range_rmse,
                                   'Trajectory RMSE': theta_rmse})

                df.to_csv(save_file,
                          mode='a',
                          index=False,
                          header=header)
                cnt_k += 1
                header = False
        theta_rmse = np.array(theta_rmse)
        range_rmse = np.array(range_rmse)

        _LOG.info("Monte Carlo done - Results saved to: %s", save_file)
        return (theta_rmse, range_rmse)

    def theta_est_theta_true(self):
        """
            Do an amount of simulations to find the monte carlo
            estimate of the error between target true position
            and estimated position
        """
        save_file = os.path.join(self.dirname, "sim_results/theta_rmse.csv")
        if os.path.exists(save_file):
            os.remove(save_file)
        header = True
        theta_err_tot = []
        for _ in tqdm(range(self.iterations)):
            exe_time = 0
            theta_err = []
            self.simulator.generate_target_states()
            self.simulator.particle_filter.init_particles_near_target(self.simulator.states[0])
            for obs in tqdm(range(self.simulator.k_obs_tot), leave=False):
                start = timer()
                target_state = self.simulator.states[obs]
                self.simulator.target_estimate(obs)
                theta_est = self.simulator.particle_filter.get_estimated_state()
                stop = timer()
                exe_time += stop - start
                theta_err.append(target_state - theta_est)
            exe_time /= self.simulator.k_obs_tot
            theta_rmse = np.sqrt(np.mean(np.square(theta_err), axis=1))
            theta_err_tot.append(theta_rmse)
            theta_rmse_tot = np.sqrt(np.mean(np.square(theta_err_tot), axis=0))
            df = pd.DataFrame({'Position RMSE x': theta_rmse[0],
                               'Position RMSE y': theta_rmse[1],
                               'Velocity RMSE x': theta_rmse[2],
                               'Velocity RMSE y': theta_rmse[3],
                               'Position total RMSE x': theta_rmse_tot[0],
                               'Position total RMSE y': theta_rmse_tot[1],
                               'Velocity total RMSE x': theta_rmse_tot[2],
                               'Velocity total RMSE y': theta_rmse_tot[3],
                               'Average Execution time': [exe_time]})
            df.to_csv(save_file, mode='a', index=False, header=header)
            header = False
        _LOG.info("Monte Carlo done - Results saved to: %s", save_file)

    def theta_est_theta_true_vector(self):
        """
            Do an amount of simulations to find the monte carlo
            estimate of the error between target true position
            and estimated position
        """
        save_file = os.path.join(self.dirname, "sim_results/theta_rmse_vectorized.csv")
        if os.path.exists(save_file):
            os.remove(save_file)
        header = True
        theta_err_tot = []
        for _ in tqdm(range(self.iterations)):
            exe_time = 0
            theta_err = []
            self.simulator.generate_target_states()
            self.simulator.particle_filter.init_particles_near_target(self.simulator.states[0])
            for obs in tqdm(range(self.simulator.k_obs_tot), leave=False):
                start = timer()
                target_state = self.simulator.states[obs]
                self.simulator.target_estimate_vectorized(obs)
                theta_est = self.simulator.particle_filter.get_estimated_state()
                stop = timer()
                exe_time += stop - start
                theta_err.append(target_state - theta_est)
            exe_time /= self.simulator.k_obs_tot
            theta_rmse = np.sqrt(np.mean(np.square(theta_err), axis=1))
            theta_err_tot.append(theta_rmse)
            theta_rmse_tot = np.sqrt(np.mean(np.square(theta_err_tot), axis=0))
            df = pd.DataFrame({'Position RMSE x': theta_rmse[0],
                               'Position RMSE y': theta_rmse[1],
                               'Velocity RMSE x': theta_rmse[2],
                               'Velocity RMSE y': theta_rmse[3],
                               'Position total RMSE x': theta_rmse_tot[0],
                               'Position total RMSE y': theta_rmse_tot[1],
                               'Velocity total RMSE x': theta_rmse_tot[2],
                               'Velocity total RMSE y': theta_rmse_tot[3],
                               'Average Execution time': [exe_time]})
            df.to_csv(save_file, mode='a', index=False, header=header)
            header = False
        _LOG.info("Monte Carlo done - Results saved to: %s", save_file)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s.%(msecs)03d] %(levelname)-8s - %(message)s",
                        datefmt="%H:%M:%S")
    _LOG.setLevel(logging.DEBUG)
    tx = Transmitter(channels=2, chirps=10, tx_power=30)
    rx = Receiver(channels=5)
    radar = Radar(tx, rx, "tdm", 2000)
    target = Target(radar.t_obs + radar.k_space, velocity=16.6)
    pf = ParticleFilter(radar.t_obs + radar.k_space, rx.channels, n_particles=1000, region=2000)
    sim = Simulator(30, radar, target, pf)
    mc = MonteCarlo(sim, iterations=200)
    #ran_err, theta_err = mc.range_theta_rmse()
    mc.theta_est_theta_true()
    #mc.theta_est_theta_true_vector()
