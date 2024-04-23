import numpy as np
from lnl_computer.cosmic_integration.mcz_grid import McZGrid
from lnl_computer.observation.mock_observation import MockObservation
import multiprocessing
import h5py

class ParameterDistributions:
    """Class to handle the generation of parameter distributions for the simulation."""

    def __init__(self):
        """Initialize the parameter distributions."""
        self.params = {
            'aSF': np.linspace(0.005, 0.015, 100),
            'dSF': np.linspace(4.2, 5.2, 100),
            'sigma0': np.linspace(0.1, 0.6, 100),
            'muz': np.linspace(-0.5, -0.001, 100)
        }

    def sample(self):
        """Randomly sample a value from each parameter distribution."""
        return {key: np.random.choice(values) for key, values in self.params.items()}


class Simulation:
    """Class to process individual simulation tasks."""

    def __init__(self, compas_fname):
        """Initialize the task with the COMPAS file path."""
        self.compas_fname = compas_fname

    def run(self, sample):
        """Run the simulation task with a given sample and return the lnl and sampled MSSFR parameters"""
        mcz_grid = McZGrid.generate_n_save(compas_h5_path=self.compas_fname, sf_sample=sample, save_plots=False)
        obs = MockObservation.from_mcz_grid(mcz_grid, duration=1.0)
        lnl, unc = McZGrid.lnl(mcz_obs=obs.mcz, duration=1, compas_h5_path=self.compas_fname, sf_sample=sample, n_bootstraps=0, outdir='temp_COMPAS_data_Jeff', save_plots=False)
        return lnl, np.array(list(sample.values()))


def process_task(sample):
    simulation = Simulation('/fred/oz016/Chayan/COMPAS_populations_project/COMPAS_Output.h5') # Replace with path to Z_all file
    return simulation.run(sample)

if __name__ == '__main__':
    np.random.seed(0)
    distributions = ParameterDistributions()
    params = [distributions.sample() for _ in range(10000)]

    with multiprocessing.Pool() as pool:
        results = pool.map(process_task, params)

    lnl_values = np.array([result[0] for result in results])
    sf_samples = np.stack([result[1] for result in results])

    with h5py.File('/fred/oz016/Chayan/COMPAS_populations_project/COMPAS_lnl_data_Z_all.hdf', 'w') as f:
        f.create_dataset('lnl', data=lnl_values)
        f.create_dataset('sf_samples', data=sf_samples)

