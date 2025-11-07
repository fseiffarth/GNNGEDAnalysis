## Real World Data
import sys
from pathlib import Path
import os
import click


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from GNNBench import ExperimentMain



def train_ged(num_threads=-1):
    # Load and preprocess the experiment
    experiment = ExperimentMain(Path('Examples/GED/Configs/main_config.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)

    # Run and evaluate all configurations defined in the config file
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results(evaluate_validation_only=True)

@click.command()
@click.option('--num_threads', default=1, help='Number of threads to use')
def main(num_threads):
    train_ged(num_threads)



if __name__ == '__main__':
    main()