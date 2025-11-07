## Real World Data
from pathlib import Path

import click
import numpy as np

from src.Architectures.ShareGNN.Parameters import Parameters
from src.Experiment.ExperimentMain import ExperimentMain, preprocess_graph_data
from src.Experiment.ModelConfiguration import ModelConfiguration
from src.Experiment.RunConfiguration import get_run_configs
from src.Preprocessing.GraphData.GraphData import ShareGNNDataset
from src.Preprocessing.load_preprocessed import load_preprocessed_data_and_parameters
from src.utils.load_splits import Load_Splits


def evaluate_gnn(num_threads=-1):
    # Load and preprocess the experiment
    experiment = ExperimentMain(Path('Examples/GED/Configs/main_config.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    run_id = 0

    # load the dataset and run the pretrained model evaluations

    datasets = set([config['name'] for config in experiment.main_config['datasets']])
    for db in datasets:
        for config in experiment.network_configurations[db]:
            split_data = Load_Splits(config['paths']['splits'])
            train_ids = split_data['train']
            validation_ids = split_data['validation']
            test_ids = split_data['test']
            # get all possible hyperparameter configurations from the config files
            run_configs = get_run_configs(config)
            for config_id, run_config in enumerate(run_configs):
                for val_id in range(config['validation_folds']):
                    model = experiment.load_ordinary_model(db_name=db, validation_id=val_id, best=False, run_id=run_id)

                    # create the model configuration object
                    graph_data = preprocess_graph_data(config)
                    para = Parameters()
                    load_preprocessed_data_and_parameters(config_id=config_id,
                                                          run_id=run_id,
                                                          validation_id=val_id,
                                                          validation_folds=config.get('validation_folds', 10),
                                                          graph_data=graph_data, run_config=run_config, para=para)
                    seed = 42 + val_id + para.n_val_runs * run_id
                    configuration = ModelConfiguration(run_id, val_id, graph_data, (train_ids, validation_ids, test_ids), seed, para)
                    # Initialize the graph neural network
                    configuration.initialize_model(pretrained_network=model,
                                          use_model=configuration.para.run_config.config.get('use_model', 'ShareGNN'))
                    print(f"Evaluating model for config_id {config_id}, val_id {val_id} on training set")
                    configuration.evaluate_network(graph_ids=train_ids[val_id])
                    print(f"Evaluating model for config_id {config_id}, val_id {val_id} on validation set")
                    configuration.evaluate_network(graph_ids=validation_ids[val_id])

@click.command()
@click.option('--num_threads', default=1, help='Number of threads to use')
def main(num_threads):
    evaluate_gnn(num_threads)



if __name__ == '__main__':
    main()