import itertools
import os
from calendar import day_abbr

import click
import torch
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars

class EditOperation():
    def __init__(self, operation_string):
        self.operation_string = operation_string
        self.node = None
        self.edge = None
        self.operation_type = None
        self.operation_element = None
        parts = operation_string.split(' ')
        if len(parts) == 3:
            self.operation_element = parts[0]
            self.operation_value = parts[1]
            self.operation_type = parts[2]
            if self.operation_element == 'NODE':
                self.node = int(self.operation_value)
            elif self.operation_element == 'EDGE':
                u_v = self.operation_value.split('--')
                if len(u_v) == 2:
                    self.edge = (int(u_v[0]), int(u_v[1]))

def LoadPathResultsFromTxt(file_path):
    # output should be a list of dicts with keys 'source_id', 'step_id', 'target_id', 'operation', 'target_value'
    data = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        data = []
        for line in lines[1:]:
            parts = line.strip().split('\t')
            source_id = int(parts[0])
            step_id = parts[1]
            if step_id != 'S':
                step_id = int(step_id)
            else:
                step_id = -1
            target_id = int(parts[2])
            output_value = parts[3].split(' ')
            # convert output_value to torch tensor
            output_value = torch.tensor([float(v) for v in output_value])
            operation_string = parts[4]
            operation = None
            if operation_string != 'NONE':
                operation = EditOperation(operation_string)
            entry = {'source_id': source_id, 'step_id': step_id, 'target_id': target_id, 'operation': operation, 'output_value': output_value}
            data.append(entry)
    return data

def LoadPathResultsFromPt(file_path):
    # output should be a list of dicts with keys 'source_id', 'step_id', 'target_id', 'operation', 'target_value'
    data = torch.load(file_path, weights_only=False)
    return data

def LoadResultsFromPt(file_path):
    data = torch.load(file_path, weights_only=False)
    return data

def LoadResultFromTxt(file_path):
    # output should be a dict where the keys are the graph ids and the values a dict with keys 'target_value' and 'output_value' and their corresponding values
    data = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        data = []
        for line in lines[1:]:
            parts = line.strip().split('\t')
            # parts are graph_id, target_value and rest are output values
            graph_id = int(parts[0])
            target_value = float(parts[1])
            output_value = parts[2].split(' ')
            # convert output_value to torch tensor
            output_value = torch.tensor([float(v) for v in output_value])
            max_index = torch.argmax(output_value).item()
            entry = {'graph_id': graph_id, 'target_label': target_value, 'output_value': output_value, 'predicted_label': max_index, 'is_correct': (max_index == int(target_value))}
            data.append(entry)
    return data

class TrainingResults():
    def __init__(self, file_path):
        pass

class MainEvaluation():
    # initialize
    def __init__(self, path, strategy, dataset_name):

        # check if path/main_evaluation_dataset_name_strategy exists
        self.path = path
        self.strategy = strategy
        self.dataset_name = dataset_name
        self.data = None

        # get all files in the path that match the strategy and dataset_name
        files = os.listdir(path)
        strategy_files = [f for f in files if strategy in f and dataset_name + '_' in f]
        dataset_files = [f for f in files if dataset_name + '_' in f or dataset_name + '.' in f]
        # separate files into path_results_files, training_results_files, validation_results_files
        self.path_results_files = [f for f in strategy_files if 'path_results' in f and f.endswith('.pt')]
        self.training_results_files = [f for f in dataset_files if 'train_results' in f and f.endswith('.pt')]
        self.validation_results_files = [f for f in dataset_files if 'validation_results' in f and f.endswith('.pt')]

        # check if all three lists contain 10 files each ow throw error
        if len(self.path_results_files) != 10 or len(self.training_results_files) != 10 or len(self.validation_results_files) != 10:
            raise ValueError(f"Expected 10 files each for path results, training results and validation results but got {len(self.path_results_files)}, {len(self.training_results_files)} and {len(self.validation_results_files)} respectively.")


        self.training_results = {}
        self.validation_results = {}
        self.path_results = {}
        self.path_results_pt = {}

        for file in self.path_results_files:
            parts = file.split('_')
            config_part = [p for p in parts if p.startswith('config')]
            val_part = [p for p in parts if p.startswith('val')]
            if config_part and val_part:
                config_id = int(config_part[0].replace('config', ''))
                val_id = int(val_part[0].replace('val', ''))
                file_path = os.path.join(path, file)
                self.path_results[(config_id, val_id)] = LoadPathResultsFromPt(file_path)

        # Load the training results in a dict with key as (config_id, val_id)
        for file in self.training_results_files:
            parts = file.split('_')
            config_part = [p for p in parts if p.startswith('config')]
            val_part = [p for p in parts if p.startswith('val')]
            if config_part and val_part:
                config_id = int(config_part[0].replace('config', ''))
                val_id = int(val_part[0].replace('val', ''))
                file_path = os.path.join(path, file)
                self.training_results[(config_id, val_id)] = LoadResultsFromPt(file_path)
        # Load the validation results in a dict with key as (config_id, val_id)
        for file in self.validation_results_files:
            parts = file.split('_')
            config_part = [p for p in parts if p.startswith('config')]
            val_part = [p for p in parts if p.startswith('val') and 'validation' not in p]
            if config_part and val_part:
                config_id = int(config_part[0].replace('config', ''))
                val_id = int(val_part[0].replace('val', ''))
                file_path = os.path.join(path, file)
                self.validation_results[(config_id, val_id)] = LoadResultsFromPt(file_path)


    def merge_results(self, config_id, val_id, recalculate=True):
        # analyze the results for the given config_id and val_id
        training_results = self.training_results.get((config_id, val_id), [])
        validation_results = self.validation_results.get((config_id, val_id), [])
        path_results = self.path_results.get((config_id, val_id), [])


        # path results to polars db using the column names: source_id, step_id, target_id, operation, output_value
        import polars
        ds_paths = polars.DataFrame(path_results)
        # delete rows containing Nan values
        ds_paths = ds_paths.drop_nans()
        # delete row when column_0 is negative
        ds_paths = ds_paths.filter(polars.col('column_0') >= 0)
        ds_paths = ds_paths.filter(polars.col('column_2') >= 0)
        # delete broken rows where column_0 is not an integer, i.e., has non-zero decimal part
        ds_paths = ds_paths.filter(polars.col('column_0').cast(polars.Int64) == polars.col('column_0'))
        ds_training = polars.DataFrame(training_results)
        ds_validation = polars.DataFrame(validation_results)
        # add the column names
        ds_paths = ds_paths.rename({'column_0': 'source_id', 'column_1': 'step_id', 'column_2': 'target_id', 'column_3': 'operation', 'column_4': 'class_0', 'column_5': 'class_1'})
        ds_training = ds_training.rename({'column_0': 'graph_id', 'column_1': 'target_label', 'column_2': 'class_0', 'column_3': 'class_1'})
        ds_validation = ds_validation.rename({'column_0': 'graph_id', 'column_1': 'target_label', 'column_2': 'class_0', 'column_3': 'class_1'})

        # (polars boolean cast to integer gives 1 for True, so test class_1 > class_0)
        ds_paths = ds_paths.with_columns((polars.col('class_1') > polars.col('class_0')).cast(polars.Int64).alias('predicted_label'))

        # add a new column is_train if the source_id is in training results (only if step_id == -1 or the max step_id for source_id, target_id pair)
        # ensure we keep the original row indices by adding a row index column
        # using with_row_index (replacement for deprecated with_row_count)
        ds_paths = ds_paths.with_row_index('row_idx')

        # handle empty training/validation dataframes gracefully
        train_graph_ids = set(ds_training['graph_id'].to_list()) if 'graph_id' in ds_training.columns else set()
        validation_graph_ids = set(ds_validation['graph_id'].to_list()) if 'graph_id' in ds_validation.columns else set()

        # get list of original row indices of rows where step_id == -1
        source_graph_indices = ds_paths.filter(polars.col('step_id') == -1).select(polars.col('row_idx')).to_series().to_list()
        # subtract 1 from each index to get the previous row index
        target_graph_indices = [idx - 1 for idx in source_graph_indices if idx > 0]
        # add the last row index of the dataframe
        if target_graph_indices:
            target_graph_indices.append(ds_paths.height - 1)

        # mark the source set by introducing a new column is_source
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('step_id') == -1)
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_source')
        )
        # mark the target set by introducing a new column is_target
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('row_idx').is_in(target_graph_indices))
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_target')
        )

        # mark graphs in training set, i.e. rows where the index is in source_graph_indices and polars.col('source_id') is in train_graph_ids or target_graph_indices and polars.col('target_id') is in train_graph_ids
        ds_paths = ds_paths.with_columns(
            polars.when(
                ((polars.col('row_idx').is_in(source_graph_indices)) & (polars.col('source_id').is_in(train_graph_ids))) |
                ((polars.col('row_idx').is_in(target_graph_indices)) & (polars.col('target_id').is_in(train_graph_ids)))
            )
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_train')
        )
        # mark the same for validation set
        ds_paths = ds_paths.with_columns(
            polars.when(
                ((polars.col('row_idx').is_in(source_graph_indices)) & (polars.col('source_id').is_in(validation_graph_ids))) |
                ((polars.col('row_idx').is_in(target_graph_indices)) & (polars.col('target_id').is_in(validation_graph_ids)))
            )
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_validation')
        )

        # add a column is path for all rows that have 0 for is_source and is_target
        ds_paths = ds_paths.with_columns(
            polars.when((polars.col('is_source') == 0) & (polars.col('is_target') == 0))
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_path')
        )

        # add a new column after predicted_label called true_label
        ds_paths = ds_paths.with_columns(
            polars.lit(-1).alias('true_label')
        )
        # put the true_label column after predicted_label
        cols = ds_paths.columns
        predicted_label_index = cols.index('predicted_label')
        cols.insert(predicted_label_index + 1, cols.pop(cols.index('true_label')))
        ds_paths = ds_paths.select(cols)

        # true label train
        # fill the true_label column for rows where is_source == 1 and is_train == 1 and get the target_label from the training results
        # get all rows where is source == 1 and is_train == 1 and get the predicted_label from the training results
        train_source_rows = ds_paths.filter((polars.col('is_source') == 1) & (polars.col('is_train') == 1))
        train_target_rows = ds_paths.filter((polars.col('is_target') == 1) & (polars.col('is_train') == 1))
        validation_source_rows = ds_paths.filter((polars.col('is_source') == 1) & (polars.col('is_validation') == 1))
        validation_target_rows = ds_paths.filter((polars.col('is_target') == 1) & (polars.col('is_validation') == 1))
        # join with training results on source_id == graph_id
        train_source_rows = train_source_rows.join(ds_training, left_on='source_id', right_on='graph_id', how='left', suffix='_train')
        train_target_rows = train_target_rows.join(ds_training, left_on='target_id', right_on='graph_id', how='left', suffix='_train')
        validation_source_rows = validation_source_rows.join(ds_validation, left_on='source_id', right_on='graph_id', how='left', suffix='_val')
        validation_target_rows = validation_target_rows.join(ds_validation, left_on='target_id', right_on='graph_id', how='left', suffix='_val')
        # update true_label column with target_label from training results
        train_source_rows = train_source_rows.with_columns(
            polars.col('target_label').alias('true_label'))
        train_target_rows = train_target_rows.with_columns(
            polars.col('target_label').alias('true_label'))
        validation_source_rows = validation_source_rows.with_columns(
            polars.col('target_label').alias('true_label'))
        validation_target_rows = validation_target_rows.with_columns(
            polars.col('target_label').alias('true_label'))

        # concatenate the two dataframes
        train_rows = polars.concat([train_source_rows, train_target_rows])
        # update the true_label column in the main dataframe
        ds_paths = ds_paths.join(train_rows.select(['row_idx', 'true_label']), on='row_idx', how='left', suffix='_train')
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('true_label_train').is_not_null())
            .then(polars.col('true_label_train'))
            .otherwise(polars.col('true_label'))
            .alias('true_label')
        ).drop('true_label_train')
        validation_rows = polars.concat([validation_source_rows, validation_target_rows])
        ds_paths = ds_paths.join(validation_rows.select(['row_idx', 'true_label']), on='row_idx', how='left', suffix='_val')
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('true_label_val').is_not_null())
            .then(polars.col('true_label_val'))
            .otherwise(polars.col('true_label'))
            .alias('true_label')
        ).drop('true_label_val')

        # insert column right after column true_label called is_correct if predicted_label == true_label
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('predicted_label') == polars.col('true_label'))
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_correct')
        )
        # set is correct to -1 if true_label is -1 (unknown)
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('true_label') == -1)
            .then(polars.lit(-1))
            .otherwise(polars.col('is_correct'))
            .alias('is_correct')
        )

        true_label_index = ds_paths.columns.index('true_label')
        cols = ds_paths.columns
        is_correct_index = cols.index('is_correct')
        cols.insert(true_label_index + 1, cols.pop(is_correct_index))
        ds_paths = ds_paths.select(cols)

        # get the rows where the prediction is flipping by subtracting the predicted_label of consecutive rows
        flipping_rows = ds_paths.with_columns(
            (polars.col('predicted_label') - polars.col('predicted_label').shift(1)).abs().alias('predicted_label_diff')
        ).filter(polars.col('predicted_label_diff') > 0)
        # remove all rows where step_id == -1 or step_id is null
        flipping_rows = flipping_rows.filter((polars.col('step_id') != -1) & (polars.col('step_id').is_not_null()))
        # add a new column flipping to the main dataframe
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('row_idx').is_in(flipping_rows['row_idx'].to_list()))
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_flipping')
        )
        # move it after true_label column
        cols = ds_paths.columns
        is_flipping_index = cols.index('is_flipping')
        cols.insert(true_label_index + 1, cols.pop(is_flipping_index))
        ds_paths = ds_paths.select(cols)

        # add a new column path idx that gives the path index for each source_id and target_id pair
        # add a persistent path_idx column to ds_paths (source_id_target_id) and compute unique list
        # first convert source and target ids to ints and then to strings with 20 characters padded with leading zeros
        ds_paths = ds_paths.with_columns(
            polars.concat_str([polars.col('source_id').cast(polars.Int64).cast(polars.Utf8).str.zfill(20),
                               polars.lit('_'),
                               polars.col('target_id').cast(polars.Int64).cast(polars.Utf8).str.zfill(20)]).alias('path_idx')
        )
        path_idx_str = ds_paths['path_idx'].unique().to_list()
        # sort
        path_idx_str.sort()
        # build a small mapping DataFrame and join to add path_idx_int (compatible across polars versions)
        mapping_df = polars.DataFrame({'path_idx': path_idx_str, 'path_idx_int': list(range(len(path_idx_str)))})
        ds_paths = ds_paths.join(mapping_df, on='path_idx', how='left')
        # move path_idx_int to the front after row_idx
        cols = ds_paths.columns
        if 'path_idx_int' in cols:
            path_idx_int_index = cols.index('path_idx_int')
            cols.insert(1, cols.pop(path_idx_int_index))
            ds_paths = ds_paths.select(cols)
        # drop the path_idx string column
        ds_paths = ds_paths.drop('path_idx')

        # set colum correct_paths where predicted_label == true_label for the source and target graphs
        ds_paths = ds_paths.with_columns(
            polars.when(
                ((polars.col('is_source') == 1) | (polars.col('is_target') == 1)) &
                (polars.col('predicted_label') == polars.col('true_label'))
            )
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_correct_path_endpoint')
        )
        # set is_correct_path_endpoint to -1 if true_label is -1 (unknown)
        ds_paths = ds_paths.with_columns(
            polars.when(
                polars.col('true_label') == -1)
            .then(polars.lit(-1))
            .otherwise(polars.col('is_correct_path_endpoint'))
            .alias('is_correct_path_endpoint')
        )


        # get all the path endpoint rows
        endpoint_rows = ds_paths.filter((polars.col('is_source') == 1) | (polars.col('is_target') == 1))
        # group by path_idx_int and check if both source and target are correct
        correct_path_indices = endpoint_rows.group_by('path_idx_int').agg(
            polars.sum('is_correct_path_endpoint').alias('correct_count'),
            polars.count('is_correct_path_endpoint').alias('total_count')
        ).filter((polars.col('correct_count') == 2) & (polars.col('total_count') == 2))['path_idx_int'].to_list()
        # sort the list
        correct_path_indices.sort()
        # fill all rows with path_idx_int in correct_path_indices with 1 and others with 0 in a new column is_correct_path
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('path_idx_int').is_in(correct_path_indices))
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_correct_path')
        )

        # add two columns denoting if the path has same_endpoint_labels_predicted or same_endpoint_labels_true using the endpoint_rows (no correctness needed)
        same_endpoint_predicted_indices = endpoint_rows.group_by('path_idx_int').agg(
            polars.min('predicted_label').alias('min_predicted_label'),
            polars.max('predicted_label').alias('max_predicted_label')
        ).filter(polars.col('min_predicted_label') == polars.col('max_predicted_label'))['path_idx_int'].to_list()
        same_endpoint_true_indices = endpoint_rows.group_by('path_idx_int').agg(
            polars.min('true_label').alias('min_true_label'),
            polars.max('true_label').alias('max_true_label')
        ).filter(polars.col('min_true_label') == polars.col('max_true_label'))['path_idx_int'].to_list()
        # sort the lists
        same_endpoint_predicted_indices.sort()
        same_endpoint_true_indices.sort()
        # fill all rows with path_idx_int in same_endpoint_predicted_indices with 1 and others with 0 in a new column same_endpoint_labels_predicted
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('path_idx_int').is_in(same_endpoint_predicted_indices))
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('same_endpoint_labels_predicted')
        )
        # fill all rows with path_idx_int in same_endpoint_true_indices with 1 and others with 0 in a new column same_endpoint_labels_true
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('path_idx_int').is_in(same_endpoint_true_indices))
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('same_endpoint_labels_true')
        )


        endpoint_train_rows = ds_paths.filter(((polars.col('is_source') == 1) | (polars.col('is_target') == 1)) & (polars.col('is_train') == 1))
        endpoint_validation_rows = ds_paths.filter(((polars.col('is_source') == 1) | (polars.col('is_target') == 1)) & (polars.col('is_validation') == 1))
        # group by path_idx_int and check if both source and target are endpoints in training set resp. validation set no correctness needed
        train_path_indices = endpoint_train_rows.group_by('path_idx_int').agg(
            polars.count('is_correct_path_endpoint').alias('total_count')
        ).filter(polars.col('total_count') == 2)['path_idx_int'].to_list()
        validation_path_indices = endpoint_validation_rows.group_by('path_idx_int').agg(
            polars.count('is_correct_path_endpoint').alias('total_count')
        ).filter(polars.col('total_count') == 2)['path_idx_int'].to_list()
        # sort the lists
        train_path_indices.sort()
        validation_path_indices.sort()
        # fill all rows with path_idx_int in train_path_indices with 1 and others with 0 in a new column is_train_path
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('path_idx_int').is_in(train_path_indices))
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_train_path')
        )
        # fill all rows with path_idx_int in validation_path_indices with 1 and others with 0 in a new column is_validation_path
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('path_idx_int').is_in(validation_path_indices))
            .then(polars.lit(1))
            .otherwise(polars.lit(0))
            .alias('is_validation_path')
        )


        # convert operation (int) to string representation
        # Vectorized mapping: faster and avoids Python-level apply
        ds_paths = ds_paths.with_columns(
            polars.when(polars.col('operation') == -1).then(polars.lit('NONE'))
            .when(polars.col('operation') == 0).then(polars.lit('NODE INSERT'))
            .when(polars.col('operation') == 1).then(polars.lit('NODE DELETE'))
            .when(polars.col('operation') == 2).then(polars.lit('NODE RELABEL'))
            .when(polars.col('operation') == 3).then(polars.lit('EDGE INSERT'))
            .when(polars.col('operation') == 4).then(polars.lit('EDGE DELETE'))
            .when(polars.col('operation') == 5).then(polars.lit('EDGE RELABEL'))
            .otherwise(polars.lit('UNKNOWN')).alias('operation_str')
        )
        # move operation_str right after operation
        cols = ds_paths.columns
        operation_index = cols.index('operation')
        operation_str_index = cols.index('operation_str')
        cols.insert(operation_index + 1, cols.pop(operation_str_index))
        ds_paths = ds_paths.select(cols)





        # save the dataframe
        output_file = os.path.join(self.path, f'merged_results_config{config_id}_val{val_id}_{self.dataset_name}_{self.strategy}.csv')
        ds_paths.write_csv(output_file)
        print(f"Analyzed results saved to {output_file}")
        if self.data is None:
            # set self.data to ds_paths adding val_id and config_id columns
            ds_paths = ds_paths.with_columns(
                polars.lit(val_id).alias('val_id'),
                polars.lit(config_id).alias('config_id')
            )
            self.data = ds_paths
        else:
            # append to self.data
            ds_paths = ds_paths.with_columns(
                polars.lit(val_id).alias('val_id'),
                polars.lit(config_id).alias('config_id')
            )
            self.data = polars.concat([self.data, ds_paths])
        pass




    def create_statistics(self, config_id):
        # replace Evaluation by Statistics in the path
        self.statistics_path = self.path.replace('Evaluation', 'Statistics')
        if not os.path.exists(self.statistics_path):
            os.makedirs(self.statistics_path)

        # get all the merged files for the given config_id
        input_files = [f for f in os.listdir(self.path) if f.startswith(f'merged_results_config{config_id}_val') and f.endswith(f'_{self.dataset_name}_{self.strategy}.csv')]
        ds_all = []
        for input_file in input_files:
            val_id = int(input_file.split('_')[3].replace('val', ''))
            ds = polars.read_csv(os.path.join(self.path, input_file))
            # add a column val_id
            ds = ds.with_columns(
                polars.lit(val_id).alias('val_id')
            )
            ds_all.append(ds)
        # concatenate all dataframes
        if not ds_all:
            print(f"No merged results files found for config_id {config_id}, cannot create statistics.")
            return
        ds_paths = polars.concat(ds_all)

        # filter only correct paths if specified
        correct_paths = ds_paths.filter(polars.col('is_correct_path') == 1)
        for path_type in ['train', 'validation', 'all']:
            if path_type == 'train':
                ds_type = correct_paths.filter(polars.col('is_train_path') == 1)
            elif path_type == 'validation':
                ds_type = correct_paths.filter(polars.col('is_validation_path') == 1)
            else:
                ds_type = correct_paths
            print(f"Creating flipping statistics for {path_type} paths...")
            self.flipping_statistics(ds_type, val_ids=None, appendix=path_type)

        pass




    def flipping_statistics(self, ds_paths, val_ids=None, appendix='all'):
        same_endpoint_paths = ds_paths.filter(polars.col('same_endpoint_labels_true') ==1)
        different_endpoint_paths = ds_paths.filter(polars.col('same_endpoint_labels_true') ==0)
        # get flip statistics (get number of flips vs. path counts) per val_id
        same_aggr_path = same_endpoint_paths.group_by(['val_id', 'path_idx_int']).agg(
            polars.sum('is_flipping').alias('num_flips')
        )
        different_aggr_path = different_endpoint_paths.group_by(['val_id', 'path_idx_int']).agg(
            polars.sum('is_flipping').alias('num_flips')
        )
        # now aggregate by num_flips and count the number of paths per val_id
        same_endpoint_flips = same_aggr_path.group_by(['val_id', 'num_flips']).agg(
            polars.count('path_idx_int').alias('path_count')
        ).sort(['val_id', 'num_flips'])
        different_endpoint_flips = different_aggr_path.group_by(['val_id', 'num_flips']).agg(
            polars.count('path_idx_int').alias('path_count')
        ).sort(['val_id', 'num_flips'])
        # get mean and std across val_ids
        same_stats = same_endpoint_flips.group_by('num_flips').agg(
            polars.mean('path_count').alias('mean_path_count'),
            polars.std('path_count').alias('std_path_count')
        ).sort('num_flips')
        different_stats = different_endpoint_flips.group_by('num_flips').agg(
            polars.mean('path_count').alias('mean_path_count'),
            polars.std('path_count').alias('std_path_count')
        ).sort('num_flips')

        # create relative versions of same_endpoint_flips and different_endpoint_flips where the y-axis is the fraction of paths with that num_flips over total paths for that val_id
        # first get total paths per val_id
        total_paths_per_val = ds_paths.group_by('val_id').agg(
            polars.n_unique('path_idx_int').alias('total_paths')
        )
        # join to same_endpoint_flips and different_endpoint_flips
        same_endpoint_flips = same_endpoint_flips.join(
            total_paths_per_val,
            on='val_id',
            how='left'
        ).with_columns(
            (polars.col('path_count') / polars.col('total_paths')).alias('relative_path_count')
        ).select(['val_id', 'num_flips', 'path_count', 'relative_path_count'])
        different_endpoint_flips = different_endpoint_flips.join(
            total_paths_per_val,
            on='val_id',
            how='left'
        ).with_columns(
            (polars.col('path_count') / polars.col('total_paths')).alias('relative_path_count')
        ).select(['val_id', 'num_flips', 'path_count', 'relative_path_count'])
        # get mean and std for relative versions
        same_stats_relative = same_endpoint_flips.group_by('num_flips').agg(
            polars.mean('relative_path_count').alias('mean_relative_path_count'),
            polars.std('relative_path_count').alias('std_relative_path_count')
        ).sort('num_flips')
        different_stats_relative = different_endpoint_flips.group_by('num_flips').agg(
            polars.mean('relative_path_count').alias('mean_relative_path_count'),
            polars.std('relative_path_count').alias('std_relative_path_count')
        ).sort('num_flips')


        # evaluating which operations cause the flips
        all_operations = ds_paths.group_by(['val_id', 'operation_str']).agg(
            polars.count('operation_str').alias('operation_count')
        ).sort(['val_id', 'operation_str'])
        # get the operation that caused the flips
        # for each path, get the operations where is_flipping == 1
        flipping_operations = ds_paths.filter(polars.col('is_flipping') == 1)
        # group by val_id and sum over the different operations
        flipping_operations_aggr = flipping_operations.group_by(['val_id', 'operation_str']).agg(
            polars.count('operation_str').alias('operation_count')
        )
        # divide by the total operations per val_id to get relative frequency
        flipping_operations_aggr_relative = flipping_operations_aggr.join(
            all_operations,
            on=['val_id', 'operation_str'],
            how='left',
            suffix='_total'
        ).with_columns(
            (polars.col('operation_count') / polars.col('operation_count_total')).alias('relative_frequency')
        ).select(['val_id', 'operation_str', 'relative_frequency'])

        # take the mean and std across val_ids
        flipping_operations_stats = flipping_operations_aggr.group_by('operation_str').agg(
            polars.mean('operation_count').alias('mean_operation_count'),
            polars.std('operation_count').alias('std_operation_count')
        ).sort('operation_str')

        flipping_operations_relative_stats = flipping_operations_aggr_relative.group_by('operation_str').agg(
            polars.mean('relative_frequency').alias('mean_relative_frequency'),
            polars.std('relative_frequency').alias('std_relative_frequency')
        ).sort('operation_str')

        #evaluation which operations change the outcome how much
        # add a column change_class_x for all columns giving the absolute difference between the prediction before
        # get all rows where source is 0
        row_ids = ds_paths.filter(polars.col('is_source') == 0).select('row_idx').to_series().to_list()
        # shift by one to get the previous row
        previous_row_ids = [rid - 1 for rid in row_ids if rid > 0]
        # create a dataframe with current and previous rows
        #add change columns for class_x over all classes
        classes = [col for col in ds_paths.columns if col.startswith('class_')]
        for class_col in classes:
            ds_paths = ds_paths.with_columns(
                (polars.col(class_col) - polars.col(class_col).shift(1)).abs().alias(f'change_{class_col}')
            )
        # now aggregate the change columns by operation_str
        change_columns = [f'change_{class_col}' for class_col in classes]
        operation_change_stats = ds_paths.filter(polars.col('row_idx').is_in(row_ids)).group_by('operation_str').agg(
            *[polars.mean(col).alias(f'mean_{col}') for col in change_columns],
            *[polars.std(col).alias(f'std_{col}') for col in change_columns]
        ).sort('operation_str')





        # plot the results from above (path num vs num flips) with boxplots (one boxplot per num_flips)
        # plot the same_endpoint_flips and different_endpoint_flips in the same figure
        # use the datasets same_endpoint_flips and different_endpoint_flips

        import matplotlib.pyplot as plt
        import numpy as np
        # plot for same endpoint paths
        # first plot the boxes for the same endpoint paths (one box per num_flips value, (only even values occur))
        plt.figure()
        same_numbers = list(set(same_endpoint_flips['num_flips']))
        same_numbers.sort()
        num_validation_folds = len(set(same_endpoint_flips['val_id']))
        different_numbers = list(set(different_endpoint_flips['num_flips']))
        different_numbers.sort()
        occuring_numbers = sorted(set(same_numbers).union(set(different_numbers)))

        # boxplots for same endpoint paths
        boxplot_data_same = []
        for num in occuring_numbers:
            counts = same_endpoint_flips.filter(polars.col('num_flips') == num)['path_count'].to_list()
            # if no counts, add zeros for each validation fold
            if not counts:
                counts = [0] * num_validation_folds
            boxplot_data_same.append(counts)
        plt.boxplot(boxplot_data_same, positions=occuring_numbers, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'),
                    flierprops=dict(color='blue', markeredgecolor='blue'))
        # boxplots for different endpoint paths
        boxplot_data_different = []
        for num in occuring_numbers:
            counts = different_endpoint_flips.filter(polars.col('num_flips') == num)['path_count'].to_list()
            # if no counts, add zeros for each validation fold
            if not counts:
                counts = [0] * num_validation_folds
            boxplot_data_different.append(counts)
        plt.boxplot(boxplot_data_different, positions=occuring_numbers, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='green'),
                    medianprops=dict(color='darkgreen'),
                    whiskerprops=dict(color='green'),
                    capprops=dict(color='green'),
                    flierprops=dict(color='green', markeredgecolor='green'))
        plt.legend(['Same Endpoint Labels', 'Different Endpoint Labels'])
        plt.xlabel('Number of Flips in Path')
        plt.ylabel('Number of Paths')
        plt.title(f'Flipping Statistics for {appendix.capitalize()} Paths\nDataset: {self.dataset_name}, Strategy: {self.strategy}')
        plt.xticks(occuring_numbers)
        plt.grid(axis='y')
        # save the figure
        output_file = os.path.join(self.statistics_path, f'flipping_statistics_config{config_id}_{self.dataset_name}_{self.strategy}_{appendix}_flips.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Flipping statistics plot saved to {output_file}")


        # plot also the relative versions
        plt.figure()
        # boxplots for same endpoint paths (relative)
        boxplot_data_same_relative = []
        for num in occuring_numbers:
            counts = same_endpoint_flips.filter(polars.col('num_flips') == num)['relative_path_count'].to_list()
            # if no counts, add zeros for each validation fold
            if not counts:
                counts = [0] * num_validation_folds
            boxplot_data_same_relative.append(counts)
        plt.boxplot(boxplot_data_same_relative, positions=occuring_numbers, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'),
                    whiskerprops=dict(color='blue'),
                    capprops=dict(color='blue'),
                    flierprops=dict(color='blue', markeredgecolor='blue'))
        # boxplots for different endpoint paths (relative)
        boxplot_data_different_relative = []
        for num in occuring_numbers:
            counts = different_endpoint_flips.filter(polars.col('num_flips') == num)['relative_path_count'].to_list()
            # if no counts, add zeros for each validation fold
            if not counts:
                counts = [0] * num_validation_folds
            boxplot_data_different_relative.append(counts)
        plt.boxplot(boxplot_data_different_relative, positions=occuring_numbers, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightgreen', color='green'),
                    medianprops=dict(color='darkgreen'),
                    whiskerprops=dict(color='green'),
                    capprops=dict(color='green'),
                    flierprops=dict(color='green',
                                    markeredgecolor='green'))
        plt.legend(['Same Endpoint Labels', 'Different Endpoint Labels'])
        plt.xlabel('Number of Flips in Path')
        plt.ylabel('Fraction of Paths')
        plt.title(f'Relative Flipping Statistics for {appendix.capitalize()} Paths\nDataset: {self.dataset_name}, Strategy: {self.strategy}')
        plt.xticks(occuring_numbers)
        plt.grid(axis='y')
        # save the figure
        output_file = os.path.join(self.statistics_path, f'flipping_statistics_config{config_id}_{self.dataset_name}_{self.strategy}_{appendix}_flips_relative.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Relative flipping statistics plot saved to {output_file}")


        # plot the flipping operations statistics using also boxplots
        plt.figure()
        operation_names = flipping_operations_stats['operation_str'].to_list()
        # boxplot data for flipping operations
        boxplot_data_operations = []
        for operation in operation_names:
            counts = flipping_operations_aggr.filter(polars.col('operation_str') == operation)['operation_count'].to_list()
            # if no counts, add zeros for each validation fold
            if not counts:
                counts = [0] * num_validation_folds
            boxplot_data_operations.append(counts)
        plt.boxplot(boxplot_data_operations, positions=np.arange(len(operation_names)), widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral', color='darkred'),
                    medianprops=dict(color='yellow'),
                    whiskerprops=dict(color='darkred'),
                    capprops=dict(color='darkred'),
                    flierprops=dict(color='darkred',
                                    markeredgecolor='darkred'))
        plt.xticks(np.arange(len(operation_names)), operation_names, rotation=15, ha='right')
        plt.xlabel('Graph Edit Operation')
        plt.ylabel('Number of Flips Caused by Operation')
        plt.title(f'Flipping Operations Statistics\nDataset: {self.dataset_name}, Strategy: {self.strategy}')
        plt.grid(axis='y')
        # save the figure
        output_file = os.path.join(self.statistics_path, f'flipping_statistics_config{config_id}_{self.dataset_name}_{self.strategy}_{appendix}_operations.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Flipping operations statistics plot saved to {output_file}")

        # plot the relative flipping operations statistics using also boxplots
        plt.figure()
        # boxplot data for flipping operations relative
        boxplot_data_operations_relative = []
        for operation in operation_names:
            counts = flipping_operations_aggr_relative.filter(polars.col('operation_str') == operation)['relative_frequency'].to_list()
            # if no counts, add zeros for each validation fold
            if not counts:
                counts = [0] * num_validation_folds
            boxplot_data_operations_relative.append(counts)
        plt.boxplot(boxplot_data_operations_relative, positions=np.arange(len(operation_names)), widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral', color='darkred'),
                    medianprops=dict(color='yellow'),
                    whiskerprops=dict(color='darkred'),
                    capprops=dict(color='darkred'),
                    flierprops=dict(color='darkred', markeredgecolor='darkred'))
        plt.xticks(np.arange(len(operation_names)), operation_names, rotation=15, ha='right')
        plt.xlabel('Graph Edit Operation')
        plt.ylabel('Relative Frequency of Flips Caused by Operation')
        plt.title(f'Relative Flipping Operations Statistics\nDataset: {self.dataset_name}, Strategy: {self.strategy}')
        plt.grid(axis='y')
        # save the figure
        output_file = os.path.join(self.statistics_path, f'flipping_statistics_config{config_id}_{self.dataset_name}_{self.strategy}_{appendix}_operations_relative.png')
        plt.savefig(output_file)
        plt.close()
        print(f"Relative flipping operations statistics plot saved to {output_file}")


        # also save the results as csv files to compare different datasets and gnn algorithms
        output_file = os.path.join(self.statistics_path, f'flipping_statistics_config{config_id}_{self.dataset_name}_{self.strategy}_{appendix}_flips.csv')
        same_stats.write_csv(output_file.replace('.csv', '_same_endpoint.csv'))
        different_stats.write_csv(output_file.replace('.csv', '_different_endpoint.csv'))
        flipping_operations_stats.write_csv(output_file.replace('.csv', '_operations.csv'))
        flipping_operations_relative_stats.write_csv(output_file.replace('.csv', '_operations_relative.csv'))
        print(f"Flipping statistics CSV files saved to {self.statistics_path}")







@click.command()
@click.option("--config-id", default=0, type=int, show_default=True, help="Configuration ID to process.")
@click.option(
    "--db",
    "dbs",
    multiple=True,
    default=("MUTAG", "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR"),
    show_default=True,
    help="Dataset to include. Repeat the option to pass multiple datasets.",
)
@click.option(
    "--path-strategy",
    "path_strategies",
    multiple=True,
    default=("d-E_d-IsoN", "i-E_d-IsoN", "Rnd"),
    show_default=True,
    help="Path strategy to include. Repeat the option to pass multiple strategies.",
)
@click.option(
    "--gnn-algorithm",
    "gnn_algorithms",
    multiple=True,
    default=("GIN", "GATv2", "GCN", "GraphSAGE"),
    show_default=True,
    help="GNN algorithm to include. Repeat the option to pass multiple algorithms.",
)
@click.option(
    "--recalculate/--no-recalculate",
    default=False,
    show_default=True,
    help="Whether to force recalculation while merging results.",
)
@click.option(
    "--results-path",
    default="results",
    show_default=True,
    help="Base directory containing evaluation results.",
)
@click.option(
    "--evaluation-folder",
    default="path_evaluation",
    show_default=True,
    help="Subfolder under each algorithm result directory with evaluation artifacts.",
)
def main(config_id, dbs, path_strategies, gnn_algorithms, recalculate, results_path, evaluation_folder):
    dbs = list(dbs)
    path_strategies = list(path_strategies)
    gnn_algorithms = list(gnn_algorithms)

    tasks = list(itertools.product(dbs, path_strategies, gnn_algorithms))
    all_data = polars.DataFrame()
    for db, strategy, gnn_algorithm in tasks:
        print(f"Processing results for Dataset: {db}, Strategy: {strategy}, GNN: {gnn_algorithm}")
        main_evaluation_path = f'{results_path}/{gnn_algorithm}/{evaluation_folder}'
        # check if the path exists
        if not os.path.exists(main_evaluation_path):
            print(f"Path {main_evaluation_path} does not exist, skipping...")
            continue
        mainEvaluation = MainEvaluation(main_evaluation_path, strategy, db)
        flipping_statistics_all_folds = dict()
        for val_id in range(0, 10):
            mainEvaluation.merge_results(config_id, val_id, recalculate=recalculate)
        # add gnn_algorithm column, db column, strategy column to mainEvaluation.data
        mainEvaluation.data = mainEvaluation.data.with_columns(
            polars.lit(gnn_algorithm).alias('gnn_algorithm'),
            polars.lit(db).alias('dataset'),
            polars.lit(strategy).alias('path_strategy')
        )
        if all_data is None:
            all_data = mainEvaluation.data
        else:

            all_data = polars.concat([all_data, mainEvaluation.data])
        #mainEvaluation.create_statistics(config_id)
    # save all_data to csv
    output_file = f'{results_path}/all_results.csv'
    all_data.write_csv(output_file)
    print(f"All results saved to {output_file}")
    pass


if __name__ == '__main__':
    main()
