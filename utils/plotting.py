import sys
from pathlib import Path

import click
import polars
import torch
from simplegnn.framework.core import FrameworkMain, preprocess_graph_data
from simplegnn.framework.model_configuration import ModelConfiguration
from simplegnn.framework.run_configuration import get_run_configs
from simplegnn.framework.utils.parameters import Parameters
from simplegnn.framework.utils.preprocessing import load_splits, load_preprocessed_data_and_parameters


def _import_framework_main():
    try:
        from simplegnn.framework.core import FrameworkMain
        return FrameworkMain
    except ImportError:
        repo_root = Path(__file__).resolve().parent
        simplegnn_src = repo_root.parent / "SimpleGNN" / "repo" / "src"
        sys.path.append(str(simplegnn_src))
        try:
            from simplegnn.framework.core import FrameworkMain
            return FrameworkMain
        except ImportError as exc:
            raise ImportError(
                "Could not import FrameworkMain from simplegnn. "
                f"Expected local source at '{simplegnn_src}' or an installed 'simplegnn' package."
            ) from exc



def table_accuracies(dbs, gnn_algos):
    # dict of key: (db, gnn_algorithm), value: dict of config_id -> (mean_validation_accuracy, std_validation_accuracy)
    results = {}
    run_id = 0
    for gnn_algorithm in gnn_algos:
        for db in dbs:
            # Load and preprocess the experiment
            experiment_base = FrameworkMain(Path(f'configs/{db}/main_config.yml'))
            experiment_base.preprocessing(num_threads=-1)
            # evaluate the pretrained model on the original data only for testing
            for db_id, config in enumerate(experiment_base.network_configurations[db]):
                algorithm = str(config['paths']['models']).split('_')[1]
                # also remove possible .yml at the end
                algorithm = algorithm.replace('.yml', '')
                if gnn_algorithm is not None and algorithm != gnn_algorithm:
                    continue
                split_data = load_splits(config['paths']['splits'])
                train_ids = split_data['train']
                validation_ids = split_data['validation']
                test_ids = split_data['test']

                # get the graph data
                graph_data = preprocess_graph_data(config)

                # get all possible hyperparameter configurations from the config files
                run_configs = get_run_configs(config)
                for config_id, run_config in enumerate(run_configs):
                    validation_results = {}
                    num_validation_folds = len(config['splits']['validation'])
                    for val_id in range(num_validation_folds):
                        model = experiment_base.load_ordinary_model(db_name=db, validation_id=val_id, best=False, run_id=run_id, experiment_db_id=db_id)
                        model.eval()
                        para = Parameters()
                        load_preprocessed_data_and_parameters(config_id=config_id,
                                                              run_id=run_id,
                                                              validation_id=val_id,
                                                              validation_folds=num_validation_folds,
                                                              graph_data=graph_data, run_config=run_config, para=para)
                        seed = 42 + val_id + para.n_val_runs * run_id
                        configuration = ModelConfiguration(run_id, val_id, graph_data, (train_ids, validation_ids, test_ids),
                                                           seed, para)
                        # Initialize the graph neural network
                        configuration.initialize_model(pretrained_network=model)
                        print(f"Evaluating model for config_id {config_id}, val_id {val_id} on validation set")
                        validation_values, validation_outputs = configuration.evaluate_network(graph_ids=validation_ids[val_id], do_print=True, with_loss=True)
                        predictions = torch.argmax(validation_outputs, dim=1)
                        validation_accuracy = 100 * torch.sum(predictions == validation_values).item() / len(validation_values)
                        validation_results[val_id] = validation_accuracy
                    # store the mean and std of the validation accuracies
                    results[(db, gnn_algorithm, config_id)] = validation_results
    # Print the results in a table format (columns dbs, rows gnn_algos) with mean and std of validation accuracies (config_id with highest mean accuracy)
    for db in dbs:
        print(f"Results for dataset {db}:")
        print("GNN Algorithm\tMean Validation Accuracy (%)\tStd Validation Accuracy (%)")
        num_validation_folds = len(config['splits']['validation'])
        for gnn_algorithm in gnn_algos:
            best_mean_accuracy = 0
            best_std_accuracy = 0
            for config_id in range(len(run_configs)):
                if (db, gnn_algorithm, config_id) in results:
                    accuracies = [results[(db, gnn_algorithm, config_id)][val_id] for val_id in range(num_validation_folds)]
                    mean_accuracy = sum(accuracies) / len(accuracies)
                    std_accuracy = (sum((x - mean_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5
                    if mean_accuracy > best_mean_accuracy:
                        best_mean_accuracy = mean_accuracy
                        best_std_accuracy = std_accuracy
            print(f"{gnn_algorithm}\t{best_mean_accuracy:.2f}\t{best_std_accuracy:.2f}")
        print("\n")



def decision_flips_boxplots(dbs, gnn_algos, data_path: Path, save_path: Path):
    # load the all_results.csv to polars
    ds = polars.read_csv(data_path)
    # filter the dataset for the given dbs and gnn_algos
    filtered_ds = ds.filter(
        (ds['dataset'].is_in(dbs)) & (ds['gnn_algorithm'].is_in(gnn_algos))
    )
    # get only correctly classified paths
    filtered_ds = filtered_ds.filter(filtered_ds['is_correct_path'] == True)

    # make three versions all, train and validation
    for path_type in ['all', 'train', 'validation']:
        if path_type == 'train':
            filtered_ds = filtered_ds.filter(filtered_ds['is_train_path'] == True)
        elif path_type == 'validation':
            filtered_ds = filtered_ds.filter(filtered_ds['is_train_path'] == False)
        else:
            filtered_ds = filtered_ds


        # make groups from path_strategy, gnn_algorithm, dataset
        grouped = filtered_ds.group_by(['path_strategy', 'gnn_algorithm', 'dataset', 'config_id'])
        # for each group, do the following:
        # six operations
        operation_name_map = {
            'EDGE INSERT': 0,
            'EDGE DELETE': 1,
            'NODE INSERT': 2,
            'NODE DELETE': 3,
            'EDGE RELABEL': 4,
            'NODE RELABEL': 5,
        }
        results_flipping_operation_counts = {path_strategy: {db : { gnn_algo :  torch.zeros(6,10) for gnn_algo in gnn_algos} for db in dbs } for path_strategy in ['Rnd', 'i-E_d-IsoN']}
        results_flipping_relative_operation_counts = {path_strategy: {db : { gnn_algo : torch.zeros(6,10) for gnn_algo in gnn_algos} for db in dbs } for path_strategy in ['Rnd', 'i-E_d-IsoN']}
        for group_name, group_data in grouped:
            # group by val_id
            validation_groups = group_data.group_by(['val_id'])
            # for each validation group, get the decision_change_rate
            for val_group_name, val_group_data in validation_groups:
                # get counts for each operation_str
                operation_counts = val_group_data.group_by('operation_str').count()
                # get only rows where is_flipping == True
                flipping_ds = val_group_data.filter(val_group_data['is_flipping'] == True)
                # get counts for each operation_str
                flipping_operation_counts = flipping_ds.group_by('operation_str').count()
                for operation_str, count in flipping_operation_counts.iter_rows():
                    if operation_str != 'NONE':
                        op_index = operation_name_map[operation_str]
                        results_flipping_operation_counts[group_name[0]][group_name[2]][group_name[1]][op_index][val_group_name[0]] = count
                for operation_str, count in flipping_operation_counts.iter_rows():
                    op_index = operation_name_map[operation_str]
                    # get the value of the count column from the row operation_str in operation_counts if the row exists, else 0
                    if operation_str != 'NONE' and operation_str in operation_counts['operation_str'].to_list():
                        total_count = operation_counts.filter(operation_counts['operation_str'] == operation_str)['count'][0]
                    else:
                        total_count = 0
                    results_flipping_relative_operation_counts[group_name[0]][group_name[2]][group_name[1]][op_index][val_group_name[0]] = count / total_count
                pass
            pass
        # plot different figures

        ## First figure: for each group y-axis operation counts, x-axis one boxplot per operation, data over the 10 val_ids
        for group_name, _ in grouped:
            data = results_flipping_operation_counts[group_name[0]][group_name[2]][group_name[1]]
            import matplotlib.pyplot as plt
            from matplotlib import cm
            plt.figure()
            # choose a pleasant categorical colormap for the six operations
            cmap = cm.get_cmap('Set2')

            # determine which operations to include: drop 'EDGE RELABEL' if it has no counts for this group
            operation_names = list(operation_name_map.keys())
            op_indices = [operation_name_map[name] for name in operation_names]
            edge_relabel_idx = operation_name_map.get('EDGE RELABEL', None)
            include_ops = op_indices.copy()
            if edge_relabel_idx is not None and data[edge_relabel_idx].sum() == 0:
                include_ops = [i for i in include_ops if i != edge_relabel_idx]

            # prepare data and colors for included operations
            data_to_plot = [data[i].numpy() for i in include_ops]
            colors = [cmap(i) for i in range(len(include_ops))]

            # create boxplot with colored boxes
            bp = plt.boxplot(data_to_plot, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors[i])
                box.set_edgecolor('black')
            for median in bp['medians']:
                median.set_color('white')
                median.set_linewidth(1.5)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
            for cap in bp['caps']:
                cap.set_color('black')
            for flier in bp.get('fliers', []):
                flier.set_markeredgecolor('black')

            # rotate the labels by 15 degrees; set xticks only for included operations
            xtick_labels = [name for name in operation_names if operation_name_map[name] in include_ops]
            plt.xticks(range(1, len(xtick_labels) + 1), xtick_labels, rotation=15, ha='right')
            plt.ylabel('Number of Decision Changes')
            # mkdir Flips_Per_Operation if not exists
            output_dir = save_path.joinpath('Flips_Per_Operation')
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_dir.joinpath(f'{path_type}_{group_name[0]}_{group_name[2]}_{group_name[1]}_flips_per_operation.png'))
            plt.close()

        # Second figure: for each group y-axis relative operation counts, x-axis one boxplot per operation, data over the 10 val_ids
        for group_name, _ in grouped:
            data = results_flipping_relative_operation_counts[group_name[0]][group_name[2]][group_name[1]]
            import matplotlib.pyplot as plt
            from matplotlib import cm
            plt.figure()
            # use a different but consistent colormap for the percentage plots
            cmap2 = cm.get_cmap('Pastel1')

            # determine which operations to include: drop 'EDGE RELABEL' if it has no relative counts for this group
            operation_names = list(operation_name_map.keys())
            op_indices = [operation_name_map[name] for name in operation_names]
            edge_relabel_idx = operation_name_map.get('EDGE RELABEL', None)
            include_ops = op_indices.copy()
            if edge_relabel_idx is not None and data[edge_relabel_idx].sum() == 0:
                include_ops = [i for i in include_ops if i != edge_relabel_idx]

            data_to_plot = [data[i].numpy() for i in include_ops]
            colors2 = [cmap2(i) for i in range(len(include_ops))]
            bp = plt.boxplot(data_to_plot, patch_artist=True)
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(colors2[i])
                box.set_edgecolor('black')
            for median in bp['medians']:
                median.set_color('white')
                median.set_linewidth(1.5)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
            for cap in bp['caps']:
                cap.set_color('black')
            for flier in bp.get('fliers', []):
                flier.set_markeredgecolor('black')

            # rotate the labels by 15 degrees; set xticks only for included operations
            xtick_labels = [name for name in operation_names if operation_name_map[name] in include_ops]
            plt.xticks(range(1, len(xtick_labels) + 1), xtick_labels, rotation=15, ha='right')
            plt.ylabel('Percentage of Decision Changes')
            # mkdir Relative_Flips_Per_Operation if not exists
            output_dir = save_path.joinpath('Relative_Flips_Per_Operation')
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_dir.joinpath(f'{path_type}_{group_name[0]}_{group_name[2]}_{group_name[1]}_relative_flips_per_operation.png'))
            plt.close()

        # Third figure: grouped by operation, one tick per operation, colors per algorithm and a legend
        for path_strategy in ['Rnd', 'i-E_d-IsoN']:
            for db in dbs:
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches
                plt.figure()
                # decide which operations to include globally for this path_strategy/db: drop 'EDGE RELABEL' if all zeros across algos
                operation_names = list(operation_name_map.keys())
                op_indices = [operation_name_map[name] for name in operation_names]
                edge_relabel_idx = operation_name_map.get('EDGE RELABEL', None)
                include_ops = op_indices.copy()
                if edge_relabel_idx is not None:
                    total = 0.0
                    for gnn_algorithm in gnn_algos:
                        total += results_flipping_relative_operation_counts[path_strategy][db][gnn_algorithm][edge_relabel_idx].sum().item()
                    if total == 0.0:
                        include_ops = [i for i in include_ops if i != edge_relabel_idx]

                n_ops = len(include_ops)
                n_algos = len(gnn_algos)

                # prepare data: for each included operation and each algorithm grab the 10-val_id array
                data_groups = []  # will hold arrays in order (op0_algo0, op0_algo1, ..., op1_algo0, ...)
                positions = []
                gap = 1  # gap between groups
                width = 0.6
                for op_group_idx, op_idx in enumerate(include_ops):
                    for algo_idx, gnn_algorithm in enumerate(gnn_algos):
                        data = results_flipping_relative_operation_counts[path_strategy][db][gnn_algorithm][op_idx].numpy()
                        data_groups.append(data)
                        # compute position: group spacing of (n_algos + gap)
                        pos = op_group_idx * (n_algos + gap) + algo_idx
                        positions.append(pos)

                # choose colors (fall back to default cycle if fewer colors available)
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0', 'C1', 'C2', 'C3'])
                colors = [color_cycle[i % len(color_cycle)] for i in range(n_algos)]

                # create boxplot with explicit positions
                bp = plt.boxplot(data_groups, positions=positions, widths=width, patch_artist=True)

                # color the boxes, medians, caps and whiskers according to algorithm index
                for i, box in enumerate(bp['boxes']):
                    algo_idx = i % n_algos
                    col = colors[algo_idx]
                    box.set_facecolor(col)
                    box.set_edgecolor('black')
                for i, whisker in enumerate(bp['whiskers']):
                    whisker.set_color('black')
                for i, cap in enumerate(bp['caps']):
                    cap.set_color('black')
                for i, median in enumerate(bp['medians']):
                    median.set_color('white')
                    median.set_linewidth(1.5)
                for i, flier in enumerate(bp.get('fliers', [])):
                    flier.set_markeredgecolor('black')

                # set one xtick per included operation at the center of the group
                xtick_positions = [op_group_idx * (n_algos + gap) + (n_algos - 1) / 2 for op_group_idx in range(n_ops)]
                xtick_labels = [name for name in operation_names if operation_name_map[name] in include_ops]
                plt.xticks(xtick_positions, xtick_labels, rotation=15, ha='right')
                plt.ylabel('Percentage of Decision Changes')

                # add legend for algorithms using colored patches
                handles = [mpatches.Patch(facecolor=colors[i], edgecolor='black', label=gnn_algos[i]) for i in range(n_algos)]
                plt.legend(handles=handles, title='GNN Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')

                # mkdir Combined_Relative_Flips_Per_Operation if not exists
                output_dir = save_path.joinpath('Combined_Relative_Flips_Per_Operation')
                output_dir.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                plt.savefig(output_dir.joinpath(f'{path_type}_{path_strategy}_{db}_combined_relative_flips_per_operation.png'))
                plt.close()


    pass

def decision_class_changes_statistics(dbs, gnn_algos, data_path: Path, save_path: Path):

    operation_name_map = {
        0: 'EDGE INSERT',
        1: 'EDGE DELETE',
        2: 'NODE INSERT',
        3: 'NODE DELETE',
        4: 'EDGE RELABEL',
        5: 'NODE RELABEL',
    }

    # load the all_results.csv to polars
    ds = polars.read_csv(data_path)
    # filter the dataset for the given dbs and gnn_algos
    filtered_ds = ds.filter(
        (ds['dataset'].is_in(dbs)) & (ds['gnn_algorithm'].is_in(gnn_algos))
    )
    # get only correctly classified paths
    filtered_ds = filtered_ds.filter(filtered_ds['is_correct_path'] == True)
    # iterate over groups using dbs, gnn_algos, path_strategy, config_id
    results = {path_strategy: {db : { gnn_algo : torch.zeros(6,) for gnn_algo in gnn_algos} for db in dbs } for path_strategy in ['Rnd', 'i-E_d-IsoN']}
    for group_name, group_data in filtered_ds.group_by(['path_strategy', 'gnn_algorithm', 'dataset', 'config_id']):
        # get the class columns (data)
        class_columns = [col for col in filtered_ds.columns if col.startswith('class_')]
        class_columns_data = filtered_ds.select(class_columns)

        # subtract the class_columns_data shifted by one from the original class_columns_data to get the changes (first row null)

        class_column_data_change = class_columns_data - class_columns_data.shift(1)
        for i, col in enumerate(class_columns):
            filtered_ds = filtered_ds.with_columns(class_column_data_change[col].alias(f'{col}_changed'))
        # now filtered_ds has new columns indicating whether the class data changed after the operation
        # group by not NONE operations using operation_str and take the mean resp. std over the class_x change columns
        filtered_ds = filtered_ds.filter(filtered_ds['operation_str'] != 'NONE')
        # get operation_str, and class_x_changed columns
        filtered_ds = filtered_ds.select(
            ['operation_str'] + [f'{col}_changed' for col in class_columns] + ['path_strategy', 'gnn_algorithm',
                                                                               'dataset', 'config_id'])
        # get absolute changes
        for col in class_columns:
            filtered_ds = filtered_ds.with_columns(polars.col(f'{col}_changed').abs().alias(f'{col}_changed'))

        # aggregate by operation_str and compute mean and std for each class_x_changed column
        result = filtered_ds.group_by("operation_str").agg([
            expr
            for col in class_columns
            for expr in [
                polars.col(f"{col}_changed").mean().alias(f"{col}_changed_mean"),
                polars.col(f"{col}_changed").std().alias(f"{col}_changed_std"),
            ]
        ])
        # iterate over result rows and fill the results dict
        for row in result.iter_rows():
            operation_str = row[0]
            if operation_str in operation_name_map.values():
                op_index = list(operation_name_map.values()).index(operation_str)
                # sum the means over all classes
                mean_sum = sum(row[1::2])  # mean columns are at odd indices
                results[group_name[0]][group_name[2]][group_name[1]][op_index] = mean_sum
    # print the results
    # First figure: heatmap per path_strategy/db with gnn_algos on y-axis and operations on x-axis values are mean class changes
    for path_strategy in ['Rnd', 'i-E_d-IsoN']:
        for db in dbs:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure()
            data = np.zeros((len(gnn_algos), 6))
            for algo_idx, gnn_algorithm in enumerate(gnn_algos):
                for op_idx in range(6):
                    data[algo_idx, op_idx] = results[path_strategy][db][gnn_algorithm][op_idx]
            im = plt.imshow(data, cmap='viridis')

            # Show all ticks and label them
            plt.xticks(ticks=np.arange(6), labels=[operation_name_map[i] for i in range(6)], rotation=15, ha='right')
            plt.yticks(ticks=np.arange(len(gnn_algos)), labels=gnn_algos)

            # Loop over data dimensions and create text annotations.
            for i in range(len(gnn_algos)):
                for j in range(6):
                    text = plt.text(j, i, f"{data[i, j]:.2f}",
                                   ha="center", va="center", color="w")

            plt.colorbar(im)
            plt.title(f'Mean Class Changes - Path Strategy: {path_strategy}, Dataset: {db}')
            # mkdir Class_Changes_Heatmaps if not exists
            output_dir = save_path.joinpath('Class_Changes_Heatmaps')
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_dir.joinpath(f'{path_strategy}_{db}_class_changes_heatmap.png'))
            plt.close()
    pass

def number_of_flips_statistics(dbs, gnn_algos, data_path: Path, save_path: Path):
    # load the all_results.csv to polars
    ds = polars.read_csv(data_path)
    # filter the dataset for the given dbs and gnn_algos
    filtered_ds = ds.filter(
        (ds['dataset'].is_in(dbs)) & (ds['gnn_algorithm'].is_in(gnn_algos))
    )
    # get only correctly classified paths
    filtered_ds = filtered_ds.filter(filtered_ds['is_correct_path'] == True)
    # make groups from path_strategy, gnn_algorithm, dataset, config_id
    grouped = filtered_ds.group_by(['path_strategy', 'gnn_algorithm', 'dataset', 'config_id'])
    results_total = {path_strategy: {db : { gnn_algo : torch.zeros(100,10) for gnn_algo in gnn_algos} for db in dbs } for path_strategy in ['Rnd', 'i-E_d-IsoN']}
    results_relative = {path_strategy: {db : { gnn_algo : torch.zeros(100,10) for gnn_algo in gnn_algos} for db in dbs } for path_strategy in ['Rnd', 'i-E_d-IsoN']}
    for group_name, group_data in grouped:
        # group by val_id
        validation_groups = group_data.group_by(['val_id'])
        for val_group_name, val_group_data in validation_groups:
            same_endpoint_paths = val_group_data.filter(polars.col('same_endpoint_labels_true') == 1)
            different_endpoint_paths = val_group_data.filter(polars.col('same_endpoint_labels_true') == 0)

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

            total_same_endpoint_paths = same_aggr_path.select(polars.n_unique('path_idx_int')).to_series()[0]
            total_different_endpoint_paths = different_aggr_path.select(polars.n_unique('path_idx_int')).to_series()[0]

            for row in same_stats.iter_rows():
                num_flips = row[0]
                mean_count = row[1]
                val_id = val_group_name[0]
                results_total[group_name[0]][group_name[2]][group_name[1]][num_flips][val_id] = mean_count
                if total_same_endpoint_paths > 0:
                    results_relative[group_name[0]][group_name[2]][group_name[1]][num_flips][val_id] = mean_count / total_same_endpoint_paths
            for row in different_stats.iter_rows():
                num_flips = row[0]
                mean_count = row[1]
                val_id = val_group_name[0]
                results_total[group_name[0]][group_name[2]][group_name[1]][num_flips][val_id] += mean_count
                if total_different_endpoint_paths > 0:
                    results_relative[group_name[0]][group_name[2]][group_name[1]][num_flips][val_id] += mean_count / total_different_endpoint_paths

    pass

def algorithm_results(dbs, gnn_algos, data_path: Path):
    # load the all_results.csv to polars
    ds = polars.read_csv(data_path)
    # filter the dataset for the given dbs and gnn_algos
    filtered_ds = ds.filter(
        (ds['dataset'].is_in(dbs)) & (ds['gnn_algorithm'].is_in(gnn_algos))
    )
    # make groups from path_strategy, gnn_algorithm, dataset
    grouped = filtered_ds.group_by(['path_strategy', 'gnn_algorithm', 'dataset'])
    # compute mean and std of test_accuracy for each group
    pass


@click.command()
@click.option(
    "--db",
    "dbs",
    multiple=True,
    default=("MUTAG", ),
    show_default=True,
    help="Dataset to include. Repeat the option to pass multiple datasets.",
)
@click.option(
    "--gnn-algo",
    "gnn_algos",
    multiple=True,
    default=("GCN", "GATv2", "GraphSAGE", "GIN"),
    show_default=True,
    help="GNN algorithm to include. Repeat the option to pass multiple algorithms.",
)
@click.option(
    "--data-path",
    type=click.Path(path_type=Path),
    default=Path("results/all_results.csv"),
    show_default=True,
    help="Path to the consolidated results CSV file.",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("results/Plots/"),
    show_default=True,
    help="Directory where plots are written.",
)
def main(dbs, gnn_algos, data_path, output_path):
    dbs = list(dbs)
    gnn_algos = list(gnn_algos)
    if not data_path.exists():
        print("Results file not found. Please run the training script first to generate results.")
        return

    table_accuracies(dbs, gnn_algos)
    algorithm_results(dbs, gnn_algos, data_path)
    decision_flips_boxplots(dbs, gnn_algos, data_path=data_path, save_path=output_path)
    decision_class_changes_statistics(dbs, gnn_algos, data_path=data_path, save_path=output_path)
    number_of_flips_statistics(dbs, gnn_algos, data_path=data_path, save_path=output_path)


if __name__ == '__main__':
    main()
