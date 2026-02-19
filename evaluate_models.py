import itertools
import sys
from pathlib import Path

import click
import joblib
import torch

from copy_ged_graphs import copy_ged_graphs


_SIMPLEGNN_SYMBOLS = None


def _load_simplegnn_symbols():
    global _SIMPLEGNN_SYMBOLS
    if _SIMPLEGNN_SYMBOLS is not None:
        return _SIMPLEGNN_SYMBOLS

    try:
        from simplegnn.framework.core import FrameworkMain, preprocess_graph_data
        from simplegnn.framework.model_configuration import ModelConfiguration
        from simplegnn.framework.run_configuration import get_run_configs
        from simplegnn.framework.utils.load_model import load_model
        from simplegnn.framework.utils.parameters import Parameters
        from simplegnn.framework.utils.preprocessing import (
            load_preprocessed_data_and_parameters,
            load_splits,
        )
    except ImportError:
        repo_root = Path(__file__).resolve().parent
        simplegnn_src = repo_root.parent / "SimpleGNN" / "repo" / "src"
        sys.path.append(str(simplegnn_src))
        try:
            from simplegnn.framework.core import FrameworkMain, preprocess_graph_data
            from simplegnn.framework.model_configuration import ModelConfiguration
            from simplegnn.framework.run_configuration import get_run_configs
            from simplegnn.framework.utils.load_model import load_model
            from simplegnn.framework.utils.parameters import Parameters
            from simplegnn.framework.utils.preprocessing import (
                load_preprocessed_data_and_parameters,
                load_splits,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import simplegnn framework symbols. "
                f"Expected local source at '{simplegnn_src}' or an installed 'simplegnn' package."
            ) from exc

    _SIMPLEGNN_SYMBOLS = {
        "FrameworkMain": FrameworkMain,
        "ModelConfiguration": ModelConfiguration,
        "Parameters": Parameters,
        "get_run_configs": get_run_configs,
        "load_model": load_model,
        "load_preprocessed_data_and_parameters": load_preprocessed_data_and_parameters,
        "load_splits": load_splits,
        "preprocess_graph_data": preprocess_graph_data,
    }
    return _SIMPLEGNN_SYMBOLS


def ensure_evaluation_dir(results_path, evaluation_folder):
    eval_dir = results_path.joinpath(evaluation_folder)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def to_text_output(output_value):
    if isinstance(output_value, torch.Tensor):
        return " ".join(str(value.item()) for value in output_value)
    return str(output_value)


def build_split_results_tensor(graph_ids, target_values, target_outputs):
    output_width = target_outputs[0].shape[0] if isinstance(target_outputs[0], torch.Tensor) else 1
    results_tensor = torch.empty((len(graph_ids), 2 + output_width))

    for index, (graph_id, target_value, output_value) in enumerate(
        zip(graph_ids, target_values, target_outputs)
    ):
        results_tensor[index, 0] = graph_id
        results_tensor[index, 1] = target_value
        if isinstance(output_value, torch.Tensor):
            results_tensor[index, 2:] = output_value
        else:
            results_tensor[index, 2] = output_value

    return results_tensor


def save_split_results(configuration, evaluation_folder, split_name, db, config_id, val_id, graph_ids, target_values, target_outputs):
    eval_dir = ensure_evaluation_dir(configuration.results_path, evaluation_folder)

    split_tensor = build_split_results_tensor(graph_ids, target_values, target_outputs)
    torch.save(
        split_tensor,
        eval_dir.joinpath(f"{split_name}_results_config{config_id}_val{val_id}_{db}.pt"),
    )

    with open(eval_dir.joinpath(f"{split_name}_results_config{config_id}_val{val_id}_{db}.txt"), "w") as file_obj:
        file_obj.write("graph_id\ttarget_value\toutput_value\n")
        for graph_id, target_value, output_value in sorted(
            zip(graph_ids, target_values, target_outputs), key=lambda row: row[0]
        ):
            file_obj.write(
                f"{graph_id}\t{target_value}\t{to_text_output(output_value)}\n"
            )


def load_operation_information(operation_file):
    with open(operation_file, "r") as file_obj:
        return file_obj.readlines()


def get_operation_id(operation_element, operation_type):
    return {
        "NODE_INSERT": 0,
        "NODE_DELETE": 1,
        "NODE_RELABEL": 2,
        "EDGE_INSERT": 3,
        "EDGE_DELETE": 4,
        "EDGE_RELABEL": 5,
    }.get(f"{operation_element}_{operation_type}", -2)


def build_path_results_tensor(num_path_graphs, target_outputs, operation_information):
    output_width = target_outputs[0].shape[0] if isinstance(target_outputs[0], torch.Tensor) else 1
    path_results_tensor = torch.empty((num_path_graphs, 4 + output_width))

    skip = 0
    for index, operation_line in enumerate(operation_information):
        if index + skip >= len(target_outputs):
            break

        target_output = target_outputs[index + skip]
        operation_line = operation_line.strip()
        if operation_line.startswith("#") or operation_line == "":
            continue

        parts = operation_line.split(" ")
        source_id = int(parts[0])
        step_id = parts[1]
        target_id = int(parts[2])
        operation_element = parts[3]
        operation_type = parts[5]

        if step_id == "0":
            path_results_tensor[index + skip, 0] = source_id
            path_results_tensor[index + skip, 1] = -1
            path_results_tensor[index + skip, 2] = target_id
            path_results_tensor[index + skip, 3] = -1
            if isinstance(target_output, torch.Tensor):
                path_results_tensor[index + skip, 4:] = target_output
            else:
                path_results_tensor[index + skip, 4] = target_output

            skip += 1
            if index + skip >= len(target_outputs):
                break
            target_output = target_outputs[index + skip]

        operation_id = get_operation_id(operation_element, operation_type)
        if operation_id == -2:
            raise ValueError(f"Unknown operation type: {operation_element} {operation_type}")

        path_results_tensor[index + skip, 0] = source_id
        path_results_tensor[index + skip, 1] = int(step_id)
        path_results_tensor[index + skip, 2] = target_id
        path_results_tensor[index + skip, 3] = operation_id
        if isinstance(target_output, torch.Tensor):
            path_results_tensor[index + skip, 4:] = target_output
        else:
            path_results_tensor[index + skip, 4] = target_output

    return path_results_tensor


def save_path_results_txt(eval_dir, db, path_strategy, config_id, val_id, operation_information, target_outputs):
    output_path = eval_dir.joinpath(
        f"path_results_config{config_id}_val{val_id}_{db}_{path_strategy}.txt"
    )
    with open(output_path, "w") as file_obj:
        file_obj.write("source_id\tstep_id\ttarget_id\toperation\ttarget_value\n")
        skip = 0
        for index, operation_line in enumerate(operation_information):
            if index + skip >= len(target_outputs):
                break

            target_output = to_text_output(target_outputs[index + skip])
            operation_line = operation_line.strip()
            if operation_line.startswith("#") or operation_line == "":
                continue

            parts = operation_line.split(" ")
            source_id = parts[0]
            step_id = parts[1]
            target_id = parts[2]
            operation_element = parts[3]
            operation_value = parts[4]
            operation_type = parts[5]

            if step_id == "0":
                file_obj.write(f"{source_id}\tS\t{target_id}\t{target_output}\tNONE\n")
                skip += 1
                if index + skip >= len(target_outputs):
                    break
                target_output = to_text_output(target_outputs[index + skip])

            file_obj.write(
                f"{source_id}\t{step_id}\t{target_id}\t{target_output}\t"
                f"{operation_element} {operation_value} {operation_type}\n"
            )


def align_path_graph_features(graph_data, path_graph_data):
    all_path_graph_ids = list(
        set(
            path_graph_data.data.edit_path_start.tolist()
            + path_graph_data.data.edit_path_end.tolist()
        )
    )
    subset_list = [graph_data[index] for index in all_path_graph_ids]
    subset, _ = graph_data.collate(subset_list)
    zero_column_indices = (
        subset.x.abs().sum(dim=0).eq(0).nonzero(as_tuple=True)[0].tolist()
    )

    for col_idx in zero_column_indices:
        zero_column = torch.zeros(
            (path_graph_data.data.x.shape[0], 1), dtype=path_graph_data.data.x.dtype
        )
        path_graph_data._data.x = torch.cat(
            (
                path_graph_data.data.x[:, :col_idx],
                zero_column,
                path_graph_data.data.x[:, col_idx:],
            ),
            dim=1,
        )


def evaluate_single_task(num_threads=1, db="MUTAG", path_strategy="i-E_d-IsoN", gnn_algorithm=None, evaluation_folder="Evaluation"):
    symbols = _load_simplegnn_symbols()
    FrameworkMain = symbols["FrameworkMain"]
    preprocess_graph_data = symbols["preprocess_graph_data"]
    get_run_configs = symbols["get_run_configs"]
    load_splits = symbols["load_splits"]
    load_model = symbols["load_model"]
    Parameters = symbols["Parameters"]
    load_preprocessed_data_and_parameters = symbols["load_preprocessed_data_and_parameters"]
    ModelConfiguration = symbols["ModelConfiguration"]

    experiment_base = FrameworkMain(Path(f"configs/{db}/main_config.yml"))
    experiment_base.preprocessing(num_threads=num_threads)

    experiment_paths = FrameworkMain(Path(f"configs/{db}/paths_config.yml"))
    experiment_paths.preprocessing(num_threads=num_threads)

    run_id = 0

    for db_id, config in enumerate(experiment_base.network_configurations[db]):
        algorithm = str(config["paths"]["models"]).split("_")[1].replace(".yml", "")
        if gnn_algorithm is not None and algorithm != gnn_algorithm:
            continue

        split_data = load_splits(config["paths"]["splits"])
        train_ids = split_data["train"]
        validation_ids = split_data["validation"]
        test_ids = split_data["test"]

        graph_data = preprocess_graph_data(config)
        path_graph_data = preprocess_graph_data(
            experiment_paths.network_configurations[f"{db}_{path_strategy}"][0]
        )
        align_path_graph_features(graph_data, path_graph_data)

        run_configs = get_run_configs(config)
        for config_id, run_config in enumerate(run_configs):
            num_validation = len(run_config.config["splits"]["validation"])
            for val_id in range(num_validation):
                model = load_model(
                    experiment_configuration=experiment_base.network_configurations[db][db_id],
                    db_name=db,
                    config_id=config_id,
                    run_id=run_id,
                    validation_id=val_id,
                    best=False,
                )
                model.eval()

                para = Parameters()
                load_preprocessed_data_and_parameters(
                    config_id=config_id,
                    run_id=run_id,
                    validation_id=val_id,
                    validation_folds=config.get("validation_folds", 10),
                    graph_data=graph_data,
                    run_config=run_config,
                    para=para,
                )

                seed = 42 + val_id + para.n_val_runs * run_id
                configuration = ModelConfiguration(
                    run_id,
                    val_id,
                    graph_data,
                    (train_ids, validation_ids, test_ids),
                    seed,
                    para,
                )
                configuration.initialize_model(pretrained_network=model)

                print(
                    f"Evaluating model {algorithm} for config_id {config_id}, val_id {val_id} on training set"
                )
                train_values, train_outputs = configuration.evaluate_network(
                    graph_ids=train_ids[val_id], do_print=True, with_loss=True
                )
                save_split_results(
                    configuration=configuration,
                    evaluation_folder=evaluation_folder,
                    split_name="train",
                    db=db,
                    config_id=config_id,
                    val_id=val_id,
                    graph_ids=train_ids[val_id],
                    target_values=train_values,
                    target_outputs=train_outputs,
                )

                print(
                    f"Evaluating model {algorithm} for config_id {config_id}, val_id {val_id} on validation set"
                )
                validation_values, validation_outputs = configuration.evaluate_network(
                    graph_ids=validation_ids[val_id], do_print=True, with_loss=True
                )
                save_split_results(
                    configuration=configuration,
                    evaluation_folder=evaluation_folder,
                    split_name="validation",
                    db=db,
                    config_id=config_id,
                    val_id=val_id,
                    graph_ids=validation_ids[val_id],
                    target_values=validation_values,
                    target_outputs=validation_outputs,
                )

                path_graph_data.number_of_output_classes = graph_data.num_classes
                configuration_paths = ModelConfiguration(
                    run_id,
                    val_id,
                    path_graph_data,
                    (train_ids, validation_ids, test_ids),
                    seed,
                    para,
                )
                configuration_paths.initialize_model(pretrained_network=model)

                print(
                    f"Evaluating model {algorithm} on all path graphs for strategy {path_strategy} "
                    f"for config_id {config_id}, val_id {val_id}"
                )
                _, target_outputs = configuration_paths.evaluate_network(
                    graph_ids=list(range(len(path_graph_data)))
                )

                operation_info_file = (
                    Path(experiment_paths.main_config["datasets"][0]["paths"]["data"])
                    .joinpath(f"{db}_{path_strategy}")
                    .joinpath(f"{db}_edit_paths_data.txt")
                )
                operation_information = load_operation_information(operation_info_file)

                eval_dir = ensure_evaluation_dir(
                    configuration_paths.results_path, evaluation_folder
                )
                path_results_tensor = build_path_results_tensor(
                    num_path_graphs=len(path_graph_data),
                    target_outputs=target_outputs,
                    operation_information=operation_information,
                )
                torch.save(
                    path_results_tensor,
                    eval_dir.joinpath(
                        f"path_results_config{config_id}_val{val_id}_{db}_{path_strategy}.pt"
                    ),
                )
                save_path_results_txt(
                    eval_dir=eval_dir,
                    db=db,
                    path_strategy=path_strategy,
                    config_id=config_id,
                    val_id=val_id,
                    operation_information=operation_information,
                    target_outputs=target_outputs,
                )


def evaluate_gnn(num_threads=1, db="MUTAG", path_strategy="i-E_d-IsoN", gnn_algorithm=None, evaluation_folder="Evaluation"):
    evaluate_single_task(
        num_threads=num_threads,
        db=db,
        path_strategy=path_strategy,
        gnn_algorithm=gnn_algorithm,
        evaluation_folder=evaluation_folder,
    )


def run_evaluations(num_threads, dbs, path_strategies, gnn_algorithms, evaluation_folder):
    tasks = list(itertools.product(dbs, path_strategies, gnn_algorithms))
    if not tasks:
        raise ValueError("No evaluation tasks selected. Provide at least one db, path strategy, and algorithm.")

    torch.set_num_threads(1)
    joblib.Parallel(n_jobs=len(tasks))(
        joblib.delayed(evaluate_single_task)(
            num_threads=num_threads,
            db=db,
            path_strategy=path_strategy,
            gnn_algorithm=gnn_algorithm,
            evaluation_folder=evaluation_folder,
        )
        for db, path_strategy, gnn_algorithm in tasks
    )


@click.command()
@click.option("--num_threads", default=1, help="Number of threads to use")
@click.option(
    "--copy-source-root",
    default="../GNNGED/Results",
    show_default=True,
    help="Source root used by copy_ged_graphs before evaluation starts",
)
@click.option(
    "--copy-dest-root",
    default="data/GEDGraphs",
    show_default=True,
    help="Destination root used by copy_ged_graphs before evaluation starts",
)
@click.option(
    "--copy-skip-existing/--copy-overwrite-existing",
    default=True,
    show_default=True,
    help="Skip datasets already present at destination when copying GED graphs",
)
@click.option(
    "--db",
    "dbs",
    multiple=True,
    default=("MUTAG",),
    show_default=True,
    help="Dataset to evaluate. Can be passed multiple times.",
)
@click.option(
    "--path-strategy",
    "path_strategies",
    multiple=True,
    default=("i-E_d-IsoN", "Rnd"),
    show_default=True,
    help="Path strategy to evaluate. Can be passed multiple times.",
)
@click.option(
    "--gnn-algorithm",
    "gnn_algorithms",
    multiple=True,
default=("GIN", "GAT", "GATv2", "GraphSAGE", "GCN"),
    show_default=True,
    help="GNN algorithm filter. Can be passed multiple times.",
)
@click.option(
    "--evaluation-folder",
    default="path_evaluation",
    show_default=True,
    help="Subfolder name under results path used to store evaluation outputs.",
)
def main(
    num_threads,
    copy_source_root,
    copy_dest_root,
    copy_skip_existing,
    dbs,
    path_strategies,
    gnn_algorithms,
    evaluation_folder,
):
    copy_result = copy_ged_graphs(
        source_root=copy_source_root,
        dest_root=copy_dest_root,
        skip_existing=copy_skip_existing,
    )
    print(f"Copied GED graph datasets: {copy_result['copied_count']}")
    print(
        "Skipped already copied GED graph datasets: "
        f"{copy_result['skipped_existing_count']}"
    )
    print(
        "Skipped incomplete dataset folders during copy: "
        f"{copy_result['skipped_missing_folders_count']}"
    )

    run_evaluations(
        num_threads=num_threads,
        dbs=list(dbs),
        path_strategies=list(path_strategies),
        gnn_algorithms=list(gnn_algorithms),
        evaluation_folder=evaluation_folder,
    )


if __name__ == "__main__":
    main()
