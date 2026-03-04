## Real World Data
import sys
from pathlib import Path

import click
try:
    from click.core import ParameterSource
except ImportError:  # pragma: no cover - compatibility fallback for older Click
    ParameterSource = None

CONFIGS_ROOT = Path(__file__).resolve().parent / "configs"


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


def _list_available_dbs(configs_root=CONFIGS_ROOT):
    if not configs_root.exists():
        return []

    return sorted(
        entry.name
        for entry in configs_root.iterdir()
        if entry.is_dir() and entry.joinpath("main_config.yml").is_file()
    )


def _validate_requested_dbs(dbs, configs_root=CONFIGS_ROOT):
    available_dbs = _list_available_dbs(configs_root)
    requested_dbs = list(dbs)

    missing_dbs = [
        db for db in requested_dbs if not configs_root.joinpath(db, "main_config.yml").is_file()
    ]
    if missing_dbs:
        available_text = ", ".join(available_dbs) if available_dbs else "<none>"
        missing_text = ", ".join(missing_dbs)
        raise click.ClickException(
            f"Missing config(s) for db: {missing_text}. "
            f"Expected 'configs/<db>/main_config.yml'. Available db options: {available_text}"
        )


def train_ged(num_threads=-1, dbs=("MUTAG",)):
    _validate_requested_dbs(dbs)
    framework_main_cls = _import_framework_main()

    for db in dbs:
        # Load and preprocess the experiment
        experiment = framework_main_cls(Path(f"configs/{db}/main_config.yml"))
        experiment.preprocessing(num_threads=num_threads)

        # Run and evaluate all configurations defined in the config file
        experiment.run_configurations(num_threads=num_threads)
        experiment.evaluate_results(evaluate_validation_only=True)


AVAILABLE_DBS = _list_available_dbs()
AVAILABLE_DBS_HELP = ", ".join(AVAILABLE_DBS) if AVAILABLE_DBS else "<none found>"


@click.command()
@click.option("--num_threads", default=-1, help="Number of threads to use")
@click.option(
    "--all",
    "train_all",
    is_flag=True,
    help="Train all datasets discovered under configs/*/main_config.yml.",
)
@click.option(
    "--db",
    "dbs",
    multiple=True,
    default=("MUTAG",),
    show_default=True,
    help=(
        "Dataset/config folder under configs to train. Repeat --db to train multiple datasets. "
        f"Available options from configs: {AVAILABLE_DBS_HELP}"
    ),
)
@click.pass_context
def main(ctx, num_threads, train_all, dbs):
    if hasattr(ctx, "get_parameter_source") and ParameterSource is not None:
        parameter_source = ctx.get_parameter_source("dbs")
        dbs_given_explicitly = parameter_source != ParameterSource.DEFAULT
    else:
        dbs_given_explicitly = any(
            arg == "--db" or arg.startswith("--db=") for arg in sys.argv[1:]
        )

    if train_all and dbs_given_explicitly:
        raise click.ClickException("Options '--all' and '--db' are mutually exclusive.")

    if train_all:
        selected_dbs = _list_available_dbs(CONFIGS_ROOT)
        if not selected_dbs:
            raise click.ClickException(
                "No dataset configs found for '--all'. "
                "Expected at least one 'configs/<db>/main_config.yml'."
            )
    else:
        selected_dbs = list(dbs)

    train_ged(num_threads=num_threads, dbs=selected_dbs)


if __name__ == "__main__":
    main()
