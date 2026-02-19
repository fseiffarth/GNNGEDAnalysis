## Real World Data
import sys
from pathlib import Path

import click
from simplegnn.framework.core import FrameworkMain


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



def train_ged(num_threads=-1, db="MUTAG"):
    # Load and preprocess the experiment
    experiment = FrameworkMain(Path(f"configs/{db}/main_config.yml"))
    experiment.preprocessing(num_threads=num_threads)

    # Run and evaluate all configurations defined in the config file
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results(evaluate_validation_only=True)

@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
@click.option(
    "--db",
    default="MUTAG",
    show_default=True,
    help="Dataset/config folder under configs to train.",
)
def main(num_threads, db):
    train_ged(num_threads=num_threads, db=db)



if __name__ == '__main__':
    main()
