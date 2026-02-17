from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import click


ALLOWED_MAPPING_STRATEGIES = {"F2", "Precomputed"}


def _extract_path_strategy(paths_dir_name: str) -> str:
    """Extract strategy from directory names like 'Paths_Rnd'."""
    if paths_dir_name.startswith("Paths_"):
        return paths_dir_name[len("Paths_"):]
    if paths_dir_name == "Paths":
        return "Paths"
    return paths_dir_name.removeprefix("Paths").lstrip("_-")


def copy_ged_graphs(
    source_root: str = "../GNNGED/Results",
    dest_root: str = "data/GEDGraphs",
    skip_existing: bool = False,
) -> Dict[str, object]:
    source_root_path = Path(source_root).resolve()
    dest_root_path = Path(dest_root).resolve()

    if not source_root_path.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root_path}")

    paths_dirs = sorted(
        path
        for path in source_root_path.iterdir()
        if path.is_dir() and path.name.startswith("Paths")
    )
    if not paths_dirs:
        raise RuntimeError(f"No 'Paths*' directories found under: {source_root_path}")

    copied: List[str] = []
    skipped_missing_folders: List[str] = []
    skipped_missing_requirements: Dict[str, List[str]] = {}
    skipped_existing: List[str] = []
    ignored_mapping_dirs: List[str] = []

    for paths_dir in paths_dirs:
        path_strategy = _extract_path_strategy(paths_dir.name)
        if not path_strategy:
            continue

        mapping_dirs = sorted(path for path in paths_dir.iterdir() if path.is_dir())
        for mapping_dir in mapping_dirs:
            if mapping_dir.name not in ALLOWED_MAPPING_STRATEGIES:
                ignored_mapping_dirs.append(str(mapping_dir))
                continue

            db_dirs = sorted(path for path in mapping_dir.iterdir() if path.is_dir())
            for db_dir in db_dirs:
                processed_dir = db_dir / "processed"
                raw_dir = db_dir / "raw"
                edit_paths_file = db_dir / f"{db_dir.name}_edit_paths_data.txt"

                missing_requirements: List[str] = []
                if not processed_dir.is_dir():
                    missing_requirements.append("processed")
                if not raw_dir.is_dir():
                    missing_requirements.append("raw")
                if not edit_paths_file.is_file():
                    missing_requirements.append(edit_paths_file.name)

                if missing_requirements:
                    skipped_missing_folders.append(str(db_dir))
                    skipped_missing_requirements[str(db_dir)] = missing_requirements
                    continue

                target_db_dir = (
                    dest_root_path / mapping_dir.name / f"{db_dir.name}_{path_strategy}"
                )
                target_processed_dir = target_db_dir / "processed"
                target_raw_dir = target_db_dir / "raw"
                target_edit_paths_file = target_db_dir / edit_paths_file.name

                if (
                    skip_existing
                    and target_processed_dir.is_dir()
                    and target_raw_dir.is_dir()
                    and target_edit_paths_file.is_file()
                ):
                    skipped_existing.append(str(target_db_dir))
                    continue

                if target_db_dir.exists():
                    shutil.rmtree(target_db_dir)
                target_db_dir.mkdir(parents=True, exist_ok=True)

                shutil.copytree(processed_dir, target_processed_dir)
                shutil.copytree(raw_dir, target_raw_dir)
                shutil.copy2(edit_paths_file, target_edit_paths_file)
                copied.append(str(target_db_dir))

    if not copied and not skipped_existing:
        raise RuntimeError(
            "No datasets copied. Verify source folders contain mapping strategy directories "
            "plus required 'processed', 'raw', and '<db>_edit_paths_data.txt' items."
        )

    return {
        "source_root": str(source_root_path),
        "dest_root": str(dest_root_path),
        "copied_count": len(copied),
        "copied": copied,
        "skipped_existing_count": len(skipped_existing),
        "skipped_existing": skipped_existing,
        "skipped_missing_folders_count": len(skipped_missing_folders),
        "skipped_missing_folders": skipped_missing_folders,
        "skipped_missing_requirements": skipped_missing_requirements,
        "ignored_mapping_dirs_count": len(ignored_mapping_dirs),
        "ignored_mapping_dirs": ignored_mapping_dirs,
    }


@click.command()
@click.option(
    "--source-root",
    default="../GNNGED/Results",
    show_default=True,
    help="Directory that contains Paths* results folders.",
)
@click.option(
    "--dest-root",
    default="data/GEDGraphs",
    show_default=True,
    help="Destination base directory for copied GED graph folders.",
)
@click.option(
    "--skip-existing/--overwrite-existing",
    default=False,
    show_default=True,
    help="Skip datasets already fully copied at destination.",
)
def main(source_root: str, dest_root: str, skip_existing: bool) -> None:
    result = copy_ged_graphs(
        source_root=source_root,
        dest_root=dest_root,
        skip_existing=skip_existing,
    )
    print(f"Copied datasets: {result['copied_count']}")
    print(f"Skipped already copied datasets: {result['skipped_existing_count']}")
    print(f"Skipped incomplete dataset folders: {result['skipped_missing_folders_count']}")
    missing_edit_paths_files = sum(
        1
        for missing in result["skipped_missing_requirements"].values()
        if any(name.endswith("_edit_paths_data.txt") for name in missing)
    )
    print(f"Skipped datasets missing edit-path file: {missing_edit_paths_files}")
    print(f"Ignored mapping directories: {result['ignored_mapping_dirs_count']}")


if __name__ == "__main__":
    main()
