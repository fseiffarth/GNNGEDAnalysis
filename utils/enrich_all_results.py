import glob
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import click
import polars as pl


METRIC_COLUMNS = [
    "n_connected_components",
    "n_cycles_len_3",
    "n_cycles_len_4",
    "n_cycles_len_5",
    "n_cycles_len_6",
]


def _as_python_list(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    if hasattr(value, "tolist"):
        return [int(v) for v in value.tolist()]
    return [int(v) for v in value]


def _get_attr(container, key: str):
    if isinstance(container, Mapping):
        return container.get(key)
    if hasattr(container, key):
        return getattr(container, key)
    return None


def _materialize_data_object(raw_data, data_cls=None):
    if isinstance(raw_data, Mapping) and data_cls is not None:
        from_dict = getattr(data_cls, "from_dict", None)
        if callable(from_dict):
            return from_dict(dict(raw_data))
    return raw_data


def _extract_data_and_slices(loaded_obj) -> Tuple[object, Mapping]:
    if isinstance(loaded_obj, tuple):
        if len(loaded_obj) == 2:
            return loaded_obj[0], loaded_obj[1]
        if len(loaded_obj) == 3:
            raw_data, slices, data_cls = loaded_obj
            return _materialize_data_object(raw_data, data_cls), slices
        if len(loaded_obj) >= 4:
            # Newer PyG variants may append extra metadata and/or data class:
            # (data, slices, ..., data_cls)
            raw_data, slices = loaded_obj[0], loaded_obj[1]
            data_cls = loaded_obj[-1] if isinstance(loaded_obj[-1], type) else None
            return _materialize_data_object(raw_data, data_cls), slices
    if isinstance(loaded_obj, Mapping):
        data_obj = loaded_obj.get("data")
        slices = loaded_obj.get("slices")
        if data_obj is not None and slices is not None:
            return data_obj, slices
    if isinstance(loaded_obj, tuple):
        detail = f"tuple length {len(loaded_obj)}"
    else:
        detail = f"type {type(loaded_obj).__name__}"
    raise ValueError(
        "Unsupported data.pt format: expected (data, slices), "
        "(data, slices, data_cls), tuple variants with at least (data, slices), "
        "or mapping with keys 'data' and 'slices' "
        f"(got {detail})."
    )


def _load_collated_dataset(data_pt_path: Path) -> Tuple[object, Mapping]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "This script requires torch to read processed/data.pt files."
        ) from exc

    loaded_obj = torch.load(data_pt_path, map_location="cpu", weights_only=False)
    return _extract_data_and_slices(loaded_obj)


def _resolve_node_slices(slices: Mapping) -> Optional[List[int]]:
    preferred_keys = ["x", "pos", "node_attr"]
    for key in preferred_keys:
        value = _get_attr(slices, key)
        if value is not None:
            return _as_python_list(value)
    return None


def _extract_graph_edges(
    data_obj, slices: Mapping, graph_idx: int
) -> Tuple[List[Tuple[int, int]], int]:
    edge_index = _get_attr(data_obj, "edge_index")
    if edge_index is None:
        raise ValueError("Could not find edge_index in collated data object.")

    edge_slices = _get_attr(slices, "edge_index")
    if edge_slices is None:
        raise ValueError("Could not find slices['edge_index'] in collated dataset.")

    edge_slices_list = _as_python_list(edge_slices)
    if graph_idx < 0 or graph_idx + 1 >= len(edge_slices_list):
        raise IndexError(
            f"Graph index {graph_idx} is out of range for edge slices of size {len(edge_slices_list)}."
        )

    edge_start = edge_slices_list[graph_idx]
    edge_end = edge_slices_list[graph_idx + 1]

    src = _as_python_list(edge_index[0][edge_start:edge_end])
    dst = _as_python_list(edge_index[1][edge_start:edge_end])
    if len(src) != len(dst):
        raise ValueError("edge_index source/target lengths differ.")

    node_slices = _resolve_node_slices(slices)
    if node_slices is not None and graph_idx + 1 < len(node_slices):
        node_start = node_slices[graph_idx]
        node_end = node_slices[graph_idx + 1]
        num_nodes = max(0, node_end - node_start)
        edges = [(u - node_start, v - node_start) for u, v in zip(src, dst)]
        return edges, num_nodes

    if src or dst:
        max_id = max(src + dst)
        return list(zip(src, dst)), max_id + 1
    return [], 0


def _build_simple_undirected_adjacency(
    edges: Sequence[Tuple[int, int]], num_nodes: int
) -> List[Set[int]]:
    adj = [set() for _ in range(num_nodes)]
    for u, v in edges:
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _count_connected_components(adj: Sequence[Set[int]]) -> int:
    n = len(adj)
    if n == 0:
        return 0
    visited = [False] * n
    components = 0

    for start in range(n):
        if visited[start]:
            continue
        components += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
    return components


def _count_cycles_of_length_k(adj: Sequence[Set[int]], k: int) -> int:
    if k < 3:
        return 0

    n = len(adj)
    count = 0

    def dfs(
        start: int, current: int, depth: int, visited: Set[int]
    ) -> None:
        nonlocal count
        if depth == k:
            if start in adj[current]:
                count += 1
            return

        for nxt in adj[current]:
            # Canonicalize by enforcing the smallest node in the cycle as start.
            if nxt < start:
                continue
            if nxt in visited:
                continue
            visited.add(nxt)
            dfs(start, nxt, depth + 1, visited)
            visited.remove(nxt)

    for start in range(n):
        visited = {start}
        dfs(start, start, 1, visited)

    # Each undirected cycle is counted twice (two traversal directions).
    return count // 2


def _resolve_data_pt_path(
    ged_root: Path, dataset: str, path_strategy: str
) -> Path:
    pattern = str(
        ged_root.joinpath("*", f"{dataset}_{path_strategy}", "processed", "data.pt")
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find processed/data.pt for pair ({dataset}, {path_strategy}) "
            f"under '{ged_root}'."
        )
    if len(matches) > 1:
        click.echo(
            f"Warning: multiple matches for ({dataset}, {path_strategy}); using '{matches[0]}'."
        )
    return Path(matches[0])


def _collect_required_graph_indices(input_csv: Path) -> pl.DataFrame:
    required_columns = {"dataset", "path_strategy", "row_idx"}
    schema = pl.scan_csv(str(input_csv)).collect_schema()
    missing = sorted(required_columns.difference(schema.names()))
    if missing:
        raise ValueError(
            f"Missing required column(s) in '{input_csv}': {', '.join(missing)}"
        )

    required = (
        pl.scan_csv(str(input_csv))
        .select(
            pl.col("dataset"),
            pl.col("path_strategy"),
            pl.col("row_idx").cast(pl.Int64).alias("row_idx_int"),
        )
        .unique()
        .collect()
    )
    return required


def _build_metrics_for_pair(
    dataset: str,
    path_strategy: str,
    graph_indices: Iterable[int],
    data_pt_path: Path,
) -> List[Dict[str, int]]:
    data_obj, slices = _load_collated_dataset(data_pt_path)
    rows: List[Dict[str, int]] = []

    sorted_indices = sorted(set(int(idx) for idx in graph_indices))
    for index_count, graph_idx in enumerate(sorted_indices, start=1):
        if index_count % 5000 == 0:
            click.echo(
                f"Processed {index_count}/{len(sorted_indices)} graph states for "
                f"({dataset}, {path_strategy})."
            )

        edges, num_nodes = _extract_graph_edges(data_obj, slices, graph_idx)
        adj = _build_simple_undirected_adjacency(edges, num_nodes)

        rows.append(
            {
                "dataset": dataset,
                "path_strategy": path_strategy,
                "row_idx_int": graph_idx,
                "n_connected_components": _count_connected_components(adj),
                "n_cycles_len_3": _count_cycles_of_length_k(adj, 3),
                "n_cycles_len_4": _count_cycles_of_length_k(adj, 4),
                "n_cycles_len_5": _count_cycles_of_length_k(adj, 5),
                "n_cycles_len_6": _count_cycles_of_length_k(adj, 6),
            }
        )

    return rows


def enrich_all_results(
    input_csv: Path,
    output_csv: Path,
    ged_root: Path,
    strict_missing: bool,
) -> None:
    required_df = _collect_required_graph_indices(input_csv)
    if required_df.height == 0:
        raise ValueError(f"No rows found in '{input_csv}'.")

    grouped = required_df.group_by(["dataset", "path_strategy"]).agg(
        pl.col("row_idx_int")
    )

    metrics_rows: List[Dict[str, int]] = []
    missing_pairs: List[Tuple[str, str]] = []
    for dataset, path_strategy, row_indices in grouped.iter_rows():
        try:
            data_pt_path = _resolve_data_pt_path(ged_root, dataset, path_strategy)
        except FileNotFoundError:
            missing_pairs.append((dataset, path_strategy))
            continue

        click.echo(
            f"Computing metrics for ({dataset}, {path_strategy}) from '{data_pt_path}'."
        )
        metrics_rows.extend(
            _build_metrics_for_pair(
                dataset=dataset,
                path_strategy=path_strategy,
                graph_indices=row_indices,
                data_pt_path=data_pt_path,
            )
        )

    if missing_pairs:
        message = (
            "Missing processed graph data for these dataset/strategy pairs: "
            + ", ".join(f"({d}, {s})" for d, s in missing_pairs)
        )
        if strict_missing:
            raise FileNotFoundError(message)
        click.echo(f"Warning: {message}")

    if not metrics_rows:
        raise ValueError("No graph metrics were computed; aborting.")

    metrics_df = pl.DataFrame(metrics_rows)

    enriched = (
        pl.scan_csv(str(input_csv))
        .with_columns(pl.col("row_idx").cast(pl.Int64).alias("row_idx_int"))
        .join(
            metrics_df.lazy(),
            on=["dataset", "path_strategy", "row_idx_int"],
            how="left",
        )
        .drop("row_idx_int")
    )

    if strict_missing:
        null_counts = enriched.select(
            [pl.col(col).is_null().sum().alias(col) for col in METRIC_COLUMNS]
        ).collect()
        unresolved = [
            col
            for col in METRIC_COLUMNS
            if int(null_counts.select(col).item()) > 0
        ]
        if unresolved:
            raise ValueError(
                "Enrichment produced null values in metric columns: "
                + ", ".join(unresolved)
            )

    try:
        enriched.sink_csv(str(output_csv))
    except AttributeError:
        enriched.collect(streaming=True).write_csv(str(output_csv))


@click.command()
@click.option(
    "--input-csv",
    default="results/all_results.csv",
    show_default=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the input all_results.csv file.",
)
@click.option(
    "--output-csv",
    default="results/all_results_enriched.csv",
    show_default=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Path to the enriched output CSV file.",
)
@click.option(
    "--ged-root",
    default="data/GEDGraphs",
    show_default=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Root folder containing GED graph datasets.",
)
@click.option(
    "--strict-missing/--no-strict-missing",
    default=True,
    show_default=True,
    help="Fail if required dataset/strategy graph data cannot be resolved.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Allow overwriting an existing output file.",
)
def main(
    input_csv: Path,
    output_csv: Path,
    ged_root: Path,
    strict_missing: bool,
    overwrite: bool,
) -> None:
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Output file '{output_csv}' already exists. Use --overwrite to replace it."
        )

    output_parent = output_csv.parent
    if output_parent and not output_parent.exists():
        os.makedirs(output_parent, exist_ok=True)

    enrich_all_results(
        input_csv=input_csv,
        output_csv=output_csv,
        ged_root=ged_root,
        strict_missing=strict_missing,
    )
    click.echo(f"Enriched results written to '{output_csv}'.")


if __name__ == "__main__":
    main()
