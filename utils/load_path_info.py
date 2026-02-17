"""Utilities to load edit-path information produced for GED examples.

This module defines EditAction and PathInfoLoader. It parses files with lines
in the format observed in
Examples/Data/GEDGraphs/MUTAG_i-E_d_IsoN/MUTAG_edit_paths_data.txt, e.g.:

0 0 2 EDGE 3--4 INSERT
0 1 2 EDGE 0--5 DELETE
...

Each line is interpreted as:
    <source_graph_id> <step_id> <target_graph_id> <ELEMENT_TYPE> <ELEMENT> <OPERATION>

- source_graph_id: integer identifying the source graph in the pair
- step_id: integer position of the edit within that solution/path
- target_graph_id: integer id of the target graph the edit belongs to
- ELEMENT_TYPE: either 'EDGE' or 'NODE'
- ELEMENT: for EDGE: 'u--v' (two ints joined by --), for NODE: a single int
- OPERATION: 'INSERT', 'DELETE', 'RELABEL', etc.

The loader returns a mapping: target_graph_id -> { source_graph_id -> [EditAction,...] }

The module also contains a small smoke-test when run as a script.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Iterable


@dataclass(frozen=True)
class EditAction:
    source_id: int
    step_id: int
    target_id: int
    element_type: str  # 'EDGE' or 'NODE'
    element: Union[int, Tuple[int, int]]  # node idx or (u, v)
    operation: str

    def __repr__(self) -> str:  # concise debug-friendly repr
        return (
            f"EditAction(src={self.source_id}, step={self.step_id}, tgt={self.target_id}, "
            f"type={self.element_type}, elem={self.element}, op={self.operation})"
        )


class PathInfoLoader:
    """Loader for edit-path files.

    Usage:
        loader = PathInfoLoader()
        mapping = loader.load(path)

    mapping is a dict: target_graph_id -> dict(source_graph_id -> list[EditAction])
    Each list of EditAction objects is sorted by step_id.
    """

    def __init__(self) -> None:
        pass

    def parse_line(self, line: str) -> EditAction:
        """Parse a single line and return an EditAction.

        Expected minimal tokens: 6
        Example tokens: ['0', '0', '2', 'EDGE', '3--4', 'INSERT']
        Interprets tokens as: source_id, step_id, target_id, element_type, element_token, operation
        """
        toks = line.strip().split()
        if not toks:
            raise ValueError("empty line")
        if len(toks) < 6:
            raise ValueError(f"unexpected format, need >=6 tokens, got {len(toks)}: {line!r}")

        try:
            source_id = int(toks[0])
            step_id = int(toks[1])
            target_id = int(toks[2])
        except Exception as e:
            raise ValueError(f"failed to parse ids in line: {line!r}") from e

        element_type = toks[3]
        element_token = toks[4]
        operation = toks[5]

        if element_type == 'EDGE':
            # Expect format 'u--v'
            if '--' not in element_token:
                raise ValueError(f"invalid edge token: {element_token!r} in line: {line!r}")
            u_str, v_str = element_token.split('--', 1)
            try:
                u = int(u_str)
                v = int(v_str)
            except Exception as e:
                raise ValueError(f"invalid edge integers in {element_token!r}") from e
            element = (u, v)
        elif element_type == 'NODE':
            try:
                element = int(element_token)
            except Exception as e:
                raise ValueError(f"invalid node token: {element_token!r}") from e
        else:
            # Unknown element type: keep token raw
            element = element_token

        return EditAction(
            source_id=source_id,
            step_id=step_id,
            target_id=target_id,
            element_type=element_type,
            element=element,
            operation=operation,
        )

    def load(self, filepath: str) -> Dict[int, Dict[int, List[EditAction]]]:
        """Load the file and return mapping: target_id -> {source_id -> [EditAction,...]}"""
        mapping: Dict[int, Dict[int, List[EditAction]]] = {}
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"edit-paths file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                # skip comment lines
                if line.startswith('#'):
                    continue
                try:
                    action = self.parse_line(line)
                except ValueError:
                    # strict: propagate parsing errors so caller is aware
                    raise

                target_map = mapping.setdefault(action.target_id, {})
                src_list = target_map.setdefault(action.source_id, [])
                src_list.append(action)

        # Sort lists by step_id to ensure correct order
        for target_id, src_map in mapping.items():
            for src_id, edits in src_map.items():
                edits.sort(key=lambda e: e.step_id)

        return mapping

    def iter_pair_paths(self, mapping: Dict[int, Dict[int, List[EditAction]]]) -> Iterable[Tuple[int, int, List[EditAction]]]:
        """Yield tuples (target_id, source_id, edits_list) from mapping."""
        for target_id, src_map in mapping.items():
            for src_id, edits in src_map.items():
                yield target_id, src_id, edits


if __name__ == '__main__':
    # Simple smoke test when executed directly. It tries to find the known MUTAG file
    # relative to this module and prints basic statistics.
    here = os.path.dirname(__file__)
    default_path = os.path.normpath(os.path.join(here, '..', '..', 'Data', 'GEDGraphs', 'MUTAG_i-E_d_IsoN', 'MUTAG_edit_paths_data.txt'))

    filepath = default_path
    print(f"Loading edit-paths from: {filepath}")
    loader = PathInfoLoader()
    mapping = loader.load(filepath)

    num_targets = len(mapping)
    num_sources = sum(len(smap) for smap in mapping.values())
    num_steps = sum(len(ed) for smap in mapping.values() for ed in smap.values())

    print(f"Parsed target graphs: {num_targets}")
    print(f"Parsed source graphs (total across all targets): {num_sources}")
    print(f"Parsed edit steps (total): {num_steps}")
    # print small sample
    sample_target = next(iter(mapping))
    sample_srcs = mapping[sample_target]
    print(f"Sample target id: {sample_target}, number of source graphs: {len(sample_srcs)}")
    # show first source's first 10 edits
    first_src_id = next(iter(sample_srcs))
    first_edits = sample_srcs[first_src_id][:10]
    for e in first_edits:
        print(e)
