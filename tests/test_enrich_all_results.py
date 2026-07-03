import pytest

pytest.importorskip("polars")

from utils.enrich_all_results import _extract_data_and_slices


class _FakeDataClass:
    @staticmethod
    def from_dict(payload):
        return {"reconstructed": payload}


def test_extract_data_and_slices_accepts_two_tuple():
    data_obj = object()
    slices = {"edge_index": [0, 2]}

    parsed_data, parsed_slices = _extract_data_and_slices((data_obj, slices))

    assert parsed_data is data_obj
    assert parsed_slices is slices


def test_extract_data_and_slices_accepts_mapping_format():
    data_obj = {"x": [1, 2]}
    slices = {"edge_index": [0, 2]}

    parsed_data, parsed_slices = _extract_data_and_slices(
        {"data": data_obj, "slices": slices}
    )

    assert parsed_data is data_obj
    assert parsed_slices is slices


def test_extract_data_and_slices_accepts_three_tuple_with_data_class():
    data_dict = {"x": [1, 2], "edge_index": [[0], [1]]}
    slices = {"edge_index": [0, 1]}

    parsed_data, parsed_slices = _extract_data_and_slices(
        (data_dict, slices, _FakeDataClass)
    )

    assert parsed_data == {"reconstructed": data_dict}
    assert parsed_slices is slices


def test_extract_data_and_slices_rejects_unsupported_structure():
    with pytest.raises(ValueError, match="Unsupported data.pt format"):
        _extract_data_and_slices(("only_one_value",))


def test_extract_data_and_slices_accepts_four_tuple_variant():
    data_obj = {"edge_index": [[0], [1]]}
    slices = {"edge_index": [0, 1]}
    extra = {"num_node_labels": 2}

    parsed_data, parsed_slices = _extract_data_and_slices(
        (data_obj, slices, extra, _FakeDataClass)
    )

    assert parsed_data == {"reconstructed": data_obj}
    assert parsed_slices is slices
