from copy import deepcopy

from easydict import EasyDict

from src.utilities.config_utils.to_container import to_easydict_recursive


def test_to_easydict_recursive_keeps_nested_non_string_key_subtree_plain_dict():
    config = {
        "latent_mask_config": {
            "mask_ratios": [0.0, 0.25, 0.5, 0.75],
            "block_sizes": {16: [1, 1], 32: [2, 2]},
            "mask_probs": {16: [0.7, 0.1, 0.1, 0.1], 32: [0.6, 0.1, 0.15, 0.15]},
        }
    }

    converted = to_easydict_recursive(config)

    assert isinstance(converted, EasyDict)
    assert isinstance(converted["latent_mask_config"], EasyDict)
    assert not isinstance(converted["latent_mask_config"]["block_sizes"], EasyDict)

    copied = deepcopy(converted)
    assert copied["latent_mask_config"]["block_sizes"][16] == [1, 1]


def test_to_easydict_recursive_preserves_easydict_for_all_string_key_tree():
    config = {"outer": {"inner": {"value": 1}}}

    converted = to_easydict_recursive(config)

    assert isinstance(converted, EasyDict)
    assert isinstance(converted.outer, EasyDict)
    assert isinstance(converted.outer.inner, EasyDict)
    assert converted.outer.inner.value == 1
