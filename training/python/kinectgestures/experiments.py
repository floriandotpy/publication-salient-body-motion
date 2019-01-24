from itertools import product
import os


def is_iterable(el):
    return type(el) == list or type(el) == tuple


class ExperimentGenerator(object):

    def __init__(self, base_config, variations, checkpoint_path_pattern=None):
        self.base_config = base_config
        self.checkpoint_path_pattern = checkpoint_path_pattern

        # split variations in two dics, s.t. that one only has iterable elements
        self.variations = {k: v for k, v in variations.items() if is_iterable(v)}
        self.additional_base_config = {k: v for k, v in variations.items() if not is_iterable(v)}

    def generate(self):

        all_combinations = dict_product(self.variations)

        configs = []
        for single_combination_dict in all_combinations:
            config = {
                **self.base_config,
                **self.additional_base_config,
                **single_combination_dict,
            }
            if self.checkpoint_path_pattern:
                config["checkpoint_path"] = os.path.join(config["checkpoint_path"],
                                                         self.checkpoint_path_pattern.format(**config))
            configs.append(config)

        # never return the empty configuration
        configs = [c for c in configs if c != {}]

        return configs


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in product(*dicts.values()))
