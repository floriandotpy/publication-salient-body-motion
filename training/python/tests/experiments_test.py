from kinectgestures.experiments import ExperimentGenerator
from nose.tools import assert_list_equal

BASIC_KEYS = ['a', 'b', 'c']
BASIC_CONFIG = {'a': 1, 'b': 2}
BASIC_VARIATIONS = {'c': [3, 4]}
BASIC_DESIRED_RESULT = [{'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2, 'c': 4}]


def setup_basic_generator():
    base_config = BASIC_CONFIG
    variations = BASIC_VARIATIONS
    generator = ExperimentGenerator(base_config, variations)
    return generator


def test_empty_config():
    generator = ExperimentGenerator({}, {})
    configs = generator.generate()

    assert len(configs) == 0


def test_generates_all_keys():
    generator = setup_basic_generator()
    configs = generator.generate()

    assert all(k in configs[0].keys() for k in BASIC_KEYS)


def test_generates_correct_number_of_configs():
    generator = setup_basic_generator()
    configs = generator.generate()

    assert len(configs) == len(BASIC_DESIRED_RESULT)


def test_generates_correct_combination():
    generator = setup_basic_generator()
    configs = generator.generate()

    desired_configs = BASIC_DESIRED_RESULT

    assert_list_equal(configs, desired_configs)


def test_checkpoint_path_pattern():
    base_config = {'a': 1, "checkpoint_path": '/tmp/path'}
    variations = {'b': [2, 3]}
    path_pattern = "checkpoint-{a}-{b}"
    generator = ExperimentGenerator(base_config, variations, checkpoint_path_pattern=path_pattern)
    generated = generator.generate()

    desired_output = [
        {'a': 1, 'b': 2, "checkpoint_path": "/tmp/path/checkpoint-1-2"},
        {'a': 1, 'b': 3, "checkpoint_path": '/tmp/path/checkpoint-1-3'}
    ]

    assert_list_equal(desired_output, generated)


def test_only_combine_list_values():
    base_config = {'a': 1}
    variations = {
        'b': 2,  # this is a primitive value: should not produce additional combinations
        'c': [3, 4],  # 2 variations from times
        'd': [5, 6]  # times 2 variations from this = 4 variations
    }
    desired_output = [
        {'a': 1, 'b': 2, 'c': 3, 'd': 5},
        {'a': 1, 'b': 2, 'c': 3, 'd': 6},
        {'a': 1, 'b': 2, 'c': 4, 'd': 5},
        {'a': 1, 'b': 2, 'c': 4, 'd': 6}
    ]

    generator = ExperimentGenerator(base_config, variations)
    generated = generator.generate()

    assert_list_equal(desired_output, generated)
