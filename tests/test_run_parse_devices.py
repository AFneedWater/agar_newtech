from agar.run import parse_train_devices


def test_run_parse_devices_default():
    assert parse_train_devices([]) == 1


def test_run_parse_devices_override():
    assert parse_train_devices(["experiment=exp", "train.devices=2"]) == 2

