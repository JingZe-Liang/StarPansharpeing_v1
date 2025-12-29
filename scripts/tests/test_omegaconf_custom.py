from omegaconf import OmegaConf


def test_interpolation():
    # Case 1: Dot notation with nested interpolation
    conf_str = """
    place: foo
    paths:
      foo:
        train: bar
    train:
      input_dir: ${paths.${place}.train}
    """
    cfg = OmegaConf.create(conf_str)
    print(f"Case 1 (Dot): {cfg.train.input_dir}")

    # Case 2: Bracket notation
    conf_str_2 = """
    place: foo
    paths:
      foo:
        train: bar
    train:
      input_dir: ${paths[${place}].train}
    """
    cfg_2 = OmegaConf.create(conf_str_2)
    print(f"Case 2 (Bracket): {cfg_2.train.input_dir}")

    # Case 3: Relative path with dot
    conf_str_3 = """
    place: foo
    paths:
      foo:
        train: bar
    train:
      input_dir: ${..paths.${..place}.train}
    """
    cfg_3 = OmegaConf.create(conf_str_3)
    print(f"Case 3 (Relative Dot): {cfg_3.train.input_dir}")


if __name__ == "__main__":
    try:
        test_interpolation()
    except Exception as e:
        print(e)
