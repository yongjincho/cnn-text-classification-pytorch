_config_file_name = "config.yml"


def _load(paths):
    import yaml

    config = {}
    for f in map(open, paths):
        obj = yaml.load(f.read())
        for k, v in obj.items():
            if k in config:
                raise RuntimeError("The configuration key '{}' is duplicated.".format(k))
            config[k] = v

    g = globals()
    for k, v in config.items():
        g[k] = v


def _load_from_model_dir(model_dir):
    import os
    config_file = os.path.join(model_dir, _config_file_name)
    if os.path.exists(config_file):
        _load([os.path.join(model_dir, _config_file_name)])
        return True
    else:
        return False


def _save(model_dir):
    import os, yaml

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config = {}
    for k, v in globals().items():
        if not k.startswith("_"):
            config[k] = v

    config_yml = yaml.dump(config, default_flow_style=False)
    with open(os.path.join(model_dir, _config_file_name), "w+") as f:
        f.write(config_yml)
