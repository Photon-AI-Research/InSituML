def load_run_settings(cfg_path):
    run_settings = {}

    # Last line in cfg.txt file should be empty.
    with open(cfg_path) as f:
        lines = f.readlines()

    for line in lines:
        key, value = line.split(': ', 1)
        run_settings[key] = value

    return run_settings
