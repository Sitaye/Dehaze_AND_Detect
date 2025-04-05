from fusion.pipline import Pipeline

def config_options():
    import argparse
    import tomllib

    parser = argparse.ArgumentParser(description="火场浓烟人体检测主程序")
    parser.add_argument("--config", type=str, default="./config.toml", help="配置文件路径")
    args = parser.parse_args()
    
    try:
        with open(args.config, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError as e:
        raise e
    except tomllib.TOMLDecodeError as e:
        raise e
    except Exception as e:
        raise e

    return config

if __name__ == "__main__":
    config = config_options()
    pipeline = Pipeline(config)
    modes = ['frame', 'video', 'stream']
    if config['general']['mode'] in modes:
        pipeline.run(config['general']['mode'])
    else:
        err = "Detect unknown mode. Please check whether 'mode' includes 'frame', 'video' or 'stream'."
        raise ValueError(err)
