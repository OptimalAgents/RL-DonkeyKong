import hydra
from hydra.utils import instantiate


@hydra.main(config_path="./config", config_name="config")
def main(cfg):
    print(cfg)
    print(instantiate(cfg.transforms))


if __name__ == "__main__":
    main()
