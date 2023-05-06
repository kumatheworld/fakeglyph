from sys import argv

from hydra import main
from hydra.utils import call
from omegaconf import DictConfig


@main(config_path="configs", config_name=argv.pop(1), version_base=None)
def run(cfg: DictConfig) -> None:
    call(cfg)


if __name__ == "__main__":
    run()
