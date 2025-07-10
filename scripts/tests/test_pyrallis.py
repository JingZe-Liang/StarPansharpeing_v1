from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyrallis


@dataclass
class Config:
    a: list = field(default_factory=lambda: [1, 2, 3])
    path: str = "/home"
    path2: Path = Path("home/user")
    b: int = 1
    lr: float = 1.02


@pyrallis.wrap()
def main(cfg: Config):
    print(cfg)
    print(cfg.a)
    print(cfg.path)
    cfg.a.append(4)
    print(cfg.a)
    print(f"lr {cfg.lr} typed as: {type(cfg.lr)}")
    print(f"path2: {cfg.path2} typed as: {type(cfg.path2)}")

    print(asdict(cfg))


if __name__ == "__main__":
    main()
