import sys
from pathlib import Path


def _append_dinov3_repo_to_path() -> None:
    repo_dir = Path(__file__).resolve().parents[3] / "stage1" / "utilities" / "losses" / "dinov3"
    if repo_dir.exists():
        sys.path.insert(0, str(repo_dir))


_append_dinov3_repo_to_path()  # load dinov3 self-holded adapter

from .dinov3_adapted import DinoUNet
from .tokenizer_backbone_adapted import TokenizerHybridUNet
from .tokenizer_backbone_adapted_multimodal import MultimodalTokenizerHybridUNet
