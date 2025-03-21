import accelerate
import torch
from accelerate.utils import ProjectConfiguration

accelerator = accelerate.Accelerator(
    log_with="tensorboard",
    project_config=ProjectConfiguration(project_dir="test", logging_dir="test/tenb"),
)
accelerator.init_trackers("test")
track = accelerator.get_tracker("tensorboard")
track.log({"psnr": 20}, step=1)
