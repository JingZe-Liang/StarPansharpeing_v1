from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
import torch


model = torch.nn.Conv2d(3, 3, 3, 1, 1)
model = CheckpointWrapper(model)
print(model.state_dict().keys())
