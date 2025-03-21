import torch
from torch import autocast


class DiagonalGaussianDistribution(object):
    @autocast("cuda", enabled=False)
    def __init__(self, parameters, deterministic=False):
        """Initializes a Gaussian distribution instance given the parameters.

        Args:
            parameters (torch.Tensor): The parameters for the Gaussian distribution. It is expected
                to be in shape [B, 2 * C, *], where B is batch size, and C is the embedding dimension.
                First C channels are used for mean and last C are used for logvar in the Gaussian distribution.
            deterministic (bool): Whether to use deterministic sampling. When it is true, the sampling results
                is purely based on mean (i.e., std = 0).
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    @autocast("cuda", enabled=False)
    def sample(self):
        x = self.mean.float() + self.std.float() * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    @autocast("cuda", enabled=False)
    def mode(self):
        return self.mean

    @autocast("cuda", enabled=False)
    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean.float(), 2)
                + self.var.float()
                - 1.0
                - self.logvar.float(),
                dim=[1, 2],
            )
