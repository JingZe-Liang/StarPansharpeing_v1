"""
Loss functions for hyperspectral anomaly detection
"""

from .adaptive_HAD_mse import AdaptiveHADMSE
from .msgms_loss import MSGMSLoss

__all__ = ["AdaptiveHADMSE", "MSGMSLoss"]
