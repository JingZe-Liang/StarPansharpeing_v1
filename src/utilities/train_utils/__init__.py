import sys

from loguru import logger

from .lr_scheduler import (
    get_cosine_schedule_reduced_restart_with_warmup,
    get_cosine_schedule_reduced_with_warmup,
)
from .state import (
    LossMetricTracker,
    StepsCounter,
    dict_tensor_sync,
    metrics_sync,
    object_all_gather,
    object_scatter,
)
from .visualization import get_rgb_image

LT_ENABLED = False
try:
    import dowhen
    import lovely_tensors as lt

    LT_ENABLED = True
except ImportError:
    logger.debug("Lovely tensor or dowhen is not installed.")
    lt, do_when = None, None


def lt_hook():
    lt.monkey_patch()
    logger.warning(
        "Lovely-Tensors hook is enabled. "
        "Make sure only call this function in trackback!"
    )


if sys.version_info >= (3, 12) and LT_ENABLED:
    # only patch tensor when any error raised, it won't slow down the training process
    dowhen.do(lt_hook).when(logger.catch, "from_decorator = self._from_decorator")
    logger.info(
        "Dowhen hook is enabled. Make sure only call this function in trackback!"
    )
