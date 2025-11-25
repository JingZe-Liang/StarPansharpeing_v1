import warnings
from functools import partial
from typing import Literal

from src.utilities.logging.print import log_print


# custom warning for dataset curriculum, only filter this one
class CurriculumWarning(UserWarning):
    """Warning for dataset curriculum exceeding total steps"""


warnings.filterwarnings("once", category=CurriculumWarning)


def linear_transition_curriculum(
    step: int,
    total_steps: int,
    start_prob: list[float],
    end_prob: list[float],
):
    """
    Linearly transitions curriculum weights from start to end probabilities.

    Args:
        step (int): Current training step
        total_steps (int): Total number of steps for the transition
        start_prob (list[float]): Initial probability distribution across datasets
        end_prob (list[float]): Final probability distribution at the end of training

    Returns:
        list[float]: Interpolated probability weights for current step

    Example:
        >>> linear_transition_curriculum(
        ...     500,
        ...     1000,
        ...     [0.8, 0.2],
        ...     [0.2, 0.8],
        ... )
        [0.5, 0.5]  # Midpoint between start and end distributions
    """
    if step > total_steps:
        warnings.warn(
            f"[Dataset curriculum] Current step exceeds total steps {total_steps}. Returning end probabilities.",
            category=CurriculumWarning,
            stacklevel=2,
        )
        return end_prob

    probs = []
    for start, end in zip(start_prob, end_prob):
        prob = start + (end - start) * (step / total_steps)
        assert prob >= 0, "Probability cannot be negative."
        probs.append(prob)

    return probs


def staged_curriculum(
    step: int,
    total_steps: int,
    stages: list[tuple[int, list[float]]],
):
    """Determines the curriculum stage based on the current training step.

    The function iterates through a list of stages, each defined by a step
    threshold and a corresponding list of probabilities. It returns the
    probabilities of the first stage whose step threshold is greater than
    the current training step. If the current step exceeds all defined
    stage thresholds, the probabilities of the last stage are returned.

    Args:
        step: The current training step.
        total_steps: The total number of training steps (currently unused).
        stages: A list of tuples, where each tuple contains an integer
                representing the step threshold for that stage and a list
                of floats representing the probabilities for that stage.
                Example: [(100, [0.1, 0.9]), (500, [0.5, 0.5])]
    """
    for stage_step, stage_prob in stages:
        if step < stage_step:
            return stage_prob

    # Check if step exceeds the last stage threshold
    if step >= stages[-1][0]:
        log_print(
            f"Training step {step} exceeds the last defined stage threshold "
            f"{stages[-1][0]}. Using probabilities from the last stage.",
            stack_level=2,
            level="warning",
            warn_once=True,
        )

    return stages[-1][1]  # Return last stage if no earlier stage matches


def get_curriculum_fn(
    c_type: str | Literal["linear", "staged"], total_steps: int, **kwargs
):  # -> partial[list[float] | list[Any]] | partial[list[float]]:
    """
    Returns the appropriate curriculum function based on the type.

    Args:
        c_type (str): Type of curriculum ('linear', 'staged')
        **kwargs: Additional parameters for the curriculum function

    Returns:
        Callable: Curriculum function
    """
    if c_type == "linear":
        log_print(
            f"[Dataset curriculum] Using linear transition curriculum with {kwargs}",
            level="info",
        )
        return partial(linear_transition_curriculum, total_steps=total_steps, **kwargs)
    elif c_type == "staged":
        log_print(f"[Dataset curriculum] Using staged curriculum with {kwargs}", level="info")
        return partial(staged_curriculum, total_steps=total_steps, **kwargs)
    else:
        raise ValueError(f"Unknown curriculum type: {c_type}")


if __name__ == "__main__":
    # Example usage
    curriculum_fn = get_curriculum_fn(
        "linear",
        total_steps=1000,
        start_prob=[0.8, 0.2],
        end_prob=[0.2, 0.8],
    )
    print(curriculum_fn(step=500))  # Should print [0.5, 0.5]

    staged_fn = get_curriculum_fn(
        "staged",
        total_steps=1000,
        stages=[(100, [0.1, 0.9]), (500, [0.5, 0.5]), (1000, [0.2, 0.8])],
    )
    print(staged_fn(step=300))  # Should print [0.5, 0.5]

    import time

    from src.utilities.train_utils.state import StepsCounter

    sc = StepsCounter(["train", "val"])

    def test_cur(curr_fn):
        # log_print(f"[Dataset curriculum] Using {curr_fn.__name__}")

        while True:
            sc.update("train")
            print(curr_fn(step=sc.get("train")))
            time.sleep(0.01)

    test_cur(curriculum_fn)
