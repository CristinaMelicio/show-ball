"""
Helper functions
"""

import torch


def choose_device(requested_device: str) -> str:
    """
    Chooses the appropriate device based on availability and user preference.

    Args:
        requested_device: The device requested by the user ("auto", "mps", "cuda", or "cpu").

    Returns:
        The device to be used.
    """

    if requested_device != "auto":
        return requested_device

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"