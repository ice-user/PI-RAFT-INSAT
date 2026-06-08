import torch

def build_temporal_sequence(frames_list, sequence_length=2):
    """
    Builds temporal sequences from a list of loaded frames.

    Args:
        frames_list (list): List of loaded frame tensors.
        sequence_length (int): Target temporal frame length (e.g., 2 or 4).

    Returns:
        list[torch.Tensor]: A list of stacked tensors shaped [T, C, H, W].
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    sequences = []
    for i in range(len(frames_list) - sequence_length + 1):
        sequence = torch.stack(frames_list[i : i + sequence_length], dim=0)
        sequences.append(sequence)

    return sequences
