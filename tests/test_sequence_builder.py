import torch

from datasets.frame_builder import build_temporal_sequence


def test_build_temporal_sequence_returns_four_frame_stacks():
    # Create a simple synthetic sequence of 6 frames, each with a distinct constant value.
    frames = [torch.full((1, 4, 4), float(i), dtype=torch.float32) for i in range(6)]

    sequences = build_temporal_sequence(frames, sequence_length=4)

    assert len(sequences) == 3, "Expected 3 four-frame sequences from 6 input frames."

    for idx, sequence in enumerate(sequences):
        assert sequence.shape == (4, 1, 4, 4), "Each sequence should stack 4 frames into shape [4, 1, H, W]."

        expected_values = torch.tensor([idx, idx + 1, idx + 2, idx + 3], dtype=torch.float32)
        actual_values = sequence[:, 0, 0, 0]

        assert torch.allclose(actual_values, expected_values), (
            f"Sequence {idx} values are incorrect: got {actual_values.tolist()}, "
            f"expected {expected_values.tolist()}"
        )
