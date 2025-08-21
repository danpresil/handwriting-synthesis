import torch
from typing import Any, Tuple, List


def rnn_free_run(
    cell: Any,
    initial_state: Any,
    initial_input: torch.Tensor,
    max_steps: int,
) -> Tuple[torch.Tensor, Any]:
    """Run ``cell`` in a free-running fashion.

    At each step the cell's :py:meth:`output_function` is fed back as the input
    for the next step. The loop terminates when ``max_steps`` is reached or when
    the cell's :py:meth:`termination_condition` signals that all sequences in
    the batch are finished.

    Args:
        cell: RNN cell with ``forward``, ``output_function`` and
            ``termination_condition`` methods.
        initial_state: Initial state for the cell.
        initial_input: Tensor of shape ``[B, F]`` used as the first input.
        max_steps: Maximum number of sampling steps to perform.

    Returns:
        Tuple containing ``outputs`` of shape ``[T, B, F]`` and the final cell
        state.
    """

    state = initial_state
    input_t = initial_input
    outputs: List[torch.Tensor] = []

    for _ in range(max_steps):
        # Perform one step of the RNN
        _, state = cell.forward(input_t, state)

        # Produce the next input using the cell's output function
        next_input = cell.output_function(state)
        outputs.append(next_input)

        # Check if all sequences have finished
        if torch.all(cell.termination_condition(state)):
            break

        input_t = next_input

    if outputs:
        stacked_outputs = torch.stack(outputs, dim=0)
    else:
        stacked_outputs = torch.zeros(
            (0, initial_input.size(0), initial_input.size(1)),
            device=initial_input.device,
        )

    return stacked_outputs, state

