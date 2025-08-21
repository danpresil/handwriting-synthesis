import torch
from torch import nn


def dense_layer(inputs, out_features, activation=None, dropout=None):
    """Apply a dense layer to ``inputs`` using ``nn.Linear``.

    Args:
        inputs (Tensor): Input tensor of shape ``[batch, in_features]``.
        out_features (int): Number of output features.
        activation (callable, optional): Activation function to apply.
        dropout (float, optional): Dropout probability. If provided, ``nn.Dropout``
            with ``p=dropout`` is applied after the activation.

    Returns:
        Tensor: Output tensor of shape ``[batch, out_features]``.
    """
    in_features = inputs.shape[-1]
    layer = nn.Linear(in_features, out_features)
    x = layer(inputs)
    if activation is not None:
        x = activation(x)
    if dropout is not None:
        x = nn.Dropout(p=dropout)(x)
    return x


def time_distributed_dense_layer(inputs, out_features, activation=None, dropout=None):
    """Apply a dense layer to each time step of ``inputs``.

    Reshapes ``inputs`` from ``[B, T, F]`` to ``[B*T, F]``, applies an
    ``nn.Linear`` layer, then reshapes back to ``[B, T, out_features]``.

    Args:
        inputs (Tensor): Input tensor of shape ``[batch, time, features]``.
        out_features (int): Number of output features.
        activation (callable, optional): Activation function to apply.
        dropout (float, optional): Dropout probability applied after activation.

    Returns:
        Tensor: Output tensor of shape ``[batch, time, out_features]``.
    """
    batch, time, in_features = inputs.shape
    layer = nn.Linear(in_features, out_features)
    x = inputs.reshape(batch * time, in_features)
    x = layer(x)
    if activation is not None:
        x = activation(x)
    if dropout is not None:
        x = nn.Dropout(p=dropout)(x)
    x = x.reshape(batch, time, out_features)
    return x
