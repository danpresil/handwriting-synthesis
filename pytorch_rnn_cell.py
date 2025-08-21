from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Bernoulli, Categorical

from pytorch_rnn_ops import rnn_free_run


@dataclass
class LSTMAttentionCellState:
    h1: torch.Tensor
    c1: torch.Tensor
    h2: torch.Tensor
    c2: torch.Tensor
    h3: torch.Tensor
    c3: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    kappa: torch.Tensor
    w: torch.Tensor
    phi: torch.Tensor


class LSTMAttentionCell(nn.Module):
    """PyTorch implementation of the LSTM with attention used for handwriting.

    This module mirrors the TensorFlow version in ``rnn_cell.py`` and exposes a
    ``forward`` method that operates on :class:`LSTMAttentionCellState`.
    """

    def __init__(
        self,
        lstm_size: int,
        num_attn_mixture_components: int,
        attention_values: torch.Tensor,
        attention_values_lengths: torch.Tensor,
        num_output_mixture_components: int,
        bias: torch.Tensor | None = None,
        input_size: int = 3,
    ) -> None:
        super().__init__()
        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.attention_values = attention_values
        self.attention_values_lengths = attention_values_lengths
        self.window_size = attention_values.shape[2]
        self.char_len = attention_values.shape[1]
        self.num_output_mixture_components = num_output_mixture_components
        self.output_units = 6 * num_output_mixture_components + 1
        self.input_size = input_size

        if bias is None:
            self.bias = torch.zeros(attention_values.shape[0], device=attention_values.device)
        else:
            self.bias = bias

        # LSTM cells
        self.lstm1 = nn.LSTMCell(self.window_size + self.input_size, lstm_size)
        self.lstm2 = nn.LSTMCell(self.input_size + lstm_size + self.window_size, lstm_size)
        self.lstm3 = nn.LSTMCell(self.input_size + lstm_size + self.window_size, lstm_size)

        # Linear layers
        self.attn_linear = nn.Linear(self.window_size + self.input_size + lstm_size,
                                     3 * num_attn_mixture_components)
        self.output_linear = nn.Linear(lstm_size, self.output_units)

    def zero_state(self, batch_size: int, device: torch.device | None = None) -> LSTMAttentionCellState:
        device = device if device is not None else self.attention_values.device
        zeros_lstm = torch.zeros(batch_size, self.lstm_size, device=device)
        zeros_attn = torch.zeros(batch_size, self.num_attn_mixture_components, device=device)
        zeros_w = torch.zeros(batch_size, self.window_size, device=device)
        zeros_phi = torch.zeros(batch_size, self.char_len, device=device)
        return LSTMAttentionCellState(
            h1=zeros_lstm.clone(),
            c1=zeros_lstm.clone(),
            h2=zeros_lstm.clone(),
            c2=zeros_lstm.clone(),
            h3=zeros_lstm.clone(),
            c3=zeros_lstm.clone(),
            alpha=zeros_attn.clone(),
            beta=zeros_attn.clone(),
            kappa=zeros_attn.clone(),
            w=zeros_w.clone(),
            phi=zeros_phi.clone(),
        )

    def forward(self, inputs: torch.Tensor, state: LSTMAttentionCellState) -> Tuple[torch.Tensor, LSTMAttentionCellState]:
        # LSTM 1
        s1_in = torch.cat([state.w, inputs], dim=1)
        h1, c1 = self.lstm1(s1_in, (state.h1, state.c1))

        # Attention mechanism
        attn_inputs = torch.cat([state.w, inputs, h1], dim=1)
        attn_params = F.softplus(self.attn_linear(attn_inputs))
        alpha, beta, kappa = torch.chunk(attn_params, 3, dim=1)
        kappa = state.kappa + kappa / 25.0
        beta = torch.clamp(beta, min=0.01)

        kappa_exp = kappa.unsqueeze(2)
        alpha_exp = alpha.unsqueeze(2)
        beta_exp = beta.unsqueeze(2)

        batch_size = inputs.shape[0]
        u = torch.arange(self.char_len, device=inputs.device).view(1, 1, -1).float()
        u = u.expand(batch_size, self.num_attn_mixture_components, -1)

        phi = torch.sum(alpha_exp * torch.exp(-((kappa_exp - u) ** 2) / beta_exp), dim=1)
        phi_exp = phi.unsqueeze(2)

        seq_mask = (
            torch.arange(self.char_len, device=inputs.device).unsqueeze(0)
            < self.attention_values_lengths.unsqueeze(1)
        ).float()
        seq_mask_exp = seq_mask.unsqueeze(2)

        w = torch.sum(phi_exp * self.attention_values * seq_mask_exp, dim=1)

        # LSTM 2
        s2_in = torch.cat([inputs, h1, w], dim=1)
        h2, c2 = self.lstm2(s2_in, (state.h2, state.c2))

        # LSTM 3
        s3_in = torch.cat([inputs, h2, w], dim=1)
        h3, c3 = self.lstm3(s3_in, (state.h3, state.c3))

        new_state = LSTMAttentionCellState(
            h1=h1,
            c1=c1,
            h2=h2,
            c2=c2,
            h3=h3,
            c3=c3,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            w=w,
            phi=phi,
        )
        return h3, new_state

    def output_function(self, state: LSTMAttentionCellState) -> torch.Tensor:
        params = self.output_linear(state.h3)
        pis, mus, sigmas, rhos, es = self._parse_parameters(params)
        mu1, mu2 = torch.chunk(mus, 2, dim=1)
        mus = torch.stack([mu1, mu2], dim=2)
        sigma1, sigma2 = torch.chunk(sigmas, 2, dim=1)

        cov = torch.stack([
            sigma1 ** 2,
            rhos * sigma1 * sigma2,
            rhos * sigma1 * sigma2,
            sigma2 ** 2,
        ], dim=2)
        cov = cov.view(-1, self.num_output_mixture_components, 2, 2)

        mvn = MultivariateNormal(loc=mus, covariance_matrix=cov)
        bern = Bernoulli(probs=es)
        cat = Categorical(probs=pis)

        sampled_e = bern.sample()
        sampled_coords = mvn.sample()
        sampled_idx = cat.sample()
        idx = torch.arange(state.h3.size(0), device=state.h3.device)
        coords = sampled_coords[idx, sampled_idx]
        return torch.cat([coords, sampled_e], dim=1)

    def termination_condition(self, state: LSTMAttentionCellState) -> torch.Tensor:
        char_idx = torch.argmax(state.phi, dim=1)
        final_char = char_idx >= (self.attention_values_lengths - 1)
        past_final_char = char_idx >= self.attention_values_lengths
        output = self.output_function(state)
        es = output[:, 2].long()
        is_eos = es == 1
        return torch.logical_or(final_char & is_eos, past_final_char)

    def free_run(
        self,
        initial_state: LSTMAttentionCellState,
        initial_input: torch.Tensor,
        max_steps: int,
    ) -> Tuple[torch.Tensor, LSTMAttentionCellState]:
        """Generate a sequence by feeding predictions back into the cell.

        Args:
            initial_state: Starting state for the RNN cell.
            initial_input: First input fed to the cell of shape ``[B, F]``.
            max_steps: Maximum number of steps to unroll.

        Returns:
            A tuple ``(outputs, final_state)`` where ``outputs`` has shape
            ``[T, B, F]``.
        """

        return rnn_free_run(
            cell=self,
            initial_state=initial_state,
            initial_input=initial_input,
            max_steps=max_steps,
        )

    def _parse_parameters(self, gmm_params: torch.Tensor, eps: float = 1e-8, sigma_eps: float = 1e-4):
        splits = [
            self.num_output_mixture_components,
            2 * self.num_output_mixture_components,
            self.num_output_mixture_components,
            2 * self.num_output_mixture_components,
            1,
        ]
        pis, sigmas, rhos, mus, es = torch.split(gmm_params, splits, dim=-1)
        bias = self.bias.unsqueeze(1)
        pis = pis * (1 + bias)
        sigmas = sigmas - bias

        pis = torch.softmax(pis, dim=-1)
        pis = torch.where(pis < 0.01, torch.zeros_like(pis), pis)
        sigmas = torch.clamp(torch.exp(sigmas), min=sigma_eps)
        rhos = torch.clamp(torch.tanh(rhos), min=eps - 1.0, max=1.0 - eps)
        es = torch.clamp(torch.sigmoid(es), min=eps, max=1.0 - eps)
        es = torch.where(es < 0.01, torch.zeros_like(es), es)
        return pis, mus, sigmas, rhos, es
