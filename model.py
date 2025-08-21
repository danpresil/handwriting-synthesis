import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

import drawing
from pytorch_rnn_cell import LSTMAttentionCell
from pytorch_rnn_ops import rnn_free_run


class HandwritingModel(nn.Module):
    """Handwriting synthesis model implemented in PyTorch."""

    def __init__(
        self,
        lstm_size: int,
        output_mixture_components: int,
        attention_mixture_components: int,
    ) -> None:
        super().__init__()
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        self.attention_mixture_components = attention_mixture_components
        self.output_units = 6 * output_mixture_components + 1

    def parse_parameters(
        self,
        gmm_params: torch.Tensor,
        bias: torch.Tensor | None = None,
        eps: float = 1e-8,
        sigma_eps: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split and activate GMM parameters."""
        splits = [
            self.output_mixture_components,
            2 * self.output_mixture_components,
            self.output_mixture_components,
            2 * self.output_mixture_components,
            1,
        ]
        pis, sigmas, rhos, mus, es = torch.split(gmm_params, splits, dim=-1)

        if bias is not None:
            bias = bias.view(-1, 1, 1)
            pis = pis * (1 + bias)
            sigmas = sigmas - bias

        pis = torch.softmax(pis, dim=-1)
        pis = torch.where(pis < 0.01, torch.zeros_like(pis), pis)
        sigmas = torch.clamp(torch.exp(sigmas), min=sigma_eps)
        rhos = torch.clamp(torch.tanh(rhos), min=eps - 1.0, max=1.0 - eps)
        es = torch.clamp(torch.sigmoid(es), min=eps, max=1.0 - eps)
        es = torch.where(es < 0.01, torch.zeros_like(es), es)
        return pis, mus, sigmas, rhos, es

    def NLL(
        self,
        y: torch.Tensor,
        lengths: torch.Tensor,
        pis: torch.Tensor,
        mus: torch.Tensor,
        sigmas: torch.Tensor,
        rhos: torch.Tensor,
        es: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sequence and element negative log-likelihood."""
        sigma1, sigma2 = sigmas.chunk(2, dim=2)
        y1, y2, y3 = y.chunk(3, dim=2)
        mu1, mu2 = mus.chunk(2, dim=2)

        norm = 1.0 / (2 * math.pi * sigma1 * sigma2 * torch.sqrt(1 - rhos ** 2))
        z = (
            ((y1 - mu1) / sigma1) ** 2
            + ((y2 - mu2) / sigma2) ** 2
            - 2 * rhos * (y1 - mu1) * (y2 - mu2) / (sigma1 * sigma2)
        )
        exp_term = torch.exp(-z / (2 * (1 - rhos ** 2)))
        gaussian_likelihoods = exp_term * norm
        gmm_likelihood = torch.sum(pis * gaussian_likelihoods, dim=2)
        gmm_likelihood = torch.clamp(gmm_likelihood, min=eps)

        bernoulli_likelihood = torch.where(y3 == 1.0, es, 1.0 - es).squeeze(-1)
        nll = -(torch.log(gmm_likelihood) + torch.log(bernoulli_likelihood))

        max_time = y.shape[1]
        mask = torch.arange(max_time, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask & ~torch.isnan(nll)
        nll = torch.where(mask, nll, torch.zeros_like(nll))
        num_valid = mask.float().sum(dim=1)

        sequence_loss = nll.sum(dim=1) / torch.clamp(num_valid, min=1.0)
        element_loss = nll.sum() / torch.clamp(num_valid.sum(), min=1.0)
        return sequence_loss, element_loss

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_len: torch.Tensor,
        c: torch.Tensor,
        c_len: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device
        attn_vals = F.one_hot(c, len(drawing.alphabet)).float().to(device)
        cell = LSTMAttentionCell(
            lstm_size=self.lstm_size,
            num_attn_mixture_components=self.attention_mixture_components,
            attention_values=attn_vals,
            attention_values_lengths=c_len.to(device),
            num_output_mixture_components=self.output_mixture_components,
            bias=bias.to(device) if bias is not None else None,
        )
        state = cell.zero_state(batch_size, device=device)

        params = []
        max_time = x.size(1)
        for t in range(max_time):
            input_t = x[:, t, :]
            output_t, state = cell.forward(input_t, state)
            params_t = cell.output_linear(output_t)
            params.append(params_t)
        params = torch.stack(params, dim=1)

        pis, mus, sigmas, rhos, es = self.parse_parameters(params, bias)
        seq_loss, elem_loss = self.NLL(y, x_len, pis, mus, sigmas, rhos, es)
        return seq_loss, elem_loss

    def sample(
        self,
        c: torch.Tensor,
        c_len: torch.Tensor,
        bias: torch.Tensor | None,
        max_steps: int,
    ) -> torch.Tensor:
        batch_size = c.size(0)
        device = c.device
        attn_vals = F.one_hot(c, len(drawing.alphabet)).float().to(device)
        cell = LSTMAttentionCell(
            lstm_size=self.lstm_size,
            num_attn_mixture_components=self.attention_mixture_components,
            attention_values=attn_vals,
            attention_values_lengths=c_len.to(device),
            num_output_mixture_components=self.output_mixture_components,
            bias=bias.to(device) if bias is not None else None,
        )
        initial_state = cell.zero_state(batch_size, device=device)
        initial_input = torch.cat([
            torch.zeros(batch_size, 2, device=device),
            torch.ones(batch_size, 1, device=device),
        ], dim=1)
        outputs, _ = rnn_free_run(cell, initial_state, initial_input, max_steps)
        return outputs

    def primed_sample(
        self,
        x_prime: torch.Tensor,
        x_prime_len: torch.Tensor,
        c: torch.Tensor,
        c_len: torch.Tensor,
        bias: torch.Tensor | None,
        max_steps: int,
    ) -> torch.Tensor:
        batch_size = x_prime.size(0)
        device = x_prime.device
        attn_vals = F.one_hot(c, len(drawing.alphabet)).float().to(device)
        cell = LSTMAttentionCell(
            lstm_size=self.lstm_size,
            num_attn_mixture_components=self.attention_mixture_components,
            attention_values=attn_vals,
            attention_values_lengths=c_len.to(device),
            num_output_mixture_components=self.output_mixture_components,
            bias=bias.to(device) if bias is not None else None,
        )
        state = cell.zero_state(batch_size, device=device)
        max_prime = x_prime.size(1)
        for t in range(max_prime):
            input_t = x_prime[:, t, :]
            mask = (t < x_prime_len).float().unsqueeze(1).to(device)
            input_t = input_t * mask
            _, state = cell.forward(input_t, state)
        idx = torch.arange(batch_size, device=device)
        last_input = x_prime[idx, x_prime_len - 1]
        outputs, _ = rnn_free_run(cell, state, last_input, max_steps)
        return outputs
