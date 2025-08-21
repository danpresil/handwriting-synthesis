from __future__ import print_function
import os

import numpy as np
import tensorflow as tf

import torch
from torch.utils.data import DataLoader, random_split

import drawing
from dataset import HandwritingDataset, handwriting_collate_fn
from rnn_cell import LSTMAttentionCell
from rnn_ops import rnn_free_run
from tf_base_model import TFBaseModel
from torch_utils import time_distributed_dense_layer


class DataLoaders(object):

    def __init__(self, train_dataset, val_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def _generator(self, dataset, batch_size, shuffle=True, infinite=True):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=handwriting_collate_fn,
        )
        while True:
            for x, y, x_len, c, c_len in loader:
                yield {
                    'x': x.numpy(),
                    'y': y.numpy(),
                    'x_len': x_len.numpy(),
                    'c': c.numpy(),
                    'c_len': c_len.numpy(),
                }
            if not infinite:
                break

    def train_batch_generator(self, batch_size):
        return self._generator(self.train_dataset, batch_size, shuffle=True, infinite=True)

    def val_batch_generator(self, batch_size):
        return self._generator(self.val_dataset, batch_size, shuffle=True, infinite=True)

    def test_batch_generator(self, batch_size):
        return self._generator(self.test_dataset, batch_size, shuffle=False, infinite=False)


class rnn(TFBaseModel):

    def __init__(
        self,
        lstm_size,
        output_mixture_components,
        attention_mixture_components,
        **kwargs
    ):
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        self.output_units = self.output_mixture_components*6 + 1
        self.attention_mixture_components = attention_mixture_components
        super(rnn, self).__init__(**kwargs)

    def parse_parameters(self, z, eps=1e-8, sigma_eps=1e-4):
        pis, sigmas, rhos, mus, es = tf.split(
            z,
            [
                1*self.output_mixture_components,
                2*self.output_mixture_components,
                1*self.output_mixture_components,
                2*self.output_mixture_components,
                1
            ],
            axis=-1
        )
        pis = tf.nn.softmax(pis, axis=-1)
        sigmas = tf.clip_by_value(tf.exp(sigmas), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos), eps - 1.0, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es), eps, 1.0 - eps)
        return pis, mus, sigmas, rhos, es

    def NLL(self, y, lengths, pis, mus, sigmas, rho, es, eps=1e-8):
        sigma_1, sigma_2 = tf.split(sigmas, 2, axis=2)
        y_1, y_2, y_3 = tf.split(y, 3, axis=2)
        mu_1, mu_2 = tf.split(mus, 2, axis=2)

        norm = 1.0 / (2*np.pi*sigma_1*sigma_2 * tf.sqrt(1 - tf.square(rho)))
        Z = tf.square((y_1 - mu_1) / (sigma_1)) + \
            tf.square((y_2 - mu_2) / (sigma_2)) - \
            2*rho*(y_1 - mu_1)*(y_2 - mu_2) / (sigma_1*sigma_2)

        exp = -1.0*Z / (2*(1 - tf.square(rho)))
        gaussian_likelihoods = tf.exp(exp) * norm
        gmm_likelihood = tf.reduce_sum(pis * gaussian_likelihoods, 2)
        gmm_likelihood = tf.clip_by_value(gmm_likelihood, eps, np.inf)

        bernoulli_likelihood = tf.squeeze(tf.where(tf.equal(tf.ones_like(y_3), y_3), es, 1 - es))

        nll = -(tf.log(gmm_likelihood) + tf.log(bernoulli_likelihood))
        sequence_mask = tf.logical_and(
            tf.sequence_mask(lengths, maxlen=tf.shape(y)[1]),
            tf.logical_not(tf.is_nan(nll)),
        )
        nll = tf.where(sequence_mask, nll, tf.zeros_like(nll))
        num_valid = tf.reduce_sum(tf.cast(sequence_mask, tf.float32), axis=1)

        sequence_loss = tf.reduce_sum(nll, axis=1) / tf.maximum(num_valid, 1.0)
        element_loss = tf.reduce_sum(nll) / tf.maximum(tf.reduce_sum(num_valid), 1.0)
        return sequence_loss, element_loss

    def sample(self, cell):
        initial_state = cell.zero_state(self.num_samples, dtype=tf.float32)
        initial_input = tf.concat([
            tf.zeros([self.num_samples, 2]),
            tf.ones([self.num_samples, 1]),
        ], axis=1)
        return rnn_free_run(
            cell=cell,
            sequence_length=self.sample_tsteps,
            initial_state=initial_state,
            initial_input=initial_input,
            scope='rnn'
        )[1]

    def primed_sample(self, cell):
        initial_state = cell.zero_state(self.num_samples, dtype=tf.float32)
        primed_state = tf.nn.dynamic_rnn(
            inputs=self.x_prime,
            cell=cell,
            sequence_length=self.x_prime_len,
            dtype=tf.float32,
            initial_state=initial_state,
            scope='rnn'
        )[1]
        return rnn_free_run(
            cell=cell,
            sequence_length=self.sample_tsteps,
            initial_state=primed_state,
            scope='rnn'
        )[1]

    def calculate_loss(self):
        self.x = tf.placeholder(tf.float32, [None, None, 3])
        self.y = tf.placeholder(tf.float32, [None, None, 3])
        self.x_len = tf.placeholder(tf.int32, [None])
        self.c = tf.placeholder(tf.int32, [None, None])
        self.c_len = tf.placeholder(tf.int32, [None])

        self.sample_tsteps = tf.placeholder(tf.int32, [])
        self.num_samples = tf.placeholder(tf.int32, [])
        self.prime = tf.placeholder(tf.bool, [])
        self.x_prime = tf.placeholder(tf.float32, [None, None, 3])
        self.x_prime_len = tf.placeholder(tf.int32, [None])
        self.bias = tf.placeholder_with_default(
            tf.zeros([self.num_samples], dtype=tf.float32), [None])

        cell = LSTMAttentionCell(
            lstm_size=self.lstm_size,
            num_attn_mixture_components=self.attention_mixture_components,
            attention_values=tf.one_hot(self.c, len(drawing.alphabet)),
            attention_values_lengths=self.c_len,
            num_output_mixture_components=self.output_mixture_components,
            bias=self.bias
        )
        self.initial_state = cell.zero_state(tf.shape(self.x)[0], dtype=tf.float32)
        outputs, self.final_state = tf.nn.dynamic_rnn(
            inputs=self.x,
            cell=cell,
            sequence_length=self.x_len,
            dtype=tf.float32,
            initial_state=self.initial_state,
            scope='rnn'
        )
        params = time_distributed_dense_layer(outputs, self.output_units)
        pis, mus, sigmas, rhos, es = self.parse_parameters(params)
        sequence_loss, self.loss = self.NLL(self.y, self.x_len, pis, mus, sigmas, rhos, es)

        self.sampled_sequence = tf.cond(
            self.prime,
            lambda: self.primed_sample(cell),
            lambda: self.sample(cell)
        )
        return self.loss


if __name__ == '__main__':
    dataset = HandwritingDataset(data_dir='data/processed/')
    total_len = len(dataset)
    train_size = int(0.95 * total_len)
    val_size = total_len - train_size
    gen = torch.Generator().manual_seed(2018)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)

    print('train size', len(train_dataset))
    print('val size', len(val_dataset))
    print('test size', len(dataset))

    loaders = DataLoaders(train_dataset, val_dataset, dataset)

    nn = rnn(
        reader=loaders,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        prediction_dir='predictions',
        learning_rates=[.0001, .00005, .00002],
        batch_sizes=[32, 64, 64],
        patiences=[1500, 1000, 500],
        beta1_decays=[.9, .9, .9],
        validation_batch_size=32,
        optimizer='rms',
        num_training_steps=100000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=2000,
        log_interval=20,
        grad_clip=10,
        lstm_size=400,
        output_mixture_components=20,
        attention_mixture_components=10
    )
    nn.fit()
