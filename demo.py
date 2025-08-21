import os
import logging

import numpy as np
import svgwrite
import torch

import drawing
import lyrics
from model import HandwritingModel


class Hand(object):

    def __init__(self):
        """Initialise the handwriting model using the PyTorch implementation."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HandwritingModel(
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10,
        )
        checkpoint_path = os.path.join('checkpoints', 'model.pt')
        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and 'model_state' in state:
                state = state['model_state']
            self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        valid_char_set = set(drawing.alphabet)
        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    (
                        "Each line must be at most 75 characters. "
                        "Line {} contains {}"
                    ).format(line_num, len(line))
                )

            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        (
                            "Invalid character {} detected in line {}. "
                            "Valid character set is {}"
                        ).format(char, line_num, valid_char_set)
                    )

        strokes = self._sample(lines, biases=biases, styles=styles)
        self._draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40 * max(len(i) for i in lines)
        biases = biases if biases is not None else [0.5] * num_samples

        samples = []
        for idx, line in enumerate(lines):
            bias = torch.tensor([biases[idx]], dtype=torch.float32, device=self.device)
            if styles is not None:
                style = styles[idx]
                x_p = np.load(f'styles/style-{style}-strokes.npy')
                c_p = np.load(f'styles/style-{style}-chars.npy').tostring().decode('utf-8')
                c_seq = drawing.encode_ascii(str(c_p) + ' ' + line)
                x_prime = torch.from_numpy(x_p).float().unsqueeze(0).to(self.device)
                x_prime_len = torch.tensor([len(x_p)], dtype=torch.long, device=self.device)
                c = torch.tensor(c_seq, dtype=torch.long, device=self.device).unsqueeze(0)
                c_len = torch.tensor([len(c_seq)], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    sample = self.model.primed_sample(
                        x_prime,
                        x_prime_len,
                        c,
                        c_len,
                        bias,
                        max_tsteps,
                    )
            else:
                c_seq = drawing.encode_ascii(line)
                c = torch.tensor(c_seq, dtype=torch.long, device=self.device).unsqueeze(0)
                c_len = torch.tensor([len(c_seq)], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    sample = self.model.sample(
                        c,
                        c_len,
                        bias,
                        max_tsteps,
                    )
            sample = sample[:, 0, :].detach().cpu().numpy()
            samples.append(sample)
        return samples

    def _draw(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
        stroke_colors = stroke_colors or ['black']*len(lines)
        stroke_widths = stroke_widths or [2]*len(lines)

        line_height = 60
        view_width = 1000
        view_height = line_height*(len(strokes) + 1)

        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(width=view_width, height=view_height)
        dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

        initial_coord = np.array([0, -(3*line_height / 4)])
        for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):

            if not line:
                initial_coord[1] -= line_height
                continue

            offsets[:, :2] *= 1.5
            strokes = drawing.offsets_to_coords(offsets)
            strokes = drawing.denoise(strokes)
            strokes[:, :2] = drawing.align(strokes[:, :2])

            strokes[:, 1] *= -1
            strokes[:, :2] -= strokes[:, :2].min() + initial_coord
            strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*strokes.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            dwg.add(path)

            initial_coord[1] -= line_height

        dwg.save()


if __name__ == '__main__':
    hand = Hand()

    # usage demo
    lines = [
        "Now this is a story all about how",
        "My life got flipped turned upside down",
        "And I'd like to take a minute, just sit right there",
        "I'll tell you how I became the prince of a town called Bel-Air",
    ]
    biases = [.75 for i in lines]
    styles = [9 for i in lines]
    stroke_colors = ['red', 'green', 'black', 'blue']
    stroke_widths = [1, 2, 1, 2]

    hand.write(
        filename='img/usage_demo.svg',
        lines=lines,
        biases=biases,
        styles=styles,
        stroke_colors=stroke_colors,
        stroke_widths=stroke_widths
    )

    # demo number 1 - fixed bias, fixed style
    lines = lyrics.all_star.split("\n")
    biases = [.75 for i in lines]
    styles = [12 for i in lines]

    hand.write(
        filename='img/all_star.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )

    # demo number 2 - fixed bias, varying style
    lines = lyrics.downtown.split("\n")
    biases = [.75 for i in lines]
    styles = np.cumsum(np.array([len(i) for i in lines]) == 0).astype(int)

    hand.write(
        filename='img/downtown.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )

    # demo number 3 - varying bias, fixed style
    lines = lyrics.give_up.split("\n")
    biases = .2*np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    styles = [7 for i in lines]

    hand.write(
        filename='img/give_up.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
