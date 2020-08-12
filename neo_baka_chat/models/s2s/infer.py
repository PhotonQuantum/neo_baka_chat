import random
from typing import List, Tuple

import torch
from torch import nn

from neo_baka_chat.infer.base import AbstractInference
from neo_baka_chat.vocab import Vocab
from .hparams import HyperParams
from .net import EncoderRNN, LuongAttnDecoderRNN


class Inference(AbstractInference):
    def __init__(self, vocab: Vocab, hparams: HyperParams):
        super().__init__(vocab, hparams)
        self.vocab = vocab

        _hparams = hparams.dumps()
        _hparams["batch_size"] = 1
        self.hparams = HyperParams.loads(_hparams)
        self.vocab_vector_size = len(vocab.i2w_table)

        self.embedding = nn.Embedding(self.vocab_vector_size, self.hparams.hidden_size)
        self.encoder = EncoderRNN(self.hparams, self.embedding)
        self.decoder = LuongAttnDecoderRNN(self.hparams, self.vocab_vector_size, self.embedding)
        self.embedding.eval()
        self.encoder.eval()
        self.decoder.eval()

        self.n = 1
        self.b = 10
        self.b_topk = 1
        self.temp = 0.8

    def load_state_dict(self, weights):
        encoder_weights, embed_weights, decoder_weights = weights
        self.encoder.load_state_dict(encoder_weights)
        self.embedding.load_state_dict(embed_weights)
        self.decoder.load_state_dict(decoder_weights)

    def _predict(self, ix: int, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        input_tensor = torch.LongTensor([[ix]]).cpu()

        output, hidden = self.decoder(input_tensor, hidden, encoder_outputs)

        # noinspection PyTypeChecker
        return output, hidden.detach()

    def _next_nodes(self, node: dict, b: int, encoder_outputs: torch.Tensor) -> List[dict]:
        output, hidden = self._predict(node["seq"][-1], node["hidden"], encoder_outputs)
        # noinspection PyArgumentList
        topk = output.flatten().topk(b * 2)
        out_nodes = []
        for i in range(b * 2):
            value, idx = float(topk[0][i].log()), int(topk[1][i])
            out_nodes.append({
                "seq": node["seq"] + [idx],
                "hidden": hidden,
                "log_prob": node["log_prob"] + value,
                "eos_count": node["eos_count"] + (1 if idx == self.vocab.eos_idx else 0),
                "seq_len": node["seq_len"] + 1
            })
        return out_nodes

    def _inference_prob(self, sequence: List[int]) -> List[int]:
        """
        Input: A sequence of word vectors, split with eos.
        Output: Multiple sequences of word vectors. Each sequence represents a complete sentence.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.embedding.eval()
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(
                torch.tensor([sequence]).transpose(0, 1),
                torch.tensor([len(sequence)])
            )

            hidden = encoder_hidden[:self.decoder.n_layers]
            choice = self.vocab.sos_idx
            result = []
            counter = 0
            while counter < self.n:
                ix = torch.tensor([[choice]]).cpu()
                output, hidden = self.decoder(ix, hidden, encoder_outputs)
                hidden = hidden.detach()

                probability = output[0].div(self.temp)
                choice = int(torch.multinomial(probability, 1)[0])

                result.append(choice)
                if choice == self.vocab.eos_idx:
                    counter += 1

        return result

    def _inference_beam(self, sequence: List[int]) -> List[int]:
        """
        Input: A sequence of word vectors, split with eos.
        Output: Multiple sequences of word vectors. Each sequence represents a complete sentence.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.embedding.eval()
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(
                torch.tensor([sequence]).transpose(0, 1),
                torch.tensor([len(sequence)])
            )

            # noinspection PyArgumentList,Mypy
            incomplete_nodes = [
                {"seq": [self.vocab.sos_idx], "hidden": encoder_hidden[:self.decoder.n_layers],
                 "log_prob": 0, "eos_count": 0, "seq_len": 0}
            ]

            complete_nodes: List[dict] = []

            while not len(complete_nodes) >= self.b:
                tmp_nodes = []
                for node in incomplete_nodes[:self.b]:
                    tmp_nodes.extend(self._next_nodes(node, self.b, encoder_outputs))
                tmp_nodes = sorted(tmp_nodes, key=lambda x: x["log_prob"], reverse=True)[:self.b * 2]

                incomplete_nodes = []
                for node in tmp_nodes:
                    if node["eos_count"] >= self.n or node["seq_len"] > 20:
                        complete_nodes.append(node)
                    else:
                        incomplete_nodes.append(node)

        result = sorted(complete_nodes, key=lambda x: x["log_prob"] / (len(x["seq"]) - 1), reverse=True)[
                     random.randint(0, min(self.b_topk - 1, self.b - 1))]["seq"][1:]
        return result

    def infer(self, sequence: List[int]) -> List[int]:
        """
        Input: A sentence/sentences split with sos.
        Output: Sentences split with eos.
        """
        input_vector = sequence + [self.vocab.eos_idx]
        output_vector = self._inference_beam(input_vector)
        return output_vector[:-1]
