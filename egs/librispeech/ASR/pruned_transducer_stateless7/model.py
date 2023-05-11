# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import k2
import torch
import torch.nn as nn
import random
import warnings
from encoder_interface import EncoderInterface

from icefall.utils import add_sos, make_pad_mask
from scaling import penalize_abs_values_gt, ScaledLinear
from torch import Tensor

class PromptedTransducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        text_embed: nn.Module,
        text_encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.text_embed = text_embed
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.joiner = joiner

        self.simple_am_proj = ScaledLinear(
            encoder_dim,
            vocab_size,
            initial_scale=0.25,
        )
        self.simple_lm_proj = ScaledLinear(
            decoder_dim,
            vocab_size,
            initial_scale=0.25,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        style_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          text:
            A 2-D tensor of integer dtype containing prompt text, of shape (N, T).
            It is exptected to contain the style prompt (first) and then the content
            prompt.
          text_lens:
            A 1-D tensor of shape (N,). It contains the number of elements (bytes)
            in `text` before padding, which will include the lengths of the
            style plus the content prompt.
          style_lens:
            A 1-D tensor of shape (N,), containing the number of elements (bytes)
            within each row of `text` that correspond to the style prompt (these
            are expected to come first).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        x, x_lens = self.encoder_embed(x, x_lens)

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        text = text.t()  # now (T, N)
        text = self.text_embed(text) # now (T, N, C)
        text_key_padding_mask = make_pad_mask(text_lens)

        memory, text_lens = self.text_encoder(text, text_lens,
                                              text_key_padding_mask)

        memory = self._add_style_indicator(memory, style_lens)

        memory_key_padding_mask = make_pad_mask(text_lens)

        encoder_out, x_lens = self.encoder(x, x_lens, src_key_padding_mask,
                                           memory=memory,
                                           memory_key_padding_mask=memory_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return (simple_loss, pruned_loss)


    def _add_style_indicator(self, memory: Tensor, style_lens: Tensor):
        """
        Adds to `memory` an indicator that is 0.1 for positions that correspond to
        the `style prompt` and 0 elsewhere.  The scale can be fixed because the
        scale of the memory vector can adjust to compensate (within limits set
        by the balancers)..

        Args:
             memory: (memory_len, batch_size, embed_dim)
         style_lens: (batch_size,),  a vector of lengths of the style prompt.
        """

        (memory_len, batch_size, embed_dim) = memory.shape


        indicator = torch.arange(memory_len, device=memory.device).unsqueeze(-1) < style_lens
        indicator = indicator.to(memory.dtype).unsqueeze(-1)

        return memory + indicator
