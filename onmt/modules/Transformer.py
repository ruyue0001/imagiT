"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import onmt
from onmt.Models import EncoderBase
from onmt.Models import DecoderState
from onmt.Utils import aeq

MAX_SIZE = 5000


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """
    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = onmt.modules.BottleLinear(size, hidden_size)
        self.w_2 = onmt.modules.BottleLinear(hidden_size, size)
        self.layer_norm = onmt.modules.BottleLayerNorm(size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class TransformerEncoderLayer(nn.Module):
    """
        A single layer of the transformer encoder.

        Args:
                size(int): the dimension of keys/values/queries in
                           MultiHeadedAttention, also the input size of
                           the first-layer of the PositionwiseFeedForward.
                droput(float): dropout probability(0-1.0).
                head_count(int): the number of head for MultiHeadedAttention.
                hidden_size(int): the second-layer of the PositionwiseFeedForward.
        """

    def __init__(self, size, dropout,
                 head_count=10, hidden_size=2048):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            head_count, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)
        self.layer_norm = onmt.modules.BottleLayerNorm(size)

    def forward(self, input, mask):
        input_norm = self.layer_norm(input)
        context, attn = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.feed_forward(context + input)
        return out, attn

class mmTransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
    """

    def __init__(self, size, dropout,
                 head_count=10, hidden_size=2048):
        super(mmTransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            head_count, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)
        self.layer_norm = onmt.modules.BottleLayerNorm(size)

    def forward(self, input, input_mm, mask):
        input_norm = self.layer_norm(input)
        emb_norm = self.layer_norm(input_mm)
        mid, attn = self.self_attn(input_norm, input_norm, emb_norm, mask=mask)
        out = self.feed_forward(mid + input_mm)
        return out, attn


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O



    Args:
       num_layers (int): number of encoder layers
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    """
    def __init__(self, num_layers, hidden_size,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [mmTransformerEncoderLayer(hidden_size, dropout)
             for i in range(num_layers)])
        self.layer_norm = onmt.modules.BottleLayerNorm(hidden_size)

    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, n_batch, emb_dim = emb.size()

        out = emb.transpose(0, 1).contiguous()
        words = input[:, :, 0].transpose(0, 1)
        # CHECKS
        out_batch, out_len, _ = out.size()
        w_batch, w_len = words.size()
        aeq(out_batch, w_batch)
        aeq(out_len, w_len)
        # END CHECKS

        # Make mask.
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out, attn = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return Variable(emb.data), out.transpose(0, 1).contiguous()

    # Mulitimodal Transformer
    def forward_mm(self, input, img_proj, lengths=None, hidden=None):
        """
                Args:
                    input (:obj:`LongTensor`):
                       padded sequences of sparse indices `[src_len x batch x nfeat]`
                    lengths (:obj:`LongTensor`): length of each sequence `[batch]`
                    hidden (class specific):
                       initial hidden state.

                Returns:k
                    (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                        * final encoder state, used to initialize decoder
                           `[layers x batch x hidden]`
                        * contexts for attention, `[src_len x batch x hidden]`
                """
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, n_batch, emb_dim = emb.size()

        emb = emb.transpose(0, 1).contiguous() # (batch, src_len, nfeat)
        input_mm = torch.cat([emb, img_proj], dim=1)
        out = input_mm
        # words = input[:, :, 0].transpose(0, 1) # (batch, src_len)
        words = emb[:, :, 0]
        # CHECKS
        out_batch, out_len, _ = out.size()
        w_batch, w_len = words.size()

        aeq(out_batch, w_batch)
        aeq(s_len, w_len)
        # END CHECKS

        # Make mask. the mask here is no use
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, out_len, s_len)

        # assert not words.data.eq(padding_idx).max(), "there are some mask items eqaul to 1"

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out, attn = self.transformer[i](emb, out, mask) # attn 1x49x10
        out = self.layer_norm(out)

        return Variable(input_mm.data), out.transpose(0, 1).contiguous(), attn

class mmTransformerEncoder(EncoderBase):
    def __init__(self, num_layers, hidden_size,
                 dropout, embeddings):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, dropout)
             for i in range(num_layers)])
        self.mmtransformer = mmTransformerEncoderLayer(hidden_size, dropout)
        self.layer_norm = onmt.modules.BottleLayerNorm(hidden_size)

    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, n_batch, emb_dim = emb.size()

        out = emb.transpose(0, 1).contiguous()
        words = input[:, :, 0].transpose(0, 1)
        # CHECKS
        out_batch, out_len, _ = out.size()
        w_batch, w_len = words.size()
        aeq(out_batch, w_batch)
        aeq(out_len, w_len)
        # END CHECKS

        # Make mask.
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out, attn = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return Variable(emb.data), out.transpose(0, 1).contiguous()

    # Mulitimodal Transformer
    def forward_mm(self, input, img_proj, lengths=None, hidden=None):
        """
                Args:
                    input (:obj:`LongTensor`):
                       padded sequences of sparse indices `[src_len x batch x nfeat]`
                    lengths (:obj:`LongTensor`): length of each sequence `[batch]`
                    hidden (class specific):
                       initial hidden state.

                Returns:k
                    (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                        * final encoder state, used to initialize decoder
                           `[layers x batch x hidden]`
                        * contexts for attention, `[src_len x batch x hidden]`
                """
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, n_batch, emb_dim = emb.size()

        out = emb.transpose(0, 1).contiguous() # (batch, src_len, nfeat)
        # input_mm = torch.cat([emb, img_proj], dim=1)
        # out = input_mm
        # words = input[:, :, 0].transpose(0, 1) # (batch, src_len)
        #words = emb[:, :, 0]
        words = input[:, :, 0].transpose(0, 1)
        # CHECKS
        out_batch, out_len, _ = out.size()
        w_batch, w_len = words.size()

        aeq(out_batch, w_batch)
        aeq(s_len, w_len)
        # END CHECKS

        # Make mask. the mask here is no use
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, out_len, s_len)

        # assert not words.data.eq(padding_idx).max(), "there are some mask items eqaul to 1"

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            # out, attn = self.transformer[i](emb, out, mask) # attn 1x49x10
            out, attn = self.transformer[i](out, mask)
        
        input_mm = torch.cat([out, img_proj], dim=1)
        out, attn = self.mmtransformer[i](emb, input_mm, mask)

        out = self.layer_norm(out)

        return Variable(input_mm.data), out.transpose(0, 1).contiguous(), attn

class TransformerDecoderLayer(nn.Module):
    """
    Args:
      size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      droput(float): dropout probability(0-1.0).
      head_count(int): the number of heads for MultiHeadedAttention.
      hidden_size(int): the second-layer of the PositionwiseFeedForward.
    """
    def __init__(self, size, dropout,
                 head_count=10, hidden_size=2048):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = onmt.modules.MultiHeadedAttention(
                head_count, size, dropout=dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(
                head_count, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)
        self.layer_norm_1 = onmt.modules.BottleLayerNorm(size)
        self.layer_norm_2 = onmt.modules.BottleLayerNorm(size)
        self.dropout = dropout
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, input, context, src_pad_mask, tgt_pad_mask):
        #print (input.device, context.device, src_pad_mask.device, tgt_pad_mask.device)
        # Args Checks
        input_batch, input_len, _ = input.size()
        contxt_batch, contxt_len, _ = context.size() # (batch, srclen+49, dim)
        aeq(input_batch, contxt_batch)

        src_batch, t_len, s_len = src_pad_mask.size() #  (batch, tgtlen, srclen)
        tgt_batch, t_len_, t_len__ = tgt_pad_mask.size() # (batch, tgtlen, tgtlen)
        aeq(input_batch, contxt_batch, src_batch, tgt_batch)
        aeq(t_len, t_len_, t_len__, input_len)
        # aeq(s_len, contxt_len)
        # END Args Checks

        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1),
                            :tgt_pad_mask.size(1)]
                            .expand_as(tgt_pad_mask), 0) # (batch, tgtlen, tgtlen)

        input_norm = self.layer_norm_1(input)
        query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                     mask=dec_mask)
        query_norm = self.layer_norm_2(query+input) # (batch, tgtlen, dim)
        mid, attn = self.context_attn(context, context, query_norm,
                                      mask=src_pad_mask)
        output = self.feed_forward(mid+query+input)

        # CHECKS
        output_batch, output_len, _ = output.size()
        aeq(input_len, output_len)
        aeq(contxt_batch, output_batch)

        n_batch_, t_len_, s_len_ = attn.size()
        aeq(input_batch, n_batch_)
        aeq(contxt_len, s_len_)
        aeq(input_len, t_len_)
        # END CHECKS

        return output, attn

    def _get_attn_subsequent_mask(self, size):
        ''' Get an attention mask to avoid using the subsequent info.'''
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

       attn_type (str): if using a seperate copy attention
    """
    def __init__(self, num_layers, hidden_size, attn_type,
                 copy_attn, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(hidden_size, dropout)
             for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type)
            self._copy = True
        self.layer_norm = onmt.modules.BottleLayerNorm(hidden_size)

    def forward(self, input, context, state, context_lengths=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        # CHECKS
        assert isinstance(state, TransformerDecoderState)
        input_len, input_batch, _ = input.size() # (len', batch, 1)
        contxt_len, contxt_batch, _ = context.size() # (len+49, batch, dim)
        aeq(input_batch, contxt_batch)

        if state.previous_input is not None:
            input = torch.cat([state.previous_input, input], 0)

        src = state.src # input_mm (batch, srclen+49, dim)
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = input[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        # aeq(input_batch, contxt_batch, src_batch, tgt_batch) train and translate state.src have different shape
        # aeq(contxt_len, src_len)
        # aeq(input_len, tgt_len)
        # END CHECKS

        # Initialize return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_context = context.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(src_batch, tgt_len, contxt_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)
        tgt_pad_mask = tgt_pad_mask.type(torch.uint8)
        for i in range(self.num_layers):
            output, attn \
                = self.transformer_layers[i](output, src_context,
                                             src_pad_mask, tgt_pad_mask)

        output = self.layer_norm(output)
        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()
        if state.previous_input is not None:
            outputs = outputs[state.previous_input.size(0):]
            attn = attn[:, state.previous_input.size(0):].squeeze()
            attn = torch.stack([attn])
        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        # Update the state.
        state.update_state(input)

        return outputs, state, attns

    def init_decoder_state(self, src, context, enc_hidden):
        return TransformerDecoderState(src)


class TransformerDecoderState(DecoderState):
    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return (self.previous_input, self.src)

    def update_state(self, input):
        """ Called for every decoder forward pass. """
        self.previous_input = input

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = Variable(self.src.data.repeat(1, beam_size, 1),
                            volatile=True)
    def repeat_beam_size_times_mm(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = Variable(self.src.data.repeat(beam_size, 1, 1),
                            volatile=True)
