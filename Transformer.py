"""
Transformer model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np

import torch
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.embeddingL = nn.Embedding(self.input_size, self.word_embedding_dim).to(self.device)      #initialize word embedding layer
        self.posembeddingL = nn.Embedding(self.max_length, self.word_embedding_dim).to(self.device)   #initialize positional embedding layer

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k).to(self.device)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v).to(self.device)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q).to(self.device)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k).to(self.device)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v).to(self.device)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q).to(self.device)
        
        self.softmax = nn.Softmax(dim=2).to(self.device)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim).to(self.device)
        self.norm_mh = nn.LayerNorm(self.hidden_dim).to(self.device)

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################
        self.feed_forward = nn.Sequential(nn.Linear(self.hidden_dim, self.dim_feedforward), nn.ReLU(), nn.Linear(self.dim_feedforward, self.hidden_dim)).to(self.device)
        self.norm_fc = nn.LayerNorm(self.hidden_dim).to(self.device)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.final_linear = nn.Linear(self.hidden_dim, self.output_size).to(self.device)
        self.final_softmax = nn.Softmax(dim = 2).to(self.device)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling TransformerTranslator class methods here.      #
        #############################################################################
        embeddings = self.embed(inputs)
        attention = self.multi_head_attention(embeddings)
        ff_out = self.feedforward_layer(attention)
        outputs = self.final_layer(ff_out)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
        N, T = inputs.shape
        pos_encoding = torch.arange(0, T).expand(N, T).to(self.device)
        inputs = inputs.to(self.device)
        embeddings = (self.embeddingL(inputs) + self.posembeddingL(pos_encoding)).to(self.device)   #remove this line when you start implementing your code
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        inputs = inputs.to(self.device)
        keys_1 = self.k1(inputs).to(self.device)
        values_1 = self.v1(inputs).to(self.device)
        queries_1 = self.q1(inputs).to(self.device)
        keys_1T = keys_1.permute(0, 2, 1).to(self.device)
        attention_1 = torch.bmm(self.softmax(torch.bmm(queries_1, keys_1T)/(self.dim_k ** 0.5)), values_1).to(self.device)

        keys_2 = self.k2(inputs).to(self.device)
        values_2 = self.v2(inputs).to(self.device)
        queries_2 = self.q2(inputs).to(self.device)
        keys_2T = keys_2.permute(0, 2, 1).to(self.device)
        attention_2 = torch.bmm(self.softmax(torch.bmm(queries_2, keys_2T)/(self.dim_k ** 0.5)), values_2).to(self.device)

        attention_cat = torch.cat((attention_1, attention_2), dim = 2).to(self.device)
        attention_proj = self.attention_head_projection(attention_cat).to(self.device)
        outputs = self.norm_mh(inputs + attention_proj)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        inputs = inputs.to(self.device)
        sublayer_inputs = self.feed_forward(inputs).to(self.device)
        outputs = self.norm_fc(inputs + sublayer_inputs) 
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        inputs = inputs.to(self.device)
        outputs = self.final_linear(inputs)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

class FullTransformerTranslator(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        seed_torch(0)

        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the Transformer Layer          #
        # You should use nn.Transformer                                              #
        ##############################################################################
        self.Transformer = nn.Transformer(self.word_embedding_dim, self.num_heads, num_layers_enc, num_layers_dec,
                                          self.dim_feedforward, dropout, batch_first = True)
        ##############################################################################
        # TODO:
        # Deliverable 2: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Initialize embeddings in order shown below.                                #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.srcembeddingL = nn.Embedding(self.input_size, self.word_embedding_dim)       #embedding for src
        self.tgtembeddingL = nn.Embedding(self.input_size, self.word_embedding_dim)       #embedding for target
        self.srcposembeddingL = nn.Embedding(self.max_length, self.word_embedding_dim)    #embedding for src positional encoding
        self.tgtposembeddingL = nn.Embedding(self.max_length, self.word_embedding_dim)    #embedding for target positional encoding
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the final layer.               #
        ##############################################################################
        self.final_linear = nn.Linear(self.hidden_dim, self.output_size)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, src, tgt):
        """
         This function computes the full Transformer forward pass used during training.
         Put together all of the layers you've developed in the correct order.

         :param src: a PyTorch tensor of shape (N,T) these are tokenized input sentences
                tgt: a PyTorch tensor of shape (N,T) these are tokenized translations
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the full Transformer stack for the forward pass. #
        #############################################################################
        outputs=None
        # shift tgt to right, add one <sos> to the beginning and shift the other tokens to right
        tgt = self.add_start_token(tgt)

        # embed src and tgt for processing by transformer
        N, T = src.shape
        pos_encoding_src = torch.arange(0, T).expand(N, T).to(self.device)
        pos_encoding_tgt = torch.arange(0, T).expand(N, T).to(self.device)
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_embedding = self.srcembeddingL(src) + self.srcposembeddingL(pos_encoding_src)
        tgt_embedding = self.tgtembeddingL(tgt) + self.tgtposembeddingL(pos_encoding_tgt)

        # create target mask and target key padding mask for decoder
        tgt_mask = self.Transformer.generate_square_subsequent_mask(T)
        tgt_mask = tgt_mask != 0.0
        tgt_mask = tgt_mask.to(self.device)
        tgt_key_padding_mask = tgt == self.pad_idx

        # invoke transformer to generate output
        outputs = self.Transformer(src_embedding, tgt_embedding, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask)
        outputs.to(self.device)
        # pass through final layer to generate outputs
        outputs = self.final_linear(outputs)
        outputs.to(self.device)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def generate_translation(self, src):
        """
         This function generates the output of the transformer taking src as its input
         it is assumed that the model is trained. The output would be the translation
         of the input

         :param src: a PyTorch tensor of shape (N,T)

         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 5: You will be calling the transformer forward function to    #
        # generate the translation for the input.                                   #
        #############################################################################
        N, T = src.shape
        tgt = torch.full((N, T), self.pad_idx)
        tgt[:, 0] = torch.clone(src[:, 0])

        outputs = torch.zeros(N, T, self.output_size)

        temp_out = self.forward(src, tgt)
        outputs[:, 0, :] = temp_out[:, 0, :]
        out_max = torch.argmax(outputs, dim = 2)
        
        #tgt[:, 1] = out_max[:, 0]
        #temp_out = self.forward(src, tgt)
        #outputs[:, 1, :] = temp_out[:, 1, :]
        #out_max = torch.argmax(outputs, dim = 2)

        #tgt[:, 2] = out_max[:, 1]
        #temp_out = self.forward(src, tgt)
        #outputs[:, 2, :] = temp_out[:, 2, :]

        for t in range(1, T):
            tgt[:, t] = out_max[:, t-1]
            temp_out = self.forward(src, tgt)
            outputs[:, t, :] = temp_out[:, t, :]
            out_max = torch.argmax(outputs, dim = 2)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def add_start_token(self, batch_sequences, start_token=2):
        """
            add start_token to the beginning of batch_sequence and shift other tokens to the right
            if batch_sequences starts with two consequtive <sos> tokens, return the original batch_sequence

            example1:
            batch_sequence = [[<sos>, 5,6,7]]
            returns:
                [[<sos>,<sos>, 5,6]]

            example2:
            batch_sequence = [[<sos>, <sos>, 5,6,7]]
            returns:
                [[<sos>, <sos>, 5,6,7]]
        """
        def has_consecutive_start_tokens(tensor, start_token):
            """
                return True if the tensor has two consecutive start tokens
            """
            consecutive_start_tokens = torch.tensor([start_token, start_token], dtype=tensor.dtype,
                                                    device=tensor.device)

            # Check if the first two tokens in each sequence are equal to consecutive start tokens
            is_consecutive_start_tokens = torch.all(tensor[:, :2] == consecutive_start_tokens, dim=1)

            # Return True if all sequences have two consecutive start tokens at the beginning
            return torch.all(is_consecutive_start_tokens).item()

        if has_consecutive_start_tokens(batch_sequences, start_token):
            return batch_sequences

        # Clone the input tensor to avoid modifying the original data
        modified_sequences = batch_sequences.clone()

        # Create a tensor with the start token and reshape it to match the shape of the input tensor
        start_token_tensor = torch.tensor(start_token, dtype=modified_sequences.dtype, device=modified_sequences.device)
        start_token_tensor = start_token_tensor.view(1, -1)

        # Shift the words to the right
        modified_sequences[:, 1:] = batch_sequences[:, :-1]

        # Add the start token to the first word in each sequence
        modified_sequences[:, 0] = start_token_tensor

        return modified_sequences

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True