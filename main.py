import mlx.core as mx
import mlx.nn as nn
import numpy as np

class BasicAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_size):
        super(BasicAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.layer_1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, attention_size)
    
    def forward(self, encoder_states, decoder_state):
        # concatenate the encoder states and decoder state
        inputs = mx.concatenate([encoder_states, mx.broadcast_to(decoder_state, [input_length, hidden_size])],  axis=1)
        activations = mx.tanh(mx.matmul(inputs, self.layer_1))
        scores = mx.matmul(activations, self.layer_2)
        weights = mx.softmax(scores)
        weighted_scores = encoder_states * weights
        context = mx.sum(weighted_scores, axis=0)
        return context