import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    """Implement a Bi-LSTM model with attention."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx, embeddings=None):
        super(BiLSTM, self).__init__()
        
        # use word embeddings?
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pad_idx = pad_idx
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))

        # bi-directional lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)

        self.attn = nn.Linear(hidden_dim, 1) # an attention layer
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_data, lengths):
        """
        input_data: (n, seq_len)
        """
        embedded = self.embedding(input_data)

        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, c = self.lstm(packed)
        fixed_output, c = pad_packed_sequence(packed_output, batch_first=True)

        attn_scores = self.attn(fixed_output).squeeze(-1)

        # Mask attention to ignore <PAD> positions
        attn_scores[input_data == self.pad_idx] = -1e10 # then softmax sends this to 0ish
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)

        attn_out = torch.sum(attn_weights * fixed_output, dim=1)
        out = self.classifier(attn_out)

        return out


# PRE-TRAINED WORD VECTOR CODE, Glove embeddings for Spanish
  
def load_vec_file(path, vocab_set, dim):
    embeddings = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != dim + 1:
                continue  # skip malformed lines
            word = parts[0]
            if vocab_set is None or word in vocab_set:
                vec = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vec
    return embeddings

def build_embed_matrix(word_to_ix, path, embedding_dim):
    vocab_size = len(word_to_ix)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    glove = load_vec_file(path, vocab_set=word_to_ix.keys(), dim=300)
    for word, idx in word_to_ix.items():
        vector = glove.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector
        else: # random init
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return embedding_matrix
