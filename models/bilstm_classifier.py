import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
# maybe import matplotlib .pyplot as plt
# maybe import pickle later for pre-trained embeddings?

UNK = '<UNK>' 
START_TAG = '--START--'
END_TAG = '--END--'

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else to_ix[UNK] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def log_sum_exp(vec):
    # calculates log_sum_exp in a stable way
    max_score = vec[0][argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return (max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast))))

class BiLSTM(nn.Module):
    """
    BiLSTM classifier
    """
    def __init__(self, vocab_size, category_to_ix, embedding_dim, hidden_dim, embeddings=None):
        super(BiLSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.category_to_ix = category_to_ix
        self.ix_to_tag = {v:k for k,v in category_to_ix.items()}

        # embedding variable
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        if embeddings is not None:
            self.word_embeds.weight.data.copy_(torch.from_numpy(embeddings))

        # lstm layer mapping the embeddings of the word into the hidden state
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim//2, num_layers=1,bidirectional=True)

        # mapping the output of the LSTM into tag space, fully connected layer
        self.hidden2out = nn.Linear(in_features=hidden_dim, out_features=self.tagset_size, bias=True)

        self.hidden = (Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def forward(self, poem):
        """
        The function obtain the scores for poem.
        Input:
        sentence: a sequence of ids for each word in the sentence
        Make sure to reshape the embeddings of the words before sending them to the BiLSTM. 
        The axes semantics are: seq_len, mini_batch, embedding_dim
        Output: 
        returns lstm_feats: scores for each tag for each token in the sentence.
        """
        self.hidden = (Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                Variable(torch.randn(2, 1, self.hidden_dim // 2)))
        
        embed = self.word_embeds(poem)
        embed_format = embed.view(len(poem), -1, self.embedding_dim)

        output, (h_n, c_n) = self.lstm(embed_format, self.hidden)
        output = self.hidden2category(output.view(len(poem), self.hidden_dim))

        return output

    def predict(self, poem):
        """
        Input:
            text for a poem
        Outputs:
            predicts the class with the maximum probability
        """
        lstm_feats = self.forward(poem)
        softmax_layer = torch.nn.Softmax(dim=1)
        probs = softmax_layer(lstm_feats)
        idx = argmax(probs)
        return self.ix_to_category[idx[0]]
