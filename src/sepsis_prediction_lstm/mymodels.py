import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MyLSTM(nn.Module):
    def __init__(self, dim_input, emb_size, hidden_size, num_layers):
        super(MyLSTM, self).__init__()
        self.embedding = nn.Linear(in_features=dim_input, out_features=emb_size)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(in_features=hidden_size, out_features=2)
    
    def forward(self, input_tuple):
        seqs, lengths = input_tuple

        embedded = (torch.tanh(self.embedding(seqs)))

        seqs_packed = pack_padded_sequence(embedded, lengths, batch_first=True)

        seqs, _ = self.rnn(seqs_packed)

        unpacked_output, _ = pad_packed_sequence(seqs, batch_first=True)

        idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(len(lengths), unpacked_output.size(2))
        idx = idx.unsqueeze(1)

        last_output = unpacked_output.gather(1, idx).squeeze(1)

        output = self.output(last_output)

        return output
