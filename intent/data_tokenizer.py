import torch


class TokenizeDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, intent_labels, intent_word2idx, tokenizer):
        self.seqs = seqs
        self.intent_labels = intent_labels
        self.intent_word2idx = intent_word2idx
        self.tokenizer = tokenizer

    def align_label(self, seq, intent_label):
        tokens = self.tokenizer(seq, padding='max_length',
                                max_length=50, truncation=True)
        tokens['intent_label_ids'] = [self.intent_word2idx[intent_label]]
        return tokens

    def __getitem__(self, index):
        bert_input = self.align_label(
            self.seqs[index], self.intent_labels[index])
        return bert_input

    def __len__(self):
        return len(self.seqs)
