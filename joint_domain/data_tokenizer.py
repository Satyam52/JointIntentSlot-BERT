import torch


class TokenizeDataset:
    def __init__(self, seqs, intent_labels, slot_labels, domain_labels, intent_word2idx, slot_word2idx, domain_word2idx,
                 tokenizer):
        self.seqs = seqs
        self.intent_labels = intent_labels
        self.slot_labels = slot_labels
        self.domain_labels = domain_labels
        self.intent_word2idx = intent_word2idx
        self.slot_word2idx = slot_word2idx
        self.domain_word2idx = domain_word2idx

        self.tokenizer = tokenizer

    def align_label(self, seq, intent_label, slot_label, domain_label):
        tokens = self.tokenizer(seq, padding='max_length', max_length=50, truncation=True)

        slot_label_ids = [-100]
        for word_idx, word in enumerate(seq.split()):
            slot_label_ids += [self.slot_word2idx[slot_label[word_idx]]] + [-100] * (
                    len(self.tokenizer.tokenize(word)) - 1)  # [slot label id] + [subword tails padding]
        if len(slot_label_ids) >= 50:
            slot_label_ids = slot_label_ids[:49] + [-100]
        else:
            slot_label_ids += [-100] * (50 - len(slot_label_ids))

        tokens['domain_label_ids'] = [self.domain_word2idx[domain_label]]
        tokens['intent_label_ids'] = [self.intent_word2idx[intent_label]]
        tokens['slot_label_ids'] = slot_label_ids

        return tokens

    def __getitem__(self, index):
        bert_input = self.align_label(self.seqs[index], self.intent_labels[index], self.slot_labels[index],
                                      self.domain_labels[index])
        return bert_input

    def __len__(self):
        return len(self.seqs)
