import warnings
import sys
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
from transformers import BertConfig, BertTokenizer, TrainingArguments, Trainer, BertModel, BertPreTrainedModel
from utils import seed_all, empty_cuda_cache, compute_metrics
from models import SlotClassifier
from data_loader import LoadDataset
from data_tokenizer import TokenizeDataset
import torch.nn as nn
import tqdm
warnings.filterwarnings("ignore")


def main(args):
    # Parse Argument
    TASK = str(args.task)
    EPOCH = int(args.epoch)
    LR = float(args.lr)
    BATCH_SIZE = int(args.batch)
    SEED = int(args.seed)
    BEST = bool(args.best)

    print(f'============================================================')
    print(f"{time.strftime('%c', time.localtime(time.time()))}")
    print(f'TASK: {TASK}')
    print(f'EPOCH: {EPOCH}')
    print(f'LR: {LR}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'SEED: {SEED}')
    print(f'BEST: {BEST}\n')

    # Set Random Seed
    seed_all(SEED)

    # Load Dataset
    seq_train = LoadDataset.load_dataset(f'data/{TASK}/train/seq.in')
    seq_dev = LoadDataset.load_dataset(f'data/{TASK}/dev/seq.in')

    slot_train = LoadDataset.load_dataset(f'data/{TASK}/train/seq.out', slot = True)
    slot_dev = LoadDataset.load_dataset(f'data/{TASK}/dev/seq.out', slot = True)
    slot_labels = LoadDataset.load_dataset(f'data/{TASK}/slot_label_vocab')

    # Label Indexing
    slot_word2idx = defaultdict(int, {k: v for v, k in enumerate(slot_labels)})
    slot_idx2word = {v: k for v, k in enumerate(slot_labels)}

    # Load Tokenizer & Model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model_config = BertConfig.from_pretrained("bert-base-uncased", problem_type="single_label_classification")

    model = SlotClassifier.from_pretrained(
        "bert-base-uncased", config=model_config, num_slot_labels=len(slot_labels))
    model.to('cuda')

    # Tokenize Datasets
    train_dataset = TokenizeDataset(
        seq_train, slot_train,  slot_word2idx,  tokenizer)
    dev_dataset = TokenizeDataset(
        seq_dev, slot_dev,  slot_word2idx,  tokenizer)

    # Set Training Arguments and Train
    arguments = TrainingArguments(
        output_dir='checkpoints',
        do_train=True,
        do_eval=True,

        num_train_epochs=EPOCH,
        learning_rate=LR,

        save_strategy="epoch",
        save_total_limit=10,
        evaluation_strategy="epoch",
        load_best_model_at_end=BEST,

        report_to='none',

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        fp16=True,

    )

    trainer = Trainer(
        model,
        arguments,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset
    )

    empty_cuda_cache()
    trainer.train()
    model.save_pretrained(f"checkpoints/{TASK}_ep{EPOCH}")


def evalFun(path, args):
    # Parse Argument
    TASK = str(args.task)
    EPOCH = int(args.epoch)
    LR = float(args.lr)
    BATCH_SIZE = int(args.batch)
    SEED = int(args.seed)
    BEST = bool(args.best)

    print(f'===========================Evaluating================================')
    print(f"{time.strftime('%c', time.localtime(time.time()))}")
    print(f'TASK: {TASK}')
    print(f'EPOCH: {EPOCH}')
    print(f'LR: {LR}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'SEED: {SEED}')
    print(f'BEST: {BEST}\n')

    seed_all(SEED)

    seq_test = LoadDataset.load_dataset(f'data/{TASK}/test/seq.in')

    slot_labels = LoadDataset.load_dataset(f'data/{TASK}/slot_label_vocab')
    slot_test = LoadDataset.load_dataset(f'data/{TASK}/test/seq.out', slot = True)

    # Label Indexing
    slot_word2idx = defaultdict(int, {k: v for v, k in enumerate(slot_labels)})
    slot_idx2word = {v: k for v, k in enumerate(slot_labels)}

    # Load Tokenizer & Model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model_config = BertConfig.from_pretrained("bert-base-uncased", problem_type="single_label_classification")

    model = SlotClassifier.from_pretrained(path, num_slot_labels=len(slot_labels))

    # Tokenize Datasets
    test_dataset = TokenizeDataset(seq_test, slot_test,  slot_word2idx,  tokenizer)

    intent_label_ids = []
    slot_label_ids = []

    with open(f'./data/{TASK}/test/label', 'r', encoding='utf-8') as intent_f, \
            open(f'./data/{TASK}/test/seq.out', 'r', encoding='utf-8') as slot_f:
        for line in intent_f:
            line = line.strip()
            intent_label_ids.append(line)
        intent_label_ids = np.array(intent_label_ids)

        for line in slot_f:
            line = line.strip().split()
            slot_label_ids.append(line)

    # Predict on test data
    def predict(model, seqs):
        model.to('cpu')
        pred_slot_ids = []

        for i in tqdm.tqdm(range(len(seqs))):
            input_seq = tokenizer(seqs[i], return_tensors='pt')

            model.eval()
            with torch.no_grad():
                _, (slot_logits) = model(**input_seq)

            slot_logits_size = slot_logits[0].shape[0]
            slot_logits_mask = np.array(test_dataset[i]['slot_label_ids'][:slot_logits_size]) != -100
            slot_logits_clean = slot_logits[0][slot_logits_mask]
            pred_slot_ids.append([slot_idx2word[i.item()] for i in slot_logits_clean.argmax(dim=1)])

        return pred_slot_ids

    pred_slot_ids = predict(model, seq_test)

    print(f"\n{time.strftime('%c', time.localtime(time.time()))}")
    res = compute_metrics(pred_slot_ids, slot_label_ids)
    for k, v in res.items():
        print(f'{k:<20}: {v}')
    print(f'============================================================\n\n\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='massive')
    parser.add_argument('--epoch', default=10)
    parser.add_argument('--lr', default=5e-5)
    parser.add_argument('--batch', default=128)
    parser.add_argument('--seed', default=1234)
    parser.add_argument('--best', default=True)
    parser.add_argument('--mode', default='eval')

    args = parser.parse_args()
    if args.mode=='eval':
        evalFun(path=f"checkpoints/{args.task}_ep{args.epoch}/",args=args)
    else:
        main(args)
        evalFun(path=f"checkpoints/{args.task}_ep{args.epoch}/",args=args)
