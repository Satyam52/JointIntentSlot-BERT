import warnings
import sys
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
from transformers import BertConfig, BertTokenizer, TrainingArguments, Trainer, BertModel, BertPreTrainedModel
from utils import seed_all, empty_cuda_cache, compute_metrics
from models import JointIntentSlot
from data_loader import LoadDataset
from data_tokenizer import TokenizeDataset
import torch.nn as nn
import tqdm
import json
import csv

warnings.filterwarnings("ignore")


def main(args):
    # Parse Argument
    TASK = str(args.task)
    EPOCH = int(args.epoch)
    LR = float(args.lr)
    BATCH_SIZE = int(args.batch)
    SEED = int(args.seed)
    BEST = bool(args.best)
    MODEL_NAME = str(args.model_name)

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
    load_dataset = LoadDataset.load_dataset
    seq_train = load_dataset(f'data/{TASK}/train/seq.in')
    seq_dev = load_dataset(f'data/{TASK}/dev/seq.in')

    domain_train = load_dataset(f'data/{TASK}/train/domain')
    domain_dev = load_dataset(f'data/{TASK}/dev/domain')
    domain_labels = load_dataset(f'data/{TASK}/domain_label_vocab')

    intent_train = load_dataset(f'data/{TASK}/train/label')
    intent_dev = load_dataset(f'data/{TASK}/dev/label')
    intent_labels = load_dataset(f'data/{TASK}/intent_label_vocab')

    slot_train = load_dataset(f'data/{TASK}/train/seq.out', slot=True)
    slot_dev = load_dataset(f'data/{TASK}/dev/seq.out', slot=True)
    slot_labels = load_dataset(f'data/{TASK}/slot_label_vocab')

    # Label Indexing
    domain_word2idx = defaultdict(int, {k: v for v, k in enumerate(domain_labels)})
    domain_idx2word = {v: k for v, k in enumerate(domain_labels)}
    intent_word2idx = defaultdict(int, {k: v for v, k in enumerate(intent_labels)})
    intent_idx2word = {v: k for v, k in enumerate(intent_labels)}
    slot_word2idx = defaultdict(int, {k: v for v, k in enumerate(slot_labels)})
    slot_idx2word = {v: k for v, k in enumerate(slot_labels)}

    # Load Tokenizer & Model
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model_config = BertConfig.from_pretrained(MODEL_NAME, num_labels=len(intent_idx2word),
                                              problem_type="single_label_classification", id2label=intent_idx2word,
                                              label2id=intent_word2idx)

    model = JointIntentSlot.from_pretrained(MODEL_NAME, config=model_config,
                                            num_intent_labels=len(intent_labels), num_slot_labels=len(slot_labels),
                                            num_domain_labels=len(domain_labels), ignore_mismatched_sizes=True)
    model.to('cuda')

    # Tokenize Datasets
    train_dataset = TokenizeDataset(seq_train, intent_train, slot_train, domain_train, intent_word2idx, slot_word2idx,
                                    domain_word2idx, tokenizer)
    dev_dataset = TokenizeDataset(seq_dev, intent_dev, slot_dev, domain_dev, intent_word2idx, slot_word2idx,
                                  domain_word2idx, tokenizer)

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
    MODEL_NAME = str(args.model_name)

    print(f'===========================Evaluating================================')
    print(f"{time.strftime('%c', time.localtime(time.time()))}")
    print(f'TASK: {TASK}')
    print(f'EPOCH: {EPOCH}')
    print(f'LR: {LR}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'SEED: {SEED}')
    print(f'BEST: {BEST}\n')

    seed_all(SEED)

    load_dataset = LoadDataset.load_dataset
    seq_test = load_dataset(f'data/{TASK}/test/seq.in')

    domain_test = load_dataset(f'data/{TASK}/test/domain')
    domain_labels = load_dataset(f'data/{TASK}/domain_label_vocab')

    intent_test = load_dataset(f'data/{TASK}/test/label')
    intent_labels = load_dataset(f'data/{TASK}/intent_label_vocab')

    slot_test = load_dataset(f'data/{TASK}/test/seq.out', slot=True)
    slot_labels = load_dataset(f'data/{TASK}/slot_label_vocab')

    # Label Indexing
    domain_word2idx = defaultdict(int, {k: v for v, k in enumerate(domain_labels)})
    domain_idx2word = {v: k for v, k in enumerate(domain_labels)}

    intent_word2idx = defaultdict(int, {k: v for v, k in enumerate(intent_labels)})
    intent_idx2word = {v: k for v, k in enumerate(intent_labels)}

    slot_word2idx = defaultdict(int, {k: v for v, k in enumerate(slot_labels)})
    slot_idx2word = {v: k for v, k in enumerate(slot_labels)}

    # Load Tokenizer & Model
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    model_config = BertConfig.from_pretrained(MODEL_NAME,
                                              num_labels=len(intent_idx2word),
                                              problem_type="intent_classification",
                                              id2label=intent_idx2word,
                                              label2id=intent_word2idx)

    model = JointIntentSlot.from_pretrained(path, num_intent_labels=len(intent_labels),
                                            num_slot_labels=len(slot_labels), num_domain_labels=len(domain_labels))

    # Tokenize Datasets
    test_dataset = TokenizeDataset(seq_test, intent_test, slot_test, domain_test, intent_word2idx, slot_word2idx,
                                   domain_word2idx, tokenizer)

    domain_label_ids = []
    intent_label_ids = []
    slot_label_ids = []

    with open(f'./data/{TASK}/test/domain', 'r', encoding='utf-8') as domain_f, \
            open(f'./data/{TASK}/test/label', 'r', encoding='utf-8') as intent_f, \
            open(f'./data/{TASK}/test/seq.out', 'r', encoding='utf-8') as slot_f:
        for line in domain_f:
            domain_label_ids.append(line.strip())
        domain_label_ids = np.array(domain_label_ids)

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
        pred_domain_ids = []
        pred_intent_ids = []
        pred_slot_ids = []
        idx = 0
        for i in tqdm.tqdm(range(len(seqs))):
            input_seq = tokenizer(seqs[i], return_tensors='pt')

            model.eval()
            with torch.no_grad():
                _, (domain_logits, intent_logits, slot_logits) = model(**input_seq)

            # Domain
            pred_domain_ids.append(domain_idx2word[domain_logits[0].argmax().item()])

            # Intent
            pred_intent_ids.append(intent_idx2word[intent_logits[0].argmax().item()])

            # Slot
            slot_logits_size = slot_logits[0].shape[0]
            slot_logits_mask = np.array(test_dataset[i]['slot_label_ids'][:slot_logits_size]) != -100
            slot_logits_clean = slot_logits[0][slot_logits_mask]
            pred_slot_ids.append([slot_idx2word[i.item()] for i in slot_logits_clean.argmax(dim=1)])
            # if idx == 5:
            #     break
            # idx += 1
        return np.array(pred_domain_ids), np.array(pred_intent_ids), pred_slot_ids

    pred_domain_ids, pred_intent_ids, pred_slot_ids = predict(model, seq_test)

    res = {}
    print(f"\n{time.strftime('%c', time.localtime(time.time()))}")
    res = compute_metrics(pred_intent_ids, intent_label_ids,
                          pred_slot_ids, slot_label_ids, pred_domain_ids, domain_label_ids)
    for k, v in res.items():
        print(f'{k:<20}: {v}')

    # Save results and outputs
    with open('result/pred_intent', 'w') as f, \
            open('result/pred_slots', 'w', newline='') as f2, \
            open('result/actual_intent', 'w') as f3, \
            open('result/actual_slots', 'w', newline='') as f4, \
            open('result/eval_metric.json', 'w') as f5, \
            open('result/pred_domain', 'w') as f6, \
            open('result/actual_domain', 'w') as f7, \
            open('result/mismatch.csv', 'w', newline='') as f8:
        csv.writer(f, delimiter='\n').writerow(pred_intent_ids.tolist())
        csv.writer(f2, delimiter=' ', ).writerows(pred_slot_ids)
        csv.writer(f3, delimiter='\n').writerow(intent_label_ids.tolist())
        csv.writer(f4, delimiter=' ').writerows(slot_label_ids)
        csv.writer(f6, delimiter='\n').writerow(pred_domain_ids.tolist())
        csv.writer(f7, delimiter='\n').writerow(domain_label_ids.tolist())
        # print(list(zip(seq_test, pred_domain_ids, domain_label_ids)))

        # Save Mismatched CSV
        csv.writer(f8, delimiter=',').writerow(["Text", "Domain", "Predicted domain", "Intent", "Predicted intent",
                                                "Slots", "Predicted slots"])
        for i in range(len(pred_domain_ids)):
            if (domain_label_ids[i] != pred_domain_ids[i]) or (intent_label_ids[i] != pred_intent_ids[i]) or \
                    (slot_label_ids[i] != pred_slot_ids[i]):
                csv.writer(f8, delimiter=',').writerow([seq_test[i], domain_label_ids[i], pred_domain_ids[i],
                                                        intent_label_ids[i], pred_intent_ids[i],
                                                        " ".join(slot_label_ids[i]),
                                                        " ".join(pred_slot_ids[i])
                                                        ])
        json.dump(res, f5)

    print(f'============================================================\n\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='massive')
    parser.add_argument('--epoch', default=7)
    parser.add_argument('--lr', default=5e-5)
    parser.add_argument('--batch', default=64)
    parser.add_argument('--seed', default=1234)
    parser.add_argument('--best', default=True)
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--model_name', default="bert-base-uncased")

    args = parser.parse_args()
    if args.mode == 'eval':
        evalFun(path=f"checkpoints/{args.task}_ep{args.epoch}/", args=args)
    else:
        main(args)
        evalFun(path=f"checkpoints/{args.task}_ep{args.epoch}/", args=args)
