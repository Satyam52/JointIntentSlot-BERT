import warnings
import sys
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaModel,
)
from utils import seed_all, empty_cuda_cache, compute_metrics
from models import JointIntentSlot
from data_loader import LoadDataset
from data_tokenizer import TokenizeDataset
import torch.nn as nn
import tqdm
import json
import csv
import os
from config import *

warnings.filterwarnings("ignore")


def main(args):
    # Parse Argument
    TASK = str(args.task)
    EPOCH = int(args.epoch)
    LR = float(args.lr)
    BATCH_SIZE = int(args.batch)
    SEED = int(args.seed)
    BEST = bool(args.best)
    LANG = str(args.lang)

    print(f"============================================================")
    print(f"{time.strftime('%c', time.localtime(time.time()))}")
    print(f"TRAINING LANGUAGE: {LANG}\n")
    # print(f"TASK: {TASK}")
    print(f"EPOCH: {EPOCH}")
    print(f"LR: {LR}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"SEED: {SEED}")
    print(f"BEST: {BEST}\n")

    # Set Random Seed
    seed_all(SEED)

    # Load Dataset
    default_path = f"data/processed/{LANG}"
    seq_train = LoadDataset.load_dataset(f"{default_path}/train/seq.in")
    seq_dev = LoadDataset.load_dataset(f"{default_path}/dev/seq.in")

    intent_train = LoadDataset.load_dataset(f"{default_path}/train/label")
    intent_dev = LoadDataset.load_dataset(f"{default_path}/dev/label")
    intent_labels = LoadDataset.load_dataset(f"{default_path}/intent_label_vocab")

    slot_train = LoadDataset.load_dataset(f"{default_path}/train/seq.out", slot=True)
    slot_dev = LoadDataset.load_dataset(f"{default_path}/dev/seq.out", slot=True)
    slot_labels = LoadDataset.load_dataset(f"{default_path}/slot_label_vocab")

    # Label Indexing
    intent_word2idx = defaultdict(int, {k: v for v, k in enumerate(intent_labels)})
    intent_idx2word = {v: k for v, k in enumerate(intent_labels)}
    slot_word2idx = defaultdict(int, {k: v for v, k in enumerate(slot_labels)})
    slot_idx2word = {v: k for v, k in enumerate(slot_labels)}

    # Load Tokenizer & Model
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    model_config = XLMRobertaConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(intent_idx2word),
        problem_type="single_label_classification",
        id2label=intent_idx2word,
        label2id=intent_word2idx,
    )

    model = JointIntentSlot.from_pretrained(
        MODEL_NAME,
        config=model_config,
        num_intent_labels=len(intent_labels),
        num_slot_labels=len(slot_labels),
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Tokenize Datasets
    train_dataset = TokenizeDataset(seq_train, intent_train, slot_train, intent_word2idx, slot_word2idx, tokenizer)
    dev_dataset = TokenizeDataset(seq_dev, intent_dev, slot_dev, intent_word2idx, slot_word2idx, tokenizer)

    # Set Training Arguments and Train
    arguments = TrainingArguments(
        output_dir="checkpoints",
        do_train=True,
        do_eval=True,
        num_train_epochs=EPOCH,
        learning_rate=LR,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        load_best_model_at_end=BEST,
        report_to="none",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        # fp16=True,
    )

    trainer = Trainer(model, arguments, train_dataset=train_dataset, eval_dataset=dev_dataset)

    empty_cuda_cache()
    trainer.train()
    model.save_pretrained(f"checkpoints/{TASK}_ep{EPOCH}")


def evalFun(path, args, lang):
    # Parse Argument
    TASK = str(args.task)
    EPOCH = int(args.epoch)
    LR = float(args.lr)
    BATCH_SIZE = int(args.batch)
    SEED = int(args.seed)
    BEST = bool(args.best)
    LANG = str(lang)

    print(f"===========================Evaluating================================")
    print(f"{time.strftime('%c', time.localtime(time.time()))}")
    # print(f"TASK: {TASK}")
    print(f"EVALUATION LANGUAGE: {LANG}")
    print(f"EPOCH: {EPOCH}")
    print(f"LR: {LR}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"SEED: {SEED}")
    print(f"BEST: {BEST}\n")

    seed_all(SEED)
    default_path = f"data/processed/{LANG}"

    seq_test = LoadDataset.load_dataset(f"{default_path}/test/seq.in")

    intent_test = LoadDataset.load_dataset(f"{default_path}/test/label")
    intent_labels = LoadDataset.load_dataset(f"{default_path}/intent_label_vocab")

    slot_test = LoadDataset.load_dataset(f"{default_path}/test/seq.out", slot=True)
    slot_labels = LoadDataset.load_dataset(f"{default_path}/slot_label_vocab")

    # Label Indexing
    intent_word2idx = defaultdict(int, {k: v for v, k in enumerate(intent_labels)})
    intent_idx2word = {v: k for v, k in enumerate(intent_labels)}

    slot_word2idx = defaultdict(int, {k: v for v, k in enumerate(slot_labels)})
    slot_idx2word = {v: k for v, k in enumerate(slot_labels)}

    # Load Tokenizer & Model
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    model_config = XLMRobertaConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(intent_idx2word),
        problem_type="intent_classification",
        id2label=intent_idx2word,
        label2id=intent_word2idx,   
    )

    model = JointIntentSlot.from_pretrained(
        path, num_intent_labels=len(intent_labels), num_slot_labels=len(slot_labels), ignore_mismatched_sizes=True
    )

    # Tokenize Datasets
    test_dataset = TokenizeDataset(seq_test, intent_test, slot_test, intent_word2idx, slot_word2idx, tokenizer)

    intent_label_ids = []
    slot_label_ids = []

    with open(f"{default_path}/test/label", "r", encoding="utf-8") as intent_f, open(
        f"{default_path}/test/seq.out", "r", encoding="utf-8"
    ) as slot_f:
        for line in intent_f:
            line = line.strip()
            intent_label_ids.append(line)
        intent_label_ids = np.array(intent_label_ids)

        for line in slot_f:
            line = line.strip().split()
            slot_label_ids.append(line)

    # Predict on test data
    def predict(model, seqs):
        model.to("cpu")
        pred_intent_ids = []
        pred_slot_ids = []

        for i in tqdm.tqdm(range(2040, len(seqs))):
            input_seq = tokenizer(seqs[i], return_tensors="pt", max_length=50)

            model.eval()
            with torch.no_grad():
                _, (intent_logits, slot_logits) = model(**input_seq)

            # Intent
            pred_intent_ids.append(intent_idx2word[intent_logits[0].argmax().item()])

            # Slot
            slot_logits_size = slot_logits[0].shape[0]
            slot_logits_mask = np.array(test_dataset[i]["slot_label_ids"][:slot_logits_size]) != -100
            slot_logits_clean = slot_logits[0][slot_logits_mask]
            pred_slot_ids.append([slot_idx2word[i.item()] for i in slot_logits_clean.argmax(dim=1)])

        return np.array(pred_intent_ids), pred_slot_ids

    pred_intent_ids, pred_slot_ids = predict(model, seq_test)

    print(f"\n{time.strftime('%c', time.localtime(time.time()))}")
    res = compute_metrics(pred_intent_ids, intent_label_ids, pred_slot_ids, slot_label_ids)
    for k, v in res.items():
        print(f"{k:<20}: {v}")
    os.makedir("results/metrics")
    with open(f"results/metrics/{LANG}.json", "w", encoding="utf-8") as f:
        json.dump(res, f)

    # Save results and outputs
    with open("result/pred_intent", "w") as f, open("result/pred_slots", "w", newline="") as f2, open(
        "result/actual_intent", "w"
    ) as f3, open("result/actual_slots", "w", newline="") as f4, open("result/eval_metric.json", "w") as f5:
        csv.writer(f, delimiter="\n").writerow(pred_intent_ids.tolist())
        csv.writer(
            f2,
            delimiter=" ",
        ).writerows(pred_slot_ids)
        csv.writer(f3, delimiter="\n").writerow(intent_label_ids.tolist())
        csv.writer(f4, delimiter=" ").writerows(slot_label_ids)
        json.dump(res, f5)

    print(f"============================================================\n\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="massive")
    parser.add_argument("--epoch", default=10)
    parser.add_argument("--lr", default=5e-5)
    parser.add_argument("--batch", default=64)
    parser.add_argument("--seed", default=1234)
    parser.add_argument("--best", default=True)
    parser.add_argument("--mode", default="eval")
    parser.add_argument("--lang", default="en_US")  # default language for training

    args = parser.parse_args()
    if args.mode == "train":
        main(args)

    # Benchmark on all language
    if BENCHMARK_ALL:
        multilingual_data = os.listdir("data/processed")
        os.makedirs("results", exist_ok=True)
        for singlelingual_data in multilingual_data:
            name = singlelingual_data.split(".")[0].replace("-", "_")
            if ONLY_INDIAN and name not in INDIAN_LANGUAGE:  # IF benchmarking only on INDIAN LANG
                continue
            evalFun(path=f"checkpoints/{args.task}_ep{args.epoch}/", args=args, lang=name)
    else:  # Only on trained language
        evalFun(path=f"checkpoints/{args.task}_ep{args.epoch}/", args=args, lang=str(args.lang))
