import warnings
import json
import sys
import os
import argparse
import time
from collections import defaultdict
import numpy as np
import torch
from transformers import BertConfig, BertTokenizer, TrainingArguments, \
    Trainer, BertModel, BertPreTrainedModel, PretrainedConfig, AutoConfig, AutoTokenizer, AutoModelForMaskedLM, \
        XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaModel
from utils import seed_all, empty_cuda_cache, compute_metrics
from models import IntentClassification
from data_loader import LoadDataset
from data_tokenizer import TokenizeDataset
import torch.nn as nn
from config import *
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
    LANG = str(args.lang)

    print(f'============================================================')
    print(f"{time.strftime('%c', time.localtime(time.time()))}")
    print(f'TRAINING LANGUAGE: {LANG}\n')
    # print(f'TASK: {TASK}')
    print(f'EPOCH: {EPOCH}')
    print(f'LR: {LR}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'SEED: {SEED}')
    print(f'BEST: {BEST}')

    # Set Random Seed
    seed_all(SEED)
    

    # Load Dataset
    default_path = f'data/processed/{LANG}'
    seq_train = LoadDataset.load_dataset(f'{default_path}/train/seq.in')
    seq_dev = LoadDataset.load_dataset(f'{default_path}/dev/seq.in')

    intent_train = LoadDataset.load_dataset(f'{default_path}/train/label')
    intent_dev = LoadDataset.load_dataset(f'{default_path}/dev/label')
    intent_labels = LoadDataset.load_dataset(f'{default_path}/intent_label_vocab')

    # Label Indexing
    intent_word2idx = defaultdict(int, {k: v for v, k in enumerate(intent_labels)})
    intent_idx2word = {v: k for v, k in enumerate(intent_labels)}

    # Load Tokenizer & Model
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    model_config = XLMRobertaConfig.from_pretrained(MODEL_NAME, num_labels=len(
        intent_idx2word), problem_type="single_label_classification", id2label=intent_idx2word, label2id=intent_word2idx)

    model = IntentClassification.from_pretrained(MODEL_NAME, config=model_config, num_intent_labels=len(intent_labels))
    model.to('cuda')

    # Tokenize Datasets
    train_dataset = TokenizeDataset(seq_train, intent_train,  intent_word2idx,  tokenizer)
    dev_dataset = TokenizeDataset(seq_dev, intent_dev,  intent_word2idx,  tokenizer)

    # Set Training Arguments and Train
    arguments = TrainingArguments(
        output_dir='checkpoints',
        do_train=True,
        do_eval=True,

        num_train_epochs=EPOCH,
        learning_rate=LR,

        save_strategy="epoch",
        save_total_limit=CHECKPOINTS,
        evaluation_strategy="epoch",
        load_best_model_at_end=BEST,

        report_to='none',

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
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


def evalFun(path, args, lang):
    # Parse Argument
    TASK = str(args.task)
    EPOCH = int(args.epoch)
    LR = float(args.lr)
    BATCH_SIZE = int(args.batch)
    SEED = int(args.seed)
    BEST = bool(args.best)
    LANG = str(lang)

    print(f'===========================Evaluating================================')
    print(f"{time.strftime('%c', time.localtime(time.time()))}")
    # print(f'TASK: {TASK}')
    print(f'EVALUATION LANGUAGE: {LANG}')
    print(f'EPOCH: {EPOCH}')
    print(f'LR: {LR}')
    print(f'BATCH_SIZE: {BATCH_SIZE}')
    print(f'SEED: {SEED}')
    print(f'BEST: {BEST}\n')

    seed_all(SEED)
    default_path = f'data/processed/{LANG}'

    seq_test = LoadDataset.load_dataset(f'{default_path}/test/seq.in')

    intent_test = LoadDataset.load_dataset(f'{default_path}/test/label')
    intent_labels = LoadDataset.load_dataset(f'{default_path}/intent_label_vocab')

    # Label Indexing
    intent_word2idx = defaultdict(int, {k: v for v, k in enumerate(intent_labels)})
    intent_idx2word = {v: k for v, k in enumerate(intent_labels)}

    # Load Tokenizer & Model
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    model_config = XLMRobertaConfig.from_pretrained(MODEL_NAME, num_labels=len(
        intent_idx2word), problem_type="single_label_classification", id2label=intent_idx2word, label2id=intent_word2idx)

        
    model = IntentClassification.from_pretrained(path,config=model_config, num_intent_labels=len(intent_labels))

    # Tokenize Datasets
    test_dataset = TokenizeDataset(seq_test, intent_test,  intent_word2idx,  tokenizer)

    intent_label_ids = []
    slot_label_ids = []

    with open(f'{default_path}/test/label', 'r', encoding='utf-8') as intent_f, \
            open(f'{default_path}/test/seq.out', 'r', encoding='utf-8') as slot_f:
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
        pred_intent_ids = []

        for i in tqdm.tqdm(range(len(seqs))):
            input_seq = tokenizer(seqs[i], return_tensors='pt')

            model.eval()
            with torch.no_grad():
                _, (intent_logits) = model(**input_seq)
                # print(intent_logits)

            pred_intent_ids.append(
                intent_idx2word[intent_logits[0].argmax().item()])

        return np.array(pred_intent_ids)

    pred_intent_ids = predict(model, seq_test)

    print(f"\n{time.strftime('%c', time.localtime(time.time()))}")
    res = compute_metrics(pred_intent_ids, intent_label_ids)
    with open(f"results/{LANG}.json","w",encoding='utf-8') as f:
        json.dump(res, f)
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
    parser.add_argument("--lang", default='en_US') # default language for training

    args = parser.parse_args()
    if args.mode=='train':
        main(args)
        
    # Benchmark on all language
    if BENCHMARK_ALL:
        multilingual_data = os.listdir("data/json")
        os.makedirs('results', exist_ok=True)
        for singlelingual_data in multilingual_data:
            name = singlelingual_data.split(".")[0].replace("-", "_")
            if ONLY_INDIAN and name not in INDIAN_LANGUAGE: # IF benchmarking only on INDIAN LANG
                continue
            evalFun(path=f"checkpoints/{args.task}_ep{args.epoch}/", args=args, lang=name)
    else: # Only on trained language
        evalFun(path=f"checkpoints/{args.task}_ep{args.epoch}/", args=args, lang=str(args.lang))
