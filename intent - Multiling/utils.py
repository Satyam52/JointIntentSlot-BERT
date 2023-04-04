import os
import random
import gc
import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score


def empty_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


def seed_all(seed: int = 1004):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def compute_metrics(intent_preds, intent_labels):
    assert len(intent_preds) == len(intent_labels)

    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)

    results.update(intent_result)

    return results
