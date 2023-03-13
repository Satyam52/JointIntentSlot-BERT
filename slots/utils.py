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



def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def compute_metrics( slot_preds, slot_labels):
    assert  len(slot_preds) == len(slot_labels)

    results = {}
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    results.update(slot_result)
    
    return results
