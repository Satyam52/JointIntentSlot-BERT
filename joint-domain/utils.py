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


def get_domain_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        'domain_acc': acc
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels, domain_preds, domain_lables):
    intent_result = (intent_preds == intent_labels)
    domain_result = (domain_preds == domain_lables)

    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    semantic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "semantic_frame_acc": semantic_acc
    }


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels, domain_preds, domain_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)

    results = {}
    domain_result = get_domain_acc(domain_preds, domain_labels)
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    semantic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels, domain_preds,
                                             domain_labels)

    results.update(domain_result)
    results.update(intent_result)
    results.update(slot_result)
    results.update(semantic_result)

    return results
