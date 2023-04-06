import numpy as np
import torch
from data_loader import LoadDataset
from data_tokenizer import TokenizeDataset
import pandas as pd
import re
import sys
import os
from config import *


def createBioLabels(annot_text):
    tag = ""
    arr = re.split(r"[\[\]]", annot_text)
    for wordGroup in arr:
        if ":" in wordGroup:
            namedEntity, words = wordGroup.replace("]", "").split(":")
            for idx, word in enumerate(words.strip().split()):
                tag += f"B-{namedEntity}" if idx == 0 else f"I-{namedEntity}"
        else:
            for _ in wordGroup.strip().split():
                tag += "O "
    return tag


if __name__ == "__main__":
    multilingual_data = os.listdir("data/json")
    os.makedirs("data/processed", exist_ok=True)
    for singlelingual_data in multilingual_data:
        name = singlelingual_data.split(".")[0].replace("-", "_")
        if ONLY_INDIAN and name not in INDIAN_LANGUAGE:  # IF benchmarking only on INDIAN LANG
            continue

        # make dir for each language (train, test and dev)
        os.makedirs(f"data/processed/{name}", exist_ok=True)
        os.makedirs(f"data/processed/{name}/train", exist_ok=True)
        os.makedirs(f"data/processed/{name}/test", exist_ok=True)
        os.makedirs(f"data/processed/{name}/dev", exist_ok=True)

        data = pd.read_json(f"data/json/{singlelingual_data}", lines=True)
        data.drop(["id", "locale", "worker_id"], axis=1, inplace=True)
        data["len"] = data["utt"].map(lambda x: len(x.strip().split()))

        # drop utterances whose len is greater than max_len
        data = data[data["len"] < MAX_TOKEN_LEN]

        for type in ["train", "test", "dev"]:
            with open(f"data/processed/{name}/{type}/label", "w", encoding="utf-8") as file:
                file.writelines("\n".join(data[data["partition"] == type]["intent"].values))
                file.close()

            with open(f"data/processed/{name}/{type}/seq.in", "w", encoding="utf-8") as file:
                file.writelines("\n".join(data[data["partition"] == type]["utt"].values))
                file.close()

            with open(f"data/processed/{name}/{type}/domain", "w", encoding="utf-8") as file:
                file.writelines("\n".join(data[data["partition"] == type]["scenario"].values))
                file.close()

            with open(f"data/processed/{name}/{type}/seq.out", "w", encoding="utf-8") as file:
                file.writelines("\n".join(data[data["partition"] == type]["annot_utt"].apply(createBioLabels).values))
                file.close()

        with open(f"data/processed/{name}/intent_label_vocab", "w", encoding="utf-8") as file:
            file.writelines("\n".join(data["intent"].unique()))
            file.close()

        with open(f"data/processed/{name}/domain_label_vocab", "w", encoding="utf-8") as file:
            file.writelines("\n".join(data["scenario"].unique()))
            file.close()

        with open(f"data/processed/{name}/slot_label_vocab", "w", encoding="utf-8") as file:
            slots = data["annot_utt"].apply(createBioLabels).values
            unique_slots = list({word.strip() for x in slots for word in x.split()})
            file.writelines("\n".join(unique_slots))
            file.close()

        print(f"Completed processing {name}")
