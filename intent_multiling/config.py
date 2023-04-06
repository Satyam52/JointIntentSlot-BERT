import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 64
MODEL_NAME = "xlm-roberta-base"
MAX_TOKEN_LEN = 256
ONLY_INDIAN = False
BENCHMARK_ALL = True  # ONLY training or all or indian
CHECKPOINTS = 5  # No of snaps saved
INDIAN_LANGUAGE = ["", ""]
