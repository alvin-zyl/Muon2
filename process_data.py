"""Download a FineWeb sample, tokenize it with the GPT-2 tokenizer and write binary shards.

The first shard is used as the validation set (val_000000.bin), the rest for training
(train_000001.bin, ...). Shards are written to $DATA_DIR/<name>-<tokenizer>/.
"""
import os
import argparse

import tiktoken
from datasets import load_dataset

from utils.data_utils import process_and_save_docs
from utils.dataloader import get_dataset_path

datadict = {
    "fineweb10B": ["HuggingFaceFW/fineweb", "sample-10BT"],
    "fineweb100B": ["HuggingFaceFW/fineweb", "sample-100BT"],
}

parser = argparse.ArgumentParser(description="Tokenize a FineWeb sample into binary shards")
parser.add_argument("--name", type=str, required=True, choices=list(datadict), help="Name of the dataset")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-t", "--tokenizer", type=str, default="gpt2", help="tiktoken encoding to use")
parser.add_argument("-n", "--nprocs", type=int, default=0, help="number of processes, default N-2")
args = parser.parse_args()

hf_path, remote_name = datadict[args.name]
enc = tiktoken.get_encoding(args.tokenizer)

dataset_path = get_dataset_path(args.name, args.tokenizer)
os.makedirs(dataset_path, exist_ok=True)
print("Data will be saved in the path : ", dataset_path)

dataset = load_dataset(hf_path, name=remote_name, split="train")
process_and_save_docs(dataset, dataset_path, encoding=enc, shard_size=args.shard_size, nprocs=args.nprocs)

print(f"{args.name} data processed and saved in {dataset_path}")
