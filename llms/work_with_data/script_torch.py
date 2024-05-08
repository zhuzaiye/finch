#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# DATE: 2024/2/25 16:16
# DESC:

import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        # Use s sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_datat_loader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True):
    # 初始化Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建Dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


if __name__ == '__main__':
    rp = open("verdict.txt", "r", encoding="utf-8")
    raw_text = rp.read()
    rp.close()

    data_loader = create_datat_loader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(data_loader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)
