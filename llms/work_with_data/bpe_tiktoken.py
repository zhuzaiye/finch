#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# DATE: 2024/2/25 17:40
# DESC:

import tiktoken

from bpe_openai_gpt2 import get_encoder, download_vocab


def bpe_encoding_with_tiktoken(text: str):
    tik_tokenizer = tiktoken.get_encoding("gpt2")
    integers = tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    strings = tik_tokenizer.decode(integers)
    print(strings)

    print(tik_tokenizer.n_vocab)


def bpe_encoding_with_gpt2(text: str):
    # 下载gpt2的原始vocab
    # download_vocab()
    original_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")
    integers = original_tokenizer.encode(text)
    strings = original_tokenizer.decode(integers)
    print(strings)


if __name__ == '__main__':
    text = "Hello, world. Is this-- a test?"
    # bpe_encoding_with_gpt2(text)
    bpe_encoding_with_gpt2(text)
