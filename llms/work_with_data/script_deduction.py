#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
# DATE: 2024/2/25 14:58
# DESC: 数据文本处理

import re


def gen_vocab_from_verdict_txt(verdict_raw_txt_path: str):
    # load the raw text source
    fp = open("verdict.txt", "r", encoding="utf-8")
    verdict_raw_txt = fp.read()
    fp.close()
    print("Total number of characters: ", len(verdict_raw_txt))

    # split the raw text
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', verdict_raw_txt)
    # removes empty strings
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_tokens = sorted(list(set(preprocessed)))
    print("unique vocab size: ", len(all_tokens))

    # add special tokens like "<|unk|>" to the vocabulary to represent unknown words
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # process unknown
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


if __name__ == '__main__':
    vocab = gen_vocab_from_verdict_txt("verdict.txt")
    tokenizer = SimpleTokenizerV1(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."

    text = " <|endoftext|> ".join((text1, text2))
    print(text)

    print(tokenizer.encode(text))

    print(tokenizer.decode(tokenizer.encode(text)))
