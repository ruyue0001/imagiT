#!/usr/bin/python
# -*- coding: utf-8 -*-

import unicodedata
import pyonmttok
import argparse
import re
import sys


def process(args):
    tokenizer = pyonmttok.Tokenizer(args.tok, joiner_annotate=False)

    f_s = open(args.src_out_fnames, 'w')
    f_t = open(args.tgt_out_fnames, 'w')
    for line in open(args.src_fnames):
        line = line.strip()
        line = unicodedata.normalize('NFC', line)
        line = line.replace(u"«", u"“").replace(u"»", u"”")
        #line = line.sub(r'(^|[^S\w])#([A-Za-z0-9_]+)', '\\1｟#\\2｠')
        line = tokenizer(line)
        line = tokenizer.detokenize(line)
        f_s.write(line + '\n')
    f_s.close()
    for line in open(args.tgt_fnames):
        line = line.strip()
        line = unicodedata.normalize('NFC', line)
        line = line.replace(u"«", u"“").replace(u"»", u"”")
        #line = line.sub(r'(^|[^S\w])#([A-Za-z0-9_]+)', '\\1｟#\\2｠')
        line = tokenizer(line)
        line = tokenizer.detokenize(line)
        f_t.write(line + '\n')
    f_t.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="tokenize and normalize.")

    parser.add_argument("--src_fnames", type=str,
                        default="train.en",
                        help="""File containing train source sentence.""")
    parser.add_argument("--tgt_fnames", type=str,
                        default="train.de",
                        help="""File containing train target sentence.""")
    parser.add_argument("--src_out_fnames", type=str,
                        default="train.norm.tok.en")
    parser.add_argument("--tgt_out_fnames", type=str,
                        default="train.norm.tok.de")
    parser.add_argument("--tok", type=str,
                        default="conservative")
    args = parser.parse_args()


    process(args)
