import argparse

def process(args):
    f_s = open(args.src_fnames, 'w')
    f_t = open(args.tgt_fnames, 'w')
    for line in open(args.src_out_fnames):
        line = line.strip()
        line = line.lower()
        f_s.write(line + '\n')
    f_s.close()
    for line in open(args.tgt_out_fnames):
        line = line.strip()
        line = line.lower()
        f_t.write(line + '\n')
    f_t.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="tokenize and normalize.")

    parser.add_argument("--src_fnames", type=str,
                        default='train.norm.tok.bpe10000.en',
                        help="""File containing train source sentence.""")
    parser.add_argument("--tgt_fnames", type=str,
                        default='train.norm.tok.bpe10000.de',
                        help="""File containing train target sentence.""")
    parser.add_argument("--src_out_fnames", type=str,
                        default="train.norm.tok.lc.bpe10000.en")
    parser.add_argument("--tgt_out_fnames", type=str,
                        default="train.norm.tok.lc.bpe10000.de")
    parser.add_argument("--tok", type=str,
                        default="conservative")
    args = parser.parse_args()


    process(args)