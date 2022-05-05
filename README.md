# MMT
## Requirements
```
torchtext>=0.2.1
torchvision >= 0.11.3
pytorch>=1.10.2
pytables>=3.7.0
```
## Quickstart

### Step 0: Prepare image data.

Downloaded the [Multi30k data set](http://www.statmt.org/wmt16/multimodal-task.html), and run the code:

```bash
python extract_image.py --split=train,valid,test --images_path /path/to/flickr30k/images/ --train_fnames /path/to/flickr30k/train_images.txt --valid_fnames /path/to/flickr30k/val_images.txt --test_fnames /path/to/flickr30k/test2016_images.txt
```

### Step 1: Preprocess the text data
After pre-processing them (e.g. normalizing, [tokenising](https://github.com/OpenNMT/Tokenizer), lowercasing, and applying a [BPE model](https://github.com/rsennrich/subword-nmt)), feed the training and validation sets to the `preprocess.py` script, as below.

```bash
python preprocess.py -train_src /path/to/flickr30k/train.norm.tok.lc.bpe10000.en -train_tgt /path/to/flickr30k/train.norm.tok.lc.bpe10000.de -valid_src /path/to/flickr30k/val.norm.tok.lc.bpe10000.en -valid_tgt /path/to/flickr30k/val.norm.tok.lc.bpe10000.de -save_data data/m30k
```

### Step 2: Train the model

```bash run.sh```