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
python extract_image_features.py --splits=train,test,valid --images_path /path/to/flickr30k/images/ --train_fnames /path/to/flickr30k/train_images.txt --valid_fnames /path/to/flickr30k/val_images.txt --test_fnames /path/to/flickr30k/test2016_images.txt --gpuid 0 --pretrained_cnn restnet50
```

### Step 1: Preprocess the text data
After pre-processing them (e.g. normalizing, [tokenising](https://github.com/OpenNMT/Tokenizer), lowercasing, and applying a [BPE model](https://github.com/rsennrich/subword-nmt)), feed the training and validation sets to the `preprocess.py` script, as below.

```bash
python preprocess.py -train_src /path/to/flickr30k/train.norm.tok.lc.bpe10000.en -train_tgt /path/to/flickr30k/train.norm.tok.lc.bpe10000.de -valid_src /path/to/flickr30k/val.norm.tok.lc.bpe10000.en -valid_tgt /path/to/flickr30k/val.norm.tok.lc.bpe10000.de -save_data data/multi30k
```

Prepareing the glove 300d embedding, and then run this command to get the embedding files.
```bash
python tools/embeddings_to_torch.py -emb_file "glove_dir/glove.6B.300d.txt" \
-dict_file $path/to/multi30k/vocab.pt\
-output_file $path/to/multi30k/embeddings.pt
```

### Step 2: Train the model

```bash run.sh```