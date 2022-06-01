CUDA_VISIBLE_DEVICES=0

# tokenizing
python tools/normalize.py --src_fnames train.en --tgt_fnames train.de --src_out_fnames train.norm.tok.en --tgt_out_fnames train.norm.tok.de

# bpe
subword-nmt learn-joint-bpe-and-vocab --input train.norm.tok.en train.norm.tok.de -s 10000 -o bpe10000 --write-vocabulary vocab.norm.tok.bpe10000.en vocab.norm.tok.bpe10000.de
subword-nmt apply-bpe -c bpe10000 --vocabulary vocab.norm.tok.bpe10000.en --vocabulary-threshold 50 < train.norm.tok.en > train.norm.tok.bpe10000.en
subword-nmt apply-bpe -c bpe10000 --vocabulary vocab.norm.tok.bpe10000.de --vocabulary-threshold 50 < train.norm.tok.de > train.norm.tok.bpe10000.de

#lowercasing
python tools/low.py --src_fnames train.norm.tok.bpe10000.en --tgt_fnames train.norm.tok.bpe10000.de --src_out_fnames train.norm.tok.lc.bpe10000.en --tgt_out_fnames train.norm.tok.lc.bpe10000.en

python preprocess.py -train_src multi30K/training/train.norm.tok.lc.bpe10000.en -t train_tgt multi30K/training/train.norm.tok.lc.bpe10000.de -valid_src multi30K/validation/val.norm.tok.lc.bpe10000.en -valid_tgt multi30K/validation/val.norm.tok.lc.bpe10000.de -save_data multi30K/m30k

./tools/embeddings_to_torch.py -emb_file "glove_dir/glove.6B.300d.txt" \
-dict_file $multi30K/m30k.vocab.pt \
-output_file $multi30K/embeddings

python extract_image.py --dataset_name multi30k --splits "train,test,valid" --image_path multi30K/flickr30k-images --train_fnames multi30K/splits/train.txt --valid_fnames multi30K/splits/val.txt --test_fnames multi30K/splits/test.txt
python extract_image_features.py --dataset_name multi30k --splits "train,test,valid" --image_path multi30K/flickr30k-images --train_fnames multi30K/splits/train.txt --valid_fnames multi30K/splits/val.txt --test_fnames multi30K/splits/test.txt --gpuid 0 --pretrained_cnn restnet50

#MODEL_PATH=model.query.back.b128
#MODEL_SNAPSHOT=ADAM_acc_87.16_ppl_1.74_e22.pt

echo ${MODEL_PATH}/${MODEL_SNAPSHOT}

python train_mm_gan.py -data ${DATA_PATH}/demo -save_model ${MODEL_PATH}/ADAM \
-path_to_train_img_feats ${DATA_PATH}/flickr30k_train_resnet50_cnn_features.hdf5 \
-path_to_valid_img_feats data/flickr30k/features/flickr30k_valid_resnet50_cnn_features.hdf5 \
-gpuid 0 -epochs 200 -layers 6 -rnn_size 300 -word_vec_size 512 \
-encoder_type transformer -decoder_type mmtransformer -position_encoding \
-max_generator_batches 2 -dropout 0.1 \
-batch_size 128 -accum_count 1 -use_both \
-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
-max_grad_norm 0 -param_init 0 \
-label_smoothing 0.1 --multimodal_model_type gantransformer \
-pre_word_vecs_enc ${DATA_PATH}/embeddings.enc.pt \
-pre_word_vecs_dec ${DATA_PATH}/embeddings.dec.pt \
-branch_num 2 -mm_enc_layers 6


# -train_from ${MODEL_PATH}/${MODEL_SNAPSHOT} \

# -train_from model_snapshots.stanford.graphtransformer.query/IMGD_ADAM_acc_68.29_ppl_10.79_e94.pt \
 
python translate_mm.py -src $multi30K/test_2016_flickr.lc.norm.tok.en -model ${MODEL_PATH}/${MODEL_SNAPSHOT} -path_to_test_img_feats data/flickr30k/features/flickr30k_test_resnet50_cnn_features.hdf5 -output ${MODEL_PATH}/${MODEL_SNAPSHOT}.translations-test2016

perl tools/multi-bleu.perl $multi30K/test_2016_flickr.lc.norm.tok.de < ${MODEL_PATH}/${MODEL_SNAPSHOT}.translations-test2016 #> result.query.back.b128.86ep

echo ${MODEL_PATH}/${MODEL_SNAPSHOT}
