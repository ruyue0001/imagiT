CUDA_VISIBLE_DEVICES=7
DATA_PATH=data/

#python extract_image_features.py --gpuid 0 --pretrained_cnn resnet50 \
#--splits=train,valid,test --images_path /MultimodalNMT/flickr30k/images \ --train_fnames /MultimodalNMT/flickr30k/train_images.txt \
#--valid_fnames /MultimodalNMT/flickr30k/val_images.txt \
#--test_fnames /MultimodalNMT/flickr30k/test_images.txt

# ./tools/embeddings_to_torch.py -emb_file "glove_dir/glove.6B.300d.txt" \
# -dict_file ${DATA_PATH}/demo.vocab.pt \
# -output_file ${DATA_PATH}/embeddings

echo ${MODEL_PATH}/${MODEL_SNAPSHOT}

 python train_mm.py -data ${DATA_PATH} -save_model ${MODEL_PATH} \
 -path_to_train_img_feats ${DATA_PATH}/flickr30k_train_resnet50_cnn_features.hdf5 \
 -path_to_valid_img_feats ${DATA_PATH}/flickr30k/features/flickr30k_valid_resnet50_cnn_features.hdf5 \
 -gpuid 0 -epochs 200 -layers 6 -rnn_size 300 -word_vec_size 512 \
 -encoder_type transformer -decoder_type transformer -position_encoding \
 -max_generator_batches 2 -dropout 0.1 \
 -batch_size 128 -accum_count 1 -use_both \
 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
 -max_grad_norm 0 -param_init 0 \
 -label_smoothing 0.1 --multimodal_model_type graphtransformer \
 -pre_word_vecs_enc ${DATA_PATH}/embeddings.enc.pt \
 -pre_word_vecs_dec ${DATA_PATH}/embeddings.dec.pt \
 -branch_num 2


# -train_from ${MODEL_PATH}/${MODEL_SNAPSHOT} \

# -train_from model_snapshots.stanford.graphtransformer.query/IMGD_ADAM_acc_68.29_ppl_10.79_e94.pt \
 
#python translate_mm.py -src ${DATA_PATH}/test_2016_flickr.lc.norm.tok.en -model ${MODEL_PATH}/${MODEL_SNAPSHOT} -path_to_test_img_feats data/flickr30k/features/flickr30k_test_resnet50_cnn_features.hdf5 -output ${MODEL_PATH}/${MODEL_SNAPSHOT}.translations-test2016
#
#perl tools/multi-bleu.perl ${DATA_PATH}/test_2016_flickr.lc.norm.tok.de < ${MODEL_PATH}/${MODEL_SNAPSHOT}.translations-test2016 #> result.query.back.b128.86ep
#
#echo ${MODEL_PATH}/${MODEL_SNAPSHOT}
