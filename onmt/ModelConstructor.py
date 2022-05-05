"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Models import NMTModel, MeanEncoder, RNNEncoder, \
                        StdRNNDecoder, InputFeedRNNDecoder
from onmt.modules import Embeddings, ImageEncoder, CopyGenerator, \
                         TransformerEncoder, mmTransformerEncoder, TransformerDecoder, mmTransformerEncoderLayer,\
                         CNNEncoder, CNNDecoder, AudioEncoder
from onmt.Utils import use_gpu

# additional imports for multi-modal NMT
from onmt.Models import ImageGlobalFeaturesProjector, \
                        ImageLocalFeaturesProjector, \
                        FakeImageLocalFeaturesProjector, \
                        StdRNNDecoderDoublyAttentive, \
                        InputFeedRNNDecoderDoublyAttentive, \
                        NMTImgDModel, NMTImgEModel, NMTImgWModel, \
                        NMTSrcImgModel, RNNEncoderImageAsWord, GraphTransformer, GANTransformer
#from onmt.GAN import G_DCGAN, G_NET, CNN_ENCODER, CAPTION_CNN, CAPTION_RNN
from onmt.GAN import CNN_ENCODER, CAPTION_CNN, CAPTION_RNN

import pretrainedmodels
import pretrainedmodels.utils

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    print(opt.pre_word_vecs_enc )
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
        if opt.pre_word_vecs_enc is not None:
            embedding_dim = 300
    else:
        embedding_dim = opt.tgt_word_vec_size
        if opt.pre_word_vecs_dec is not None:
            embedding_dim = 300

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[onmt.io.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings)


def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "mmtransformer":
        return mmTransformerEncoder(opt.enc_layers, opt.rnn_size,
                                  opt.dropout, embeddings)
    elif opt.encoder_type == "cnn":
        return CNNEncoder(opt.enc_layers, opt.rnn_size,
                          opt.cnn_kernel_width,
                          opt.dropout, embeddings)
    elif opt.encoder_type == "mean":
        return MeanEncoder(opt.enc_layers, embeddings)
    else:
        # "rnn" or "brnn"
        return RNNEncoder(opt.rnn_type, opt.brnn, opt.enc_layers,
                          opt.rnn_size, opt.dropout, embeddings)


def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.decoder_type == "transformer":
        return TransformerDecoder(opt.dec_layers, opt.rnn_size,
                                  opt.global_attention, opt.copy_attn,
                                  opt.dropout, embeddings)
    elif opt.decoder_type == "cnn":
        return CNNDecoder(opt.dec_layers, opt.rnn_size,
                          opt.global_attention, opt.copy_attn,
                          opt.cnn_kernel_width, opt.dropout,
                          embeddings)
    elif opt.decoder_type == "doubly-attentive-rnn" and not opt.input_feed:
        return StdRNNDecoderDoublyAttentive(opt.rnn_type,
                             opt.brnn,
                             opt.dec_layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings)
    elif opt.decoder_type == "doubly-attentive-rnn" and opt.input_feed:
        return InputFeedRNNDecoderDoublyAttentive(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn)
    elif opt.input_feed:
        return InputFeedRNNDecoder(opt.rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings,
                                   opt.reuse_copy_attn)
    else:
        return StdRNNDecoder(opt.rnn_type, opt.brnn,
                             opt.dec_layers, opt.rnn_size,
                             opt.global_attention,
                             opt.coverage_attn,
                             opt.context_gate,
                             opt.copy_attn,
                             opt.dropout,
                             embeddings,
                             opt.reuse_copy_attn)


def load_test_model(opt, dummy_opt):
    checkpoint = torch.load(opt.model,
                            map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    if 'multimodal_model_type' in opt:
        print( 'Building multi-modal model...' )
        model = make_base_model_mmt(model_opt, fields,
                                    use_gpu(opt), checkpoint)
    else:
        print( 'Building text-only model...' )
        model = make_base_model(model_opt, fields,
                                use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts)
        encoder = make_encoder(model_opt, src_embeddings)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    model = NMTModel(encoder, decoder)
    model.model_type = model_opt.model_type

    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def make_base_model_mmt(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the Multimodal NMT model.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        #print (len(src_dict))
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts)
        #encoder = make_encoder(model_opt, src_embeddings)
        if model_opt.multimodal_model_type in ['imgd', 'imge', 'src+img', 'graphtransformer']:
            encoder = make_encoder(model_opt, src_embeddings)
        elif  model_opt.multimodal_model_type == 'imgw':
            # model ImgW uses a specific source-language encoder
            encoder = RNNEncoderImageAsWord(model_opt.rnn_type,
                    model_opt.brnn, model_opt.enc_layers,
                    model_opt.rnn_size, model_opt.dropout, src_embeddings)
        else:
            raise Exception("Multi-modal model type not implemented: %s"%
                            model_opt.multimodal_model_type)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)

    if model_opt.multimodal_model_type in ['src+img', 'graphtransformer']:
        # use the local image features "as is": encoder only reshapes them
        encoder_image = make_encoder_image_local_features(model_opt)
    else:
        # transform global image features before using them
        encoder_image = make_encoder_image_global_features(model_opt)

    # Make NMTModel(= encoder + decoder).
    #model = NMTModel(encoder, decoder)
    #model.model_type = model_opt.model_type
    if model_opt.multimodal_model_type == 'imgd':
        model = NMTImgDModel(encoder, decoder, encoder_image)
    elif model_opt.multimodal_model_type == 'imge':
        model = NMTImgEModel(encoder, decoder, encoder_image)
    elif model_opt.multimodal_model_type == 'imgw':
        model = NMTImgWModel(encoder, decoder, encoder_image)
    elif model_opt.multimodal_model_type == 'src+img':
        # using image encoder only to reshape local features
        model = NMTSrcImgModel(encoder, decoder, encoder_image)
    elif model_opt.multimodal_model_type == 'graphtransformer':
        # using image encoder only to reshape local features
        model = GraphTransformer(encoder, decoder, encoder_image)
    else:
        raise Exception("Multi-modal model type not yet implemented: %s"%(
                        opt.multimodal_model_type))

    model.model_type = model_opt.model_type

    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Initializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def make_base_model_mmt_gan(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the Multimodal NMT model.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        ("Unsupported model type %s" % (model_opt.model_type))

    # Make encoder.
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
        src_embeddings = make_embeddings(model_opt, src_dict,
                                         feature_dicts)
        #encoder = make_encoder(model_opt, src_embeddings)
        if model_opt.multimodal_model_type in ['imgd', 'imge', 'src+img', 'graphtransformer', 'gantransformer']:
            encoder = make_encoder(model_opt, src_embeddings)
        elif  model_opt.multimodal_model_type == 'imgw':
            # model ImgW uses a specific source-language encoder
            encoder = RNNEncoderImageAsWord(model_opt.rnn_type,
                    model_opt.brnn, model_opt.enc_layers,
                    model_opt.rnn_size, model_opt.dropout, src_embeddings)
        else:
            raise Exception("Multi-modal model type not implemented: %s"%
                            model_opt.multimodal_model_type)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    # Image CNN Encoder
    cnn_model_name = 'resnet50'
    cnn_model = pretrainedmodels.__dict__[cnn_model_name](num_classes=1000, pretrained='imagenet')
    cnn_model.last_linear = pretrainedmodels.utils.Identity()
    for p in cnn_model.parameters():
        p.requires_grad = False
    cnn_model.eval()
    # image_encoder = CNN_ENCODER (model_opt.rnn_size)
    #img_encoder_path = model_opt.NET_E
    #state_dict = \
    #    torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    #image_encoder.load_state_dict(state_dict)
    #for p in image_encoder.parameters():
    #    p.requires_grad = False
    #print('Load image encoder from:', img_encoder_path)
    #image_encoder.eval()

    # Caption models
    # caption_cnn = CAPTION_CNN(model_opt.CAP_embed_size)
    # caption_cnn.load_state_dict(torch.load(model_opt.CAP_cnn_path, map_location=lambda storage, loc: storage))
    # for p in caption_cnn.parameters():
    #     p.requires_grad = False
    # print('Load caption model from:', model_opt.CAP_cnn_path)
    # caption_cnn.eval()
    #
    # caption_rnn = CAPTION_RNN(model_opt.CAP_embed_size, model_opt.CAP_hidden_size * 2, len(src_dict), model_opt.CAP_num_layers)
    # caption_rnn.load_state_dict(torch.load(cfg.CAP.caption_rnn_path, map_location=lambda storage, loc: storage))
    # for p in caption_rnn.parameters():
    #     p.requires_grad = False
    # print('Load caption model from:', model_opt.CAP_rnn_path)

    # Generator and Discriminator
    from onmt.GAN import G_NET, D_NET64, D_NET128, D_NET256
    netG = G_NET(BRANCH_NUM=model_opt.branch_num)
    #print (netG)
    netsD = []
    if model_opt.branch_num > 0:
        netsD.append(D_NET64())
    if model_opt.branch_num > 1:
        netsD.append(D_NET128())
    if model_opt.branch_num > 2:
        netsD.append(D_NET256())
    netG.apply(weights_init)
    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        # print(netsD[i])
    print('# of netsD', len(netsD))


    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict,
                                     feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)

    encoder_image = make_encoder_fake_image_local_features(model_opt)
    # if model_opt.multimodal_model_type in ['src+img', 'graphtransformer']:
    #     # use the local image features "as is": encoder only reshapes them
    #     encoder_image = make_encoder_image_local_features(model_opt)
    # else:
    #     # transform global image features before using them
    #     encoder_image = make_encoder_image_global_features(model_opt)

    # Make NMTModel(= encoder + decoder).
    #model = NMTModel(encoder, decoder)
    #model.model_type = model_opt.model_type
    # if model_opt.multimodal_model_type == 'imgd':
    #     model = NMTImgDModel(encoder, decoder, encoder_image)
    # elif model_opt.multimodal_model_type == 'imge':
    #     model = NMTImgEModel(encoder, decoder, encoder_image)
    # elif model_opt.multimodal_model_type == 'imgw':
    #     model = NMTImgWModel(encoder, decoder, encoder_image)
    # elif model_opt.multimodal_model_type == 'src+img':
    #     # using image encoder only to reshape local features
    #     model = NMTSrcImgModel(encoder, decoder, encoder_image)
    # elif model_opt.multimodal_model_type == 'graphtransformer':
    #     # using image encoder only to reshape local features
    #     model = GraphTransformer(encoder, decoder, encoder_image)

    if model_opt.multimodal_model_type == 'gantransformer':
        mmTransformerEncoder = mmTransformerEncoderLayer(model_opt.rnn_size, model_opt.dropout)
        model = GANTransformer(encoder, mmTransformerEncoder, decoder, netG, netsD, cnn_model, encoder_image)
    else:
        raise Exception("Multi-modal model type not yet implemented: %s"%(
                        model_opt.multimodal_model_type))

    model.model_type = model_opt.model_type

    # Make Generator.
    if not model_opt.copy_attn:
        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, len(fields["tgt"].vocab)),
            nn.LogSoftmax())
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.rnn_size,
                                  fields["tgt"].vocab)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
    else:
        if model_opt.param_init != 0.0:
            print('Initializing model parameters.')
            for p in model.encoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in model.decoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in model.mm_encoder.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in model.encoder_images.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                    model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
        for i in range(len(model.netsD)):
            model.netsD[i].cuda()
    else:
        model.cpu()

    return model






def make_encoder_image_global_features(opt):
    """
    Global image features encoder dispatcher function(s).
    Args:
        opt: the option in current environment.
    """
    # TODO: feat_size and num_layers only tested with vgg and resnet networks.
    # Validate that these values work for other CNN architectures as well.
    if 'vgg' in opt.path_to_train_img_feats.lower():
        feat_size = 4096
    else:
        feat_size = 2048

    if opt.multimodal_model_type == 'imgw':
        num_layers = 2
    elif opt.multimodal_model_type == 'imge':
        num_layers = opt.enc_layers
    elif opt.multimodal_model_type == 'imgd':
        num_layers = opt.dec_layers
    return ImageGlobalFeaturesProjector(num_layers, feat_size, opt.rnn_size,
            opt.dropout_imgs, opt.use_nonlinear_projection)

def make_encoder_image_local_features(opt):
    """
    Local image features encoder dispatcher function(s).
    Args:
        opt: the option in current environment.
    """
    # TODO: feat_size and num_layers only tested with vgg network.
    # Validate that these values work for other CNN architectures as well.
    if 'vgg' in opt.path_to_train_img_feats.lower():
        feat_size = 512
    else:
        feat_size = 2048
    num_layers = 1
    return ImageLocalFeaturesProjector(num_layers, feat_size, opt.rnn_size,
            opt.dropout_imgs, opt.use_nonlinear_projection)

def make_encoder_fake_image_local_features(opt):
    feat_size = 2048
    return FakeImageLocalFeaturesProjector(feat_size, opt.rnn_size, opt.dropout_imgs, opt.use_nonlinear_projection)
