from __future__ import division
"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import time
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import onmt
import onmt.io
import onmt.modules

from PIL import Image
import torchvision.transforms as transforms

from onmt.Trainer import Statistics
from onmt.Loss import generator_loss, discriminator_loss, KL_loss

class TrainerMultimodal(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            train_img_feats: training global image features.
            valid_img_feats: validation global image features.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1,
                 train_img_feats=None, valid_img_feats=None,
                 multimodal_model_type=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.train_img_feats = train_img_feats
        self.valid_img_feats = valid_img_feats
        self.multimodal_model_type = multimodal_model_type

        assert(not self.train_img_feats is None), \
                'Must provide training image features!'
        assert(not self.valid_img_feats is None), \
                'Must provide validation image features!'
        assert(self.multimodal_model_type in ['imgw', 'imge', 'imgd', 'src+img', 'graphtransformer', 'gantransformer']), \
                'Invalid multimodal model type: %s!'%(self.multimodal_model_type)

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                normalization += batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            total_stats.start_time, self.optim.lr,
                            report_stats)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # extract indices for all entries in the mini-batch
            idxs = batch.indices.cpu().data.numpy()
            # load image features for this minibatch into a pytorch Variable
            img_feats = torch.from_numpy( self.valid_img_feats[idxs] )
            img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
            if next(self.model.parameters()).is_cuda:
                img_feats = img_feats.cuda()
            else:
                img_feats = img_feats.cpu()

            # F-prop through the model.
            if self.multimodal_model_type == 'src+img':
                outputs, outputs_img, attns, _ = self.model(src, tgt, src_lengths, img_feats)
            elif self.multimodal_model_type in ['imgw', 'imge', 'imgd', 'graphtransformer']:
                outputs, attns, _ = self.model(src, tgt, src_lengths, img_feats)
            else:
                raise Exception("Multimodal model type not yet supported: %s"%(
                        self.multimodal_model_type))

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            # extract indices for all entries in the mini-batch
            # print('111')
            idxs = batch.indices.cpu().data.numpy()
            # load image features for this minibatch into a pytorch Variable
            img_feats = torch.from_numpy( self.train_img_feats[idxs] )
            img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
            if next(self.model.parameters()).is_cuda:
                img_feats = img_feats.cuda()
            else:
                img_feats = img_feats.cpu()

            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            else:
                src_lengths = None

            tgt_outer = onmt.io.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.grad_accum_count == 1:
                    self.model.zero_grad()
                if self.multimodal_model_type == 'src+img':
                    outputs, outputs_img, attns, dec_state = \
                        self.model(src, tgt, src_lengths, img_feats, dec_state)
                elif self.multimodal_model_type in ['imgw', 'imge', 'imgd', 'graphtransformer']:
                    outputs, attns, dec_state = \
                        self.model(src, tgt, src_lengths, img_feats, dec_state)
                else:
                    raise Exception("Multimodal model type not yet supported: %s"%(
                            self.multimodal_model_type))

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization)

                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        import torch.optim as ganoptim
        for i in range(num_Ds):
            opt = ganoptim.Adam(netsD[i].parameters(),
                             lr=0.0002,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = ganoptim.Adam(netG.parameters(),
                                lr=0.0002,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD


class TrainerMultimodalGAN(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train

            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            train_img_feats: training global image features.
            valid_img_feats: validation global image features.
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1,
                 train_img_names=None, valid_img_names=None,
                 multimodal_model_type=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.train_img_names = train_img_names
        self.valid_img_names = valid_img_names
        self.multimodal_model_type = multimodal_model_type
        self.imsize = 64 * (2 ** 2)
        self.image_transform = transforms.Compose([
            transforms.Scale(int(self.imsize * 76 / 64)),
            transforms.RandomCrop(self.imsize),
            transforms.RandomHorizontalFlip()])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        assert(not self.train_img_names is None), \
                'Must provide training image names!'
        assert(not self.valid_img_names is None), \
                'Must provide validation image names!'
        assert(self.multimodal_model_type in ['imgw', 'imge', 'imgd', 'src+img', 'graphtransformer', 'gantransformer']), \
                'Invalid multimodal model type: %s!'%(self.multimodal_model_type)

        assert(grad_accum_count > 0)
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter, epoch, report_func=None):
        """ Train next epoch.
        Args:
            train_iter: training data iterator
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()
        idx = 0
        true_batchs = []
        accum = 0
        normalization = 0
        try:
            add_on = 0
            if len(train_iter) % self.grad_accum_count > 0:
                add_on += 1
            num_batches = len(train_iter) / self.grad_accum_count + add_on
        except NotImplementedError:
            # Dynamic batching
            num_batches = -1

        for i, batch in enumerate(train_iter):
            cur_dataset = train_iter.get_cur_dataset()
            self.train_loss.cur_dataset = cur_dataset

            true_batchs.append(batch)
            accum += 1
            if self.norm_method == "tokens":
                normalization += batch.tgt[1:].data.view(-1) \
                    .ne(self.train_loss.padding_idx).sum()
            else:
                normalization += batch.batch_size

            if accum == self.grad_accum_count:
                self._gradient_accumulation(
                        true_batchs, total_stats,
                        report_stats, normalization)

                if report_func is not None:
                    report_stats = report_func(
                            epoch, idx, num_batches,
                            total_stats.start_time, self.optim.lr,
                            report_stats)

                true_batchs = []
                accum = 0
                normalization = 0
                idx += 1

        if len(true_batchs) > 0:
            self._gradient_accumulation(
                    true_batchs, total_stats,
                    report_stats, normalization)
            true_batchs = []

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        for batch in valid_iter:
            cur_dataset = valid_iter.get_cur_dataset()
            self.valid_loss.cur_dataset = cur_dataset

            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
            else:
                src_lengths = None

            tgt = onmt.io.make_features(batch, 'tgt')

            # extract indices for all entries in the mini-batch
            idxs = batch.indices.cpu().data.numpy()
            # load image features for this minibatch into a pytorch Variable
            img_feats = torch.from_numpy( self.valid_img_feats[idxs] )
            img_feats = torch.autograd.Variable(img_feats, requires_grad=False)
            if next(self.model.parameters()).is_cuda:
                img_feats = img_feats.cuda()
            else:
                img_feats = img_feats.cpu()

            # F-prop through the model.
            if self.multimodal_model_type == 'src+img':
                outputs, outputs_img, attns, _ = self.model(src, tgt, src_lengths, img_feats)
            elif self.multimodal_model_type in ['imgw', 'imge', 'imgd', 'graphtransformer']:
                outputs, attns, _ = self.model(src, tgt, src_lengths, img_feats)
            else:
                raise Exception("Multimodal model type not yet supported: %s"%(
                        self.multimodal_model_type))

            # Compute loss.
            batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats

    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)

    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))

    def _gradient_accumulation(self, true_batchs, total_stats,
                               report_stats, normalization):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            # extract indices for all entries in the mini-batch
            # print('111')
            idxs = batch.indices.cpu().data.numpy()
            # load image features for this minibatch into a pytorch Variable
            img_names = self.train_img_names[idxs]
            #print (img_names)
            imgs = []
            tmp_64 = []
            tmp_128 = []
            for name in img_names:
                img = Image.open(name).convert('RGB')
                img = self.image_transform(img)
                re_img_64 = transforms.Scale(64)(img)
                re_img_128 = transforms.Scale(128)(img)
                re_norm_img_64 = self.norm(re_img_64)
                re_norm_img_128 = self.norm(re_img_128)
                re_norm_img_64 = re_norm_img_64.unsqueeze(0)
                re_norm_img_128 = re_norm_img_128.unsqueeze(0)
                tmp_64.append(re_norm_img_64)
                tmp_128.append(re_norm_img_128)
            tmp_64 = torch.cat(tmp_64, dim=0)
            tmp_64 = tmp_64.cuda()
            imgs.append(tmp_64)
            tmp_128 = torch.cat(tmp_128, dim=0)
            tmp_128 = tmp_128.cuda()
            imgs.append(tmp_128)

            target_size = batch.tgt.size(0)
            # Truncated BPTT
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)
            if self.data_type == 'text':
                _, src_lengths = batch.src
                report_stats.n_src_words += src_lengths.sum()
            else:
                src_lengths = None

            tgt_outer = onmt.io.make_features(batch, 'tgt')

            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.multimodal_model_type == 'gantransformer':
                    #src [length, batch_size, 1]
                    batch_size = 128
                    nz = 100
                    noise = Variable(torch.FloatTensor(batch_size, nz))
                    fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
                    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
                    noise.data.normal_(0, 1)

                    self.model.netG.zero_grad()

                    context, mu, logvar, fake_imgs, outputs, attns, dec_state = self.model(src, tgt, src_lengths, noise, dec_state)

                    context_ = context.detach()
                else:
                    raise Exception("Multimodal model type not yet supported: %s"%(
                            self.multimodal_model_type))

                # 3. Compute loss in shards for memory efficiency.
                batch_stats = self.train_loss.sharded_compute_loss(
                        batch, outputs, attns, j,
                        trunc_size, self.shard_size, normalization)


                # 4. compute netG and netD loss
                if self.multimodal_model_type == 'gantransformer':
                    '''
                        src: [s_length, batch_size, 1]
                        context: [s_length, batch_size, emb_dim]
                        img_feats: [batch_size, 100352]
                        words_embs: [batch, emb_dim, length]
                        sent_emb: [batch, emb_dim]
                        mask: [batch, length]
                    '''
                    optimizerG, optimizersD = self.define_optimizers(self.model.netG, self.model.netsD)
                    # batch_size = context.shape[1]
                    # nz = 100
                    # noise = Variable(torch.FloatTensor(batch_size, nz))
                    # fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
                    # noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
                    words_embs = context_.permute(1,2,0).contiguous()
                    sent_emb = torch.mean(words_embs, dim=2)
                    #print (words_embs.shape, sent_emb.shape)

                    # noise.data.normal_(0, 1)
                    fake_imgs, _, mu, logvar = self.model.netG(noise, sent_emb, words_embs, mask=None)
                    #print (fake_imgs[0].shape)  #[128, 3, 64, 64]

                    #update netD
                    errD_total = 0
                    D_logs = ''
                    real_labels = torch.ones(batch_size).cuda()
                    fake_labels = torch.zeros(batch_size).cuda()
                    for i in range(len(self.model.netsD)):
                        self.model.netsD[i].zero_grad()

                        #print(imgs[i].shape, fake_imgs[i].shape, sent_emb.shape)

                        errD = discriminator_loss(self.model.netsD[i], imgs[i], fake_imgs[i].detach(), sent_emb, real_labels, fake_labels)
                        errD.backward()
                        optimizersD[i].step()
                        errD_total += errD
                        D_logs += 'errD%d: %.2f ' % (i, errD.item())
                        #print (errD_total)

                    # self.model.netG.zero_grad()
                    errG_total, G_logs = generator_loss(netsD=self.model.netsD, image_encoder=0, caption_cnn=self.model.caption_cnn, caption_rnn=self.model.caption_rnn, captions=src, fake_imgs=fake_imgs, real_labels=real_labels, words_embs=words_embs, sent_emb=sent_emb, match_labels=0, cap_lens=src_lengths, class_ids=0)
                    #print (errG_total)
                    kl_loss = KL_loss(mu, logvar)
                    errG_total += kl_loss
                    G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                    # backward and update parameters
                    errG_total.backward()
                    #optimizerG.step()


                # 4. Update the parameters and statistics.
                if self.grad_accum_count == 1:
                    self.optim.step()
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

                # If truncated, don't backprop fully.
                if dec_state is not None:
                    dec_state.detach()

        if self.grad_accum_count > 1:
            self.optim.step()

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        import torch.optim as ganoptim
        for i in range(num_Ds):
            opt = ganoptim.Adam(netsD[i].parameters(),
                             lr=0.0002,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = ganoptim.Adam(netG.parameters(),
                                lr=0.0002,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD