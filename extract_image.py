# Derived from https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow/blob/master/make_flickr_dataset.py
# Also from https://github.com/elliottd/satyrid/blob/master/make_dataset.py

import numpy as np
import os
import tables
import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms


from onmt.PretrainedCNNModels import PretrainedCNN


def get_cnn_features(image_list, split, batch_size, dataset_name, pretrained_cnn, pretrained_cnn_name):
    """ Function that does the actual job.

        Creates a hdf5 compressed file, iterates a list of images in minibatches,
        extracts both global and local features for these images and saves these
        features into the hdf5 file.
    """
    # create hdf5 file
    hdf5_path = "%s_%s_%s_%s" % (dataset_name, split, pretrained_cnn_name, "cnn_features.hdf5")
    hdf5_file = tables.open_file(hdf5_path, mode='w')

    # make sure feature sizes are as expected by underlying CNN architectures
    if pretrained_cnn_name.startswith('vgg'):
        global_features_size = 4096
        local_features_size  = 512 * 7 * 7
    else:
        global_features_size = 2048
        local_features_size  = 2048 * 7 * 7

    # use compression in the hdf5 file
    filters = tables.Filters(complevel=5, complib='blosc')
    # create storage for local features
    local_features_storage = hdf5_file.create_earray(hdf5_file.root, 'local_feats',
                                                     tables.Float32Atom(),
                                                     shape=(0, local_features_size),
                                                     filters=filters,
                                                     expectedrows=len(image_list))
    # create storage for global features
    global_features_storage = hdf5_file.create_earray(hdf5_file.root, 'global_feats',
                                                      tables.Float32Atom(),
                                                      shape=(0, global_features_size),
                                                      filters=filters,
                                                      expectedrows=len(image_list))
    # iterate image list in minibatches
    for start, end in zip(range(0, len(image_list)+batch_size, batch_size),
                          range(batch_size, len(image_list)+batch_size, batch_size)):
        if start%200==0:
            print("Processing %s images %d-%d / %d"
                  % (split, start, end, len(image_list)))

        batch_list_fnames = image_list[start:end]
        batch_list = []
        # load/preprocess images for mini-batch
        for entry in batch_list_fnames:
            batch_list.append(
                    pretrained_cnn.load_image_from_path(entry))

        # create minibatch from list of variables
        # i.e., condense the list of image input variables into a mini-batch
        input_imgs_minibatch = torch.cat( batch_list, dim=0 )
        input_imgs_minibatch = input_imgs_minibatch.cuda()
        #print "input_imgs_minibatch.size(): ", input_imgs_minibatch.size()

        # forward pass using pre-trained CNN, twice for each minibatch
        lfeats = pretrained_cnn.get_local_features(input_imgs_minibatch)
        gfeats = pretrained_cnn.get_global_features(input_imgs_minibatch)
        #print("lfeats.size(): ", lfeats.size())
        #print "gfeats.size(): ", gfeats.size()

        # transpose and flatten feats to prepare for reshape
        lfeats = np.array(list(map(lambda x: x.T.flatten(), lfeats.data.cpu().numpy())))
        # flatten feature vector
        gfeats = np.array(list(map(lambda x: x.flatten(), gfeats.data.cpu().numpy())))
        local_features_storage.append(lfeats)
        global_features_storage.append(gfeats)

    print("Finished processing %d images" % len(local_features_storage))
    hdf5_file.close()


def store_image_name(image_list, split, dataset_name):
    # imsize = 64 * (2 ** 2)
    # image_transform = transforms.Compose([
    #     transforms.Scale(int(imsize * 76 / 64)),
    #     transforms.RandomCrop(imsize),
    #     transforms.RandomHorizontalFlip()])
    # norm = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    hdf5_path = "%s_%s_%s" % (dataset_name, split, "image_name.hdf5")
    hdf5_file = tables.open_file(hdf5_path, mode='w')
    image_name_storage = hdf5_file.create_earray(hdf5_file.root,
                                                 'filename',
                                                 atom=tables.StringAtom(itemsize=64),
                                                 shape=(0,),
                                                 expectedrows=len(image_list))
    for fname in image_list:
        image_name = np.array([fname], object)
        image_name_storage.append(image_name)

    print (len(image_name_storage))
    hdf5_file.close()


def store_images(images, split, dataset_name):
    img_64_save_path = "%s_%s_%s" % (dataset_name, split, "images_64.npy")
    img_128_save_path = "%s_%s_%s" % (dataset_name, split, "images_128.npy")

    imgs_64 = [i[0].numpy() for i in images]
    imgs_128 = [i[1].numpy() for i in images]

    np.save(img_64_save_path, imgs_64, allow_pickle=True)
    np.save(img_128_save_path, imgs_128, allow_pickle=True)


def load_fnames_into_dict(fh, split, path_to_images):
    """ Read image file names from a file into a dictionary."""
    data = dict()
    data['files'] = []

    num = 0
    # loop over the data
    for img in fh:
        if 'mscoco' in img:
            idx = img.index('#')
            img_path = img[:idx].strip()
        else:
            img_path = "%s/%s"%(path_to_images,img.strip())
        data['files'].append(img_path)
        num += 1

    print("%s: collected %d images"%(split, len(data['files'])))
    return data

def load_imgs_into_dict(fh, split, path_to_images):
    data = dict()
    data['files'] = []

    num = 0

    for img in fh:
        img_path = "%s/%s" % (path_to_images, img.strip())
        imsize = 64 * (2 ** 2)
        image_transform = transforms.Compose([
            transforms.Scale(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = Image.open(img_path).convert('RGB')
        img = image_transform(img)
        re_img_64 = transforms.Scale(64)(img)
        re_img_128 = transforms.Scale(128)(img)
        re_norm_img_64 = norm(re_img_64)
        re_norm_img_128 = norm(re_img_128)
        re_norm_img_64 = re_norm_img_64.unsqueeze(0)
        re_norm_img_128 = re_norm_img_128.unsqueeze(0)
        imgs = [re_norm_img_64, re_norm_img_128]

        data['files'].append(imgs)
        num += 1
        # if num == 1:
        #     print (img_path)
        #     print (re_norm_img_64)
        #     return data

    print("%s: collected %d images" % (split, len(data['files'])))
    return data

def build_pretrained_cnn(pretrained_cnn_name):
    """ Uses pytorch/cadene to load pre-trained CNN. """
    cnn = PretrainedCNN(pretrained_cnn_name)
    cnn.model = cnn.model.cuda()
    return cnn


def make_dataset(args):
    #cnn = build_pretrained_cnn(args.pretrained_cnn)

    # get the filenames of the images
    data = dict()
    if 'train' in args.splits:
        with open(args.train_fnames, 'r') as fh:
            data['train'] = load_imgs_into_dict(fh, 'train', args.images_path)
            # print (data['train']['files'][0][0])
            # return
    if 'valid' in args.splits:
        with open(args.valid_fnames, 'r') as fh:
            data['valid'] = load_imgs_into_dict(fh, 'valid', args.images_path)

    if 'test' in args.splits:
        with open(args.test_fnames, 'r') as fh:
            data['test'] = load_imgs_into_dict(fh, 'test', args.images_path)
    for split in args.splits:
        #files = ['%s/%s' % (args.images_path, x) for x in data[split]['files']]
        files = data[split]['files']
        #get_cnn_features(files, split, args.batch_size, args.dataset_name, cnn, args.pretrained_cnn)
        #print (files[0][0], files[0][1])
        #print(files[0][0].shape, files[0][1].shape)
        store_images(files, split, args.dataset_name)

    print("Finished!")

def test_dataset():
    path_to_train_img_names = 'flickr30k_train_image_name.hdf5'
    path_to_valid_img_names = 'flickr30k_valid_image_name.hdf5'
    train_file = tables.open_file(path_to_train_img_names, mode='r')
    valid_file = tables.open_file(path_to_valid_img_names, mode='r')

    train_img_names = train_file.root.filename[:]
    valid_img_names = valid_file.root.filename[:]

    # close hdf5 file handlers
    train_file.close()
    valid_file.close()

    # print (type(valid_img_names))
    # print (len(valid_img_names))
    image_names = valid_img_names[[1, 2, 3]]
    imgs = []
    imsize = 64 * (2 ** 2)
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    for fname in image_names:
        img = Image.open(fname).convert('RGB')
        img = image_transform(img)
        re_img = transforms.Scale(128)(img)
        re_norm_img = norm(re_img)
        re_norm_img = re_norm_img.unsqueeze(0)
        imgs.append(re_norm_img)
        print (re_norm_img)
    imgs = torch.cat(imgs, dim=0)
    print (imgs)
    print (imgs.shape)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the dataset bundles to train or test a model (ImgD, ImgE and ImgW).")

    parser.add_argument("--dataset_name", default="flickr30k",
                        help="""Dataset name used to create output files.""")
    parser.add_argument("--splits", default="train,valid,test",
                        help="Comma-separated list of the splits to process")
    # parser.add_argument("--batch_size", type=int, default=20,
    #                     help="Minibatch size for processing images")
    parser.add_argument("--images_path", type=str,
                        help="Path to the directory containing the images",
                        default="multi30K/flickr30k-images")
    # parser.add_argument("--pretrained_cnn", type=str, required=True,
    #                     choices=['resnet50','resnet101','resnet152','fbresnet152','vgg19','vgg19_bn'],
    #                     help="""Name of the pre-trained CNN model available in
    #                     https://github.com/Cadene/pretrained-models.pytorch""")
    parser.add_argument("--train_fnames", type=str,
                        default="multi30K/splits/train_images.txt",
                        help="""File containing a list with training image file names.""")
    parser.add_argument("--valid_fnames", type=str,
                        default="multi30K/splits/val_images.txt",
                        help="""File containing a list with validation image file names.""")
    parser.add_argument("--test_fnames", type=str,
                        default="multi30K/splits/test_images.txt",
                        help="""File containing a list with test image file names.""")
    # parser.add_argument("--gpuid", type=int, r)

    arguments = parser.parse_args()

    # make sure splits are as expected
    splits = arguments.splits.split(",")
    valid_splits = ['train', 'valid', 'test']
    assert(all([s in valid_splits for s in splits])), \
        'One invalid split was found. Valid splits are: %s'%(
                valid_splits)
    arguments.splits = splits

    make_dataset(arguments)
    #test_dataset()