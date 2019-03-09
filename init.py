import numpy as np
import argparse
from train import *
from test import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        type=str,
                        help='The path to the pretrained cscapes model')

    parser.add_argument('-i', '--image-path',
                        type=str,
                        help='The path to the image to perform semantic segmentation')

    parser.add_argument('-rh', '--resize-height',
                        type=int,
                        default=1024,
                        help='The height for the resized image')

    parser.add_argument('-rw', '--resize-width',
                        type=int,
                        default=2048,
                        help='The width for the resized image')

    parser.add_argument('-lr', '--learning-rate',
                        type=float,
                        default=1e-3,
                        help='The learning rate')

    parser.add_argument('-bs', '--batch-size',
                        type=int,
                        default=2,
                        help='The batch size')

    parser.add_argument('-wd', '--weight-decay',
                        type=float,
                        default=1e-4,
                        help='The weight decay')

    parser.add_argument('-c', '--constant',
                        type=float,
                        default=1.02,
                        help='The constant used for calculating the class weights')

    parser.add_argument('-e', '--epochs',
                        type=int,
                        default=100,
                        help='The number of epochs')

    parser.add_argument('-nc', '--num-classes',
                        type=int,
                        required=True,
                        help='The number of epochs')

    parser.add_argument('-se', '--save-every',
                        type=int,
                        default=10,
                        help='The number of epochs after which to save a model')

    parser.add_argument('-iptr', '--input-path-train',
                        type=str,
                        help='The path to the input dataset')

    parser.add_argument('-lptr', '--label-path-train',
                        type=str,
                        help='The path to the label dataset')

    parser.add_argument('-ipv', '--input-path-val',
                        type=str,
                        help='The path to the input dataset')

    parser.add_argument('-lpv', '--label-path-val',
                        type=str,
                        help='The path to the label dataset')

    parser.add_argument('-iptt', '--input-path-test',
                        type=str,
                        help='The path to the input dataset')

    parser.add_argument('-lptt', '--label-path-test',
                        type=str,
                        help='The path to the label dataset')

    parser.add_argument('-pe', '--print-every',
                        type=int,
                        default=1,
                        help='The number of epochs after which to print the training loss')

    parser.add_argument('-ee', '--eval-every',
                        type=int,
                        default=10,
                        help='The number of epochs after which to print the validation loss')

    parser.add_argument('--cuda',
                        type=bool,
                        default=False,
                        help='Whether to use cuda or not')

    parser.add_argument('--mode',
                        choices=['train', 'test'],
                        default='train',
                        help='Whether to train or test')
    
    parser.add_argument('-dt', '--dtype',
                        choices=['cityscapes', 'pascal'],
                        default='pascal',
                        help='specify the dataset you are using')
    
    parser.add_argument('--scheduler',
                        type=bool,
                        default=False,
                        help='Whether to use scheduler or not')

    parser.add_argument('--save',
                        type=bool,
                        default=True,
                        help='Save the segmented image when predicting')

    FLAGS, unparsed = parser.parse_known_args()

    FLAGS.cuda = torch.device('cuda:0' if torch.cuda.is_available() and FLAGS.cuda \
                               else 'cpu')
    
    print ('[INFO]Arguments read successfully!')

    if FLAGS.mode.lower() == 'train':
        print ('[INFO]Train Mode.')

        if FLAGS.iptr == None or FLAGS.ipv == None:
            raise ('Error: Kindly provide the path to the dataset')

        train(FLAGS)

    elif FLAGS.mode.lower() == 'test':
        print ('[INFO]Predict Mode.')
        predict(FLAGS)
    else:
        raise RuntimeError('Unknown mode passed. \n Mode passed should be either \
                            of "train" or "test"')
