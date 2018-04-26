#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os


from tflearn.layers.conv import global_avg_pool
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.callbacks import *
import cv2
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils import argscope, get_model_loader



def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for top1, top5 in pred.get_result():
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))

class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (224, 224), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(224, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(224).augment(out)
        return out

class Model(ModelDesc):
    def __init__(self,k, path,num_block1, num_block2, num_block3,num_block4):
        super(Model, self).__init__()
        #self.N = int(layers_per_block)
        self.growthRate = int(k)
        self.num_path = int(path)
        self.input_channel = 2*self.growthRate
        self.num_block1 = int(num_block1)
        self.num_block2 = int(num_block2)
        self.num_block3 = int(num_block3)
        self.num_block4 = int(num_block4)

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')
               ]

    
    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        def Relu(x):
            return tf.nn.relu(x)
        def Sigmoid(x) :
            return tf.nn.sigmoid(x)
        def Global_Average_Pooling(x):
            return global_avg_pool(x, name='Global_avg_pooling')
        def Fully_connected(x, units, layer_name='fully_connected') :
            with tf.name_scope(layer_name) :
                return tf.layers.dense(inputs=x, use_bias=False, units=units)
        with tf.name_scope(layer_name) :
            squeeze = Global_Average_Pooling(input_x)

            #excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            #excitation = Relu(excitation)
            #excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            #print("squeeze")
            #print(squeeze.shape)
            squeeze = tf.reshape(squeeze, [-1,1,1,out_dim])
            with tf.variable_scope('bn3'):
                excitation = tf.contrib.layers.batch_norm(squeeze, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None, reuse=None)
            excitation = tf.nn.relu(excitation)
            #c = conv('conv2', c, curr_growthRate, 1)
            excitation = Conv2D('conv3', excitation, out_dim, 1, stride=1, use_bias=True, nl=tf.identity)
            excitation = tf.concat([excitation, squeeze], 3)
            with tf.variable_scope('bn4'):
                excitation = tf.contrib.layers.batch_norm(excitation, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None, reuse=None)
            excitation = tf.nn.relu(excitation)
            #c = conv('conv2', c, curr_growthRate, 1)
            excitation = Conv2D('conv4', excitation, out_dim, 1, stride=1, use_bias=True, nl=tf.identity)

            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            #scale = input_x * excitation
            scale = input_x + input_x * excitation

        return scale
    def _build_graph(self, input_vars):
        image, label = input_vars
        #image = image / 128.0 - 1

        
        
        def conv(name, l, channel, stride, nl=tf.identity):
            return Conv2D(name, l, channel, 7, stride=stride,
                          nl=nl, use_bias=True,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))
        def add_layer(name, l, block_idx):
            with tf.variable_scope(name) as scope:
                shape = l.get_shape().as_list()
                in_channel = shape[3]
                curr_growthRate = int(self.growthRate)
                curr_input_channel = int(self.input_channel)
                curr_num_block = (in_channel-curr_input_channel)/curr_growthRate + 1
                c  = l
                #if block_idx > 0:
                with tf.variable_scope('bn2'):
                    c = tf.contrib.layers.batch_norm(c, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None, reuse=None)
                #c = tf.nn.relu(c)
                #c = conv('conv2', c, curr_growthRate, 1)
                #bc_channel = (curr_num_block+1)//2*curr_growthRate
                #bc_channel = curr_growthRate//2*curr_num_block
                bc_channel = 4*self.growthRate

                # if block_idx == 1:
                #     bc_channel = 12
                # elif block_idx == 2:
                #     bc_channel = 24
                c = Conv2D('conv2', c, bc_channel , 1, stride=1, use_bias=True, nl=tf.identity)


                #c = BatchNorm('bn1', l)
                with tf.variable_scope('bn1'):
                    c = tf.contrib.layers.batch_norm(c, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None, reuse=None)
                c = tf.nn.relu(c)
                c = conv('conv1', c, curr_growthRate, 1)
                reduction_ratio=8
                out_dim = self.growthRate
                c = self.squeeze_excitation_layer(c, out_dim=out_dim, ratio=reduction_ratio, layer_name='squeeze_layer_')
                
                # print('curr_growthRate: %d'  %(curr_growthRate))
                # print('curr_input_channel: %d' %(curr_input_channel))
                # print('in_channel %d'  %(in_channel))
                # print('curr_num_block %d' %((in_channel-curr_input_channel)/curr_growthRate + 1))
                if(in_channel-curr_input_channel)%curr_growthRate != 0:
                    return 
                
                if curr_num_block > self.num_path:
                    split1, _, split2 = tf.split(l, [int(round(curr_num_block/2*curr_growthRate)), int(curr_growthRate), int(in_channel-curr_growthRate -round(curr_num_block/2*curr_growthRate))],3)
                    #split1, _, split2 = tf.split(l, [curr_num_block/2*curr_growthRate, curr_growthRate, in_channel-curr_growthRate -curr_num_block/2*curr_growthRate],3)
                    l = tf.concat([c, split1, split2],3)
                else:
                    l = tf.concat([c, l], 3)
            
            return l

        def add_transition(name, l, idx):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            curr_growthRate = int(self.growthRate)
            curr_input_channel = int(self.input_channel)
            curr_num_block = (in_channel-curr_input_channel)/curr_growthRate + 1
            next_growthRate = int(self.growthRate)
            next_input_channel = int(self.input_channel)
            #out_channel = next_input_channel + (curr_num_block-1)*next_growthRate
            #out_channel = next_input_channel
            #out_channel = in_channel
            # print("next_input_channel: %d"  %(next_input_channel))
            # print("next_growthRate: %d"  %(next_growthRate))
            #print("curr_num_block: %d" %(curr_num_block))
            #print("in_channel: %d" %(in_channel))
            # print("curr_growthRate: %d" %(curr_growthRate))
            # print("curr_input_channel: %d" %(curr_input_channel))
            out_channel=0
            if curr_num_block%2 == 0:
                out_channel = next_input_channel + (curr_num_block)*next_growthRate//2
            else:
                out_channel = next_input_channel + (curr_num_block-1)*next_growthRate//2

            with tf.variable_scope(name) as scope:
                #l = BatchNorm('bn1', l)
                with tf.variable_scope('bn1'):
                    l = tf.contrib.layers.batch_norm(l, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None, reuse=None)
                l = tf.nn.relu(l)
                l = Conv2D('conv1', l, out_channel, 1, stride=1, use_bias=True, nl=tf.identity)
                l = AvgPooling('pool', l, 2)
            return l


        def dense_net(name):
            total = 0
            l = conv('conv0', image, self.input_channel, 2, nl=tf.nn.relu)
            l = AvgPooling('pool', l, 3,2, padding='SAME')
            with tf.variable_scope("blcok1") as scope:
                for i in range(self.num_block1):
                    l = add_layer('dense_layer.{}'.format(i), l, 0)
            l = add_transition("trasition1", l, 0)
            with tf.variable_scope("block2") as scope:
                for i in range(self.num_block2):
                    l = add_layer('dense_layer.{}'.format(i), l, 1)
            l = add_transition("trasition2", l, 1)
            with tf.variable_scope("block3") as scope:
                for i in range(self.num_block3):
                    l = add_layer('dense_layer.{}'.format(i), l, 2)
            l = add_transition("trasition3", l, 1)
            with tf.variable_scope("block4") as scope:
                for i in range(self.num_block4):
                    l = add_layer('dense_layer.{}'.format(i), l, 3)
            #l = BatchNorm('bnlast', l)
            with tf.variable_scope('bnlast'):
                l = tf.contrib.layers.batch_norm(l, decay=0.9, scale=True, is_training = get_current_tower_context().is_training, updates_collections=None, reuse=None)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=1000, nl=tf.identity)

            return logits


        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        logits = dense_net("dense_net")

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        #wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        wd_cost = tf.multiply(1e-4, l2, name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')
        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)



def get_data(train_or_test,batch):
    isTrain = train_or_test == 'train'
    #DIR = '/media/liuwq/hd/dataset/dataset/imagenet/ILSVRC2012'
    DIR = '/home/liuwq'
    ds = dataset.ILSVRC12(DIR, train_or_test)
    #pp_mean = ds.get_per_pixel_mean()
    pc_mean = np.array([0.485, 0.456, 0.406])
    pc_std = np.array([0.229, 0.224, 0.225])
    if isTrain:
        augmentors = [
            GoogleNetResize(),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),

            imgaug.MapImage(lambda x: (x - pc_mean)/(pc_std)),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.MapImage(lambda x: (x - pc_mean)/(pc_std)),
            imgaug.CenterCrop((224, 224)),
            #imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    nr_tower = args.gpu.split(',')
    #BATCH_SIZE = 32
    batch  /= len(nr_tower)
    ds = BatchData(ds, batch, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config():
    log_dir = 'train_log/imagenet-bc-attention-k[%d]-path[%d]-[%d-%d-%d-%d]-' % ( int(args.k), int(args.path), int(args.block1), int(args.block2),int(args.block3), int(args.block4))
    logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    BATCH_SIZE = 64
    dataset_train = get_data('train',BATCH_SIZE)
    
    #dataset_test = get_data('val')
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # config = tf.ConfigProto(allow_soft_placement = True,gpu_options=gpu_options)
    # config.gpu_options.allow_growth=True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    callbacks = []
    callbacks.append(ModelSaver())
    nr_tower = len(args.gpu.split(','))
    print('nr_tower = {}'.format(nr_tower))
    steps_per_epoch = dataset_train.size()//nr_tower
    #steps_per_epoch = 100
    # if nr_tower == 1:
    #         # single-GPU inference with queue prefetch
    #     callbacks.append(InferenceRunner(dataset_test,
    #             [ScalarStats('cost'), ClassificationError()]))
    # else:
    #     # multi-GPU inference (with mandatory queue prefetch)
    #     callbacks.append(DataParallelInferenceRunner(
    #             dataset_test, [ScalarStats('cost'), ClassificationError()], list(range(nr_tower))))
    #     #callbacks.append(InferenceRunner(dataset_test,
                #[ScalarStats('cost',prefix="testing"), ClassificationError(summary_name='validataion_error1')]))

    # callbacks.append(DataParallelInferenceRunner(
    #             dataset_test, [ScalarStats('cost'), ClassificationError()], list(range(nr_tower))))
    callbacks.append(ScheduledHyperParamSetter('learning_rate', [(0, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001),(args.drop_3, 0.0001),(args.drop_4, 0.00001)]))
    return TrainConfig(
        dataflow=dataset_train,
        # callbacks=[
        #     ModelSaver(),
        #     InferenceRunner(dataset_test,
        #         [ScalarStats('cost'), ClassificationError()]),
        #     ScheduledHyperParamSetter('learning_rate',
        #                               [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001),(args.drop_2, 0.0001)])
        # ],
        callbacks=callbacks,
        model=Model(args.k, args.path, args.block1, args.block2, args.block3,args.block4),
        #model=Model(args.k, args.num_block, args.layers_per_block, args.path),
        steps_per_epoch=steps_per_epoch,
        max_epoch=args.max_epoch,
        #session_config = config,
        nr_tower=nr_tower,

    )

if __name__ == '__main__':
    #BATCH_SIZE = 64
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',default='0', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--drop_1',default=30, help='Epoch to drop learning rate to 0.01.') # nargs='*' in multi mode
    parser.add_argument('--drop_2',default=60,help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--drop_3',default=90,help='Epoch to drop learning rate to 0.0001')
    parser.add_argument('--drop_4',default=110,help='Epoch to drop learning rate to 0.00001')
    parser.add_argument('--max_epoch',default=120,help='max epoch')
    parser.add_argument('--k', default=16, help='number of output feature maps for each dense block')
    parser.add_argument('--path', default=14,help='number of paths to each layer')#12
    parser.add_argument('--eval', action='store_true')
    
    parser.add_argument('--block1', default=6, help='number of layers  for the first block') 
    parser.add_argument('--block2', default=8, help='number of layers  for the second block')   
    parser.add_argument('--block3', default=10, help='number of layers  for the third block')   
    parser.add_argument('--block4', default=12,  help='number of layers  for the fourth block')   

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    config = get_config()
    model = config.model
    #config.gpu_options.allow_growth = True
    if args.eval:
        batch=32
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        if args.load:
            config.session_init = SaverRestore(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        #config.nr_tower=max(get_nr_gpu(), 1),
    #SyncMultiGPUTrainer(config).train()
    #SyncMultiGPUTrainerParameterServer(config).train()
        trainer = SyncMultiGPUTrainerParameterServer(len(args.gpu.split(',')))
        launch_train_with_config(config, trainer)

