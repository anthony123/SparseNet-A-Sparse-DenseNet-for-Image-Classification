# SparseNet-A-Sparse-DenseNet-for-Image-Classification
毕业论文 “稀疏连接网络：一种用于图像分类的稀疏化稠密网络”有关的代码

共包括三种类型的网络：SparseNet， SparseNet-bc和SparseNet-abc

实验的数据集包括：CIFAR10， CIFAR100， SVHN和Imagenet

每种类型的网络，共有四种不同的参数设置，详情见毕业论文 表3-1 和 表3-2

比如：调用最小的SparseNet进行CIFAR10数据集分类，使用方法为：
python cifar10-sparsenet.py --block1=8  --block2=12 --block=16 --path=14 --gpu=0,1,2,3

更多的实验内容见 第四章和第五章

代码基于的框架为TensorFlow，并使用了tensorpack的接口。

本论文的内容主要基于投的一篇期刊文章，并在此基础上增加了更多的验证实验。其英文版本在[这里](https://pan.baidu.com/s/1B9DD9SrGtu30kJ1E82djGA)

