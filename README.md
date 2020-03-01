## 中国海洋大学视觉实验室前沿理论小组 PyTorch 学习

（内容将不断更新）

为方便大家理解代码，我编写了这些教程，并保持不停更新。为方便大家理解算法及代码原理，不浪费时间在配置环境上，下面代码均能顺利在 Google 云端 GPU 平台在线运行，并能够非常快速的运行出结果。

大家注意，列表里的所有代码上方都有一个“Open in Colab”的图标，点击以后就可以直接在 Google Colab 平台打开，也可以在平台直接运行。需要使用GPU的话，可以在Colab的菜单 "代码执行工具"  → "更改运行时类型" 里进行设置。

代码都浅显易懂，添加了大量的备注及说明，有不明白的地方，可以调试中间结果观察数据的变化情况。不明白怎么调试，可以随时联系我。



🟢 01  【[ Python图像处理基础 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/01_Image_Processing.ipynb)】

🟢 02  【[ PyTorch基础 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/02_Pytorch_Basic.ipynb)】

🟢 03  【[ 写一个简单的网络解决Spiral classifciation问题 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/03_Spiral_Classification.ipynb)】

🟢 04  【[ 写一个2层的网络解决回归问题 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/04_Regression.ipynb)】

🟢 05_01  【[ 写一个LeNet应用于MNIST分类 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/05_01_ConvNet.ipynb)】分类准确率轻松达到96%，下一课换稍难些的CIFAR10

🟢 05_02  【[ 写一个LeNet应用于CIFAR10分类 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/05_02_CNN_CIFAR10.ipynb)】CIFAR10数据集相对较难，分类准确率只有64%，下一课换更好的VGG网络

🟢 05_03  【[ 写一个VGG应用于CIFAR10分类 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/05_02_CNN_CIFAR10.ipynb)】使用VGG网络，准确率提升至84.92 %，下一课我们适当缓冲下，学习下VGG在迁移学习中的应用

🟢 05_04  【[ VGG迁移学习进行猫狗大战 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/05_04_Transfer_VGG_for_dogs_vs_cats.ipynb)】海量高分辨率图像的训练，比较玄学，很难得到一个好的网络。这里我们学习在pretrained VGG网络上 fine-tune，分类猫狗图片

🟢 05_05  【[ 写一个GoogLeNet应用于CIFAR10分类 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/05_05_GoogLeNet_CIFAR10.ipynb)】手写InceptionV3，准确率提升至85.40%

🟢 05_06  【[ 写一个ResNet18应用于CIFAR10分类 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/05_06_Resnet_CIFAR10.ipynb)】手写ResNet18，准确率提升至85.96%

🟢 06  【[ 自编码器与降噪自编码器 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/06_Autoencoder.ipynb)】这节课学习用自编码器重建MNIST，同时观察加入denoising后，自编码器在MNIST重建上的性能变化

🟢 07  【[ 用变分自编码器生成数字 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/07_VAE.ipynb)】输入随机噪声，生成数字图像

🟢 08  【[ 写简单的GAN网络生成double moon 数据 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/08_GAN_double_moon.ipynb)】

🟢 09  【[ CGAN和DCGAN在mnist上的应用 ](https://github.com/OUCTheoryGroup/colab_demo/blob/master/09_CGAN_DCGAN_mnist.ipynb)】

<br><br>

此外，我还整理了一些典型论文代码的 pytorch 实现，添加解释说明和备注放在 colab 平台上了，供大家学习，具体如下：

✅ 【[基于 PCA 和 k-means 的遥感图像变化检测](https://github.com/OUCTheoryGroup/colab_demo/blob/master/Change_detection_PCA_KM.ipynb)】 IEEE GRSL 2009

✅ 【[代码短小精悍的无监督图像分割](https://github.com/OUCTheoryGroup/colab_demo/blob/master/Unsupervised_Segmentation.ipynb)】 ICASSP 2018

✅ 【[ Dual Path Networks应于CIFAR10分类 ]()】 NIPS 2017   【3月初完成】

✅  【[ Squeeze-and-Excitation Net应用于CIFAR10分类 ]()】CVPR 2018  【3月初完成】

✅  【[ MobileNetV2应用于CIFAR10分类 ]()】CVPR 2018  【3月初完成】

✅  【[ Progressive Neural Architecture Search应用于CIFAR10分类 ]()】ECCV 2018  【3月初完成】

✅  【[ EfficientNet应用于CIFAR10分类 ]()】ICML 2019  【3月初完成】

<br><br>

联系方式：gaofeng@ouc.edu.cn