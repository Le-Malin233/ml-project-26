# 机器学习课程设计：自由手绘草图的识别与生成
本项目实现的是**CS3308-01-机器学习课程设计**中的**单模态任务：自由手绘草图的识别与生成**。\
本项目部分代码参考[TorchSketch](https://github.com/PengBoXiangShang/torchsketch)。
## 草图识别
在草图识别部分，本项目参考[TorchSketch](https://github.com/PengBoXiangShang/torchsketch)中的数据加载方式以及CNN模型和RNN模型实现了草图识别模型的训练。其中，CNN模型基于数据集中`.png`图像文件进行训练，而RNN模型基于数据集中`.npy`序列文件进行训练。
1. 将下载的数据集`QuickDraw414k`中的两个文件夹`coordinate_files`, `picture_files`放入本项目的`data`文件夹中。
2. 运行`pip install -r requirements.txt`安装所需的依赖库。
3. 运行`python train.py --model [模型类别] --epochs [训练轮数] --batch_size [批大小] --lr [学习率]`训练模型。\
   `--model`表示训练所用的模型类别，可选项为`cnn`或`rnn`;\
   `--epochs`表示训练轮数，即整个数据集将被模型学习的次数;\
   `--batch_size`表示批大小，即每次更新模型参数时，一个批次使用的图像数量;\
   `--lr`表示学习率，即控制模型参数更新的步长。
