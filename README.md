# 机器学习课程设计：自由手绘草图的识别与生成
本项目实现的是**CS3308-01-机器学习课程设计**中的**单模态任务：自由手绘草图的识别与生成**，由李欣皓、简志辉共同完成。
## 草图识别
1. 将下载的数据集`QuickDraw414k`中的两个文件夹`coordinate_files`, `picture_files`放入项目的`data`文件夹中。\
2. 运行`pip install -r requirements.txt`安装所需的依赖库。\
3. 运行`python train.py --model [模型类别] --epochs [训练轮数] --batch_size [批大小] --lr [学习率]`训练模型。\
