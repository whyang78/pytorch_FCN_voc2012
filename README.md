# FCN  
1. 使用voc2012数据集进行分割实例训练测试。数据集建立详见 `utils/dataset.py`  
    ```
    mkdir data
    mv VOCdevkit/VOC2012/JPEGImages data/images
    mv VOCdevkit/VOC2012/SegmentationClass data/classes
    ```
    在建立数据集时，可以采用直接读取文件夹里的图像建立数据集，然后再自行划分训练集与验证集。  
    也可以参照 `\VOCdevkit\VOC2012\ImageSets\Segmentation\train.txt(val.txt)`，本实例中采用前者。  
2. FCN网络用于图像分割，模型搭建详见 `utils/model.py`  
    + FCN8x / FCN16x / FCN32x 使用vgg16作为backbone。  
    + FCN8s 使用resnet34作为backbone。 
    + 使用双线性卷积核初始化反卷积核参数。   
    + 网络反卷积部分设计中可以添加一些batchnorm层，relu或者leakyrelu层等。 
3. voc数据集分割类别对应有相应的颜色，故要对label进行一定的变换。详见 `utils/transfrom.py`  
    ![类别与颜色对应图](./src/0.png)
4. 数据预处理中并没有对image和label进行resize，是因为label是像素级别上的标注，进行resize可能  
    会变成小数。故对其进行相同位置的random crop，自定义random crop。详见 `utils/transfrom.py`  
5. 图像分割衡量标准，具体实现见`utils/eval.py`。  
    参考网站：https://blog.csdn.net/majinlei121/article/details/78965435  
             https://blog.csdn.net/u014593748/article/details/71698246  
6. 目录结构  
    ![目录](./src/1.png)
