import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import random

random.seed(78)

#voc数据集class对应的color
def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])
    return cmap

#根据label结合colormap得到原始颜色数据
class label2image():
    def __init__(self, num_classes=21):
        self.colormap = colormap(256)[:num_classes].astype('uint8')

    def __call__(self, label_pred,label_true):
        '''
        :param label_pred: numpy
        :param label_true: numpy
        :return:
        '''
        pred=self.colormap[label_pred]
        true=self.colormap[label_true]
        return pred,true


#将原始label的颜色数据转换成单个数字 [h,w,3]->[h,w]
class image2label():
    '''
    voc classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
    '''
    def __init__(self,num_classes=21):
        self.colormap=colormap(256)[:num_classes]

        cm2lb=np.zeros(256**3)
        for i,cm in enumerate(self.colormap):
            cm2lb[(cm[0]*256+cm[1])*256+cm[2]]=i
        self.cm2lb=cm2lb

    def __call__(self, image):
        '''
        :param image: PIL image
        :return:
        '''
        image=np.array(image,dtype=np.int64)
        idx=(image[:,:,0]*256+image[:,:,1])*256+image[:,:,2]
        label=np.array(self.cm2lb[idx],dtype=np.int64)
        return label

class randomCrop(object):
    """Crop the given PIL Image at a random location.
        自定义实现图像与label随机裁剪相同的位置
    """
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, label):
        i, j, h, w = self.get_params(img, self.size)
        return img.crop((j,i,j+w,i+h)),label.crop((j,i,j+w,i+h))


def train_transform(image,label,crop_size=(256,256)):
    '''
    :param image: PIL image
    :param label: PIL image
    :param crop_size: tuple
    '''

    image,label=randomCrop(crop_size)(image,label)
    tfs=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    image=tfs(image)

    label=image2label()(label)
    label=torch.from_numpy(label).long()
    return image,label

if __name__ == '__main__':
    #测试随机裁剪的效果：image和label是否对应好位置
    image=Image.open('./1.jpg').convert('RGB')
    label=Image.open('./1.png').convert('RGB')
    image, label = randomCrop((128,128))(image, label)
    import matplotlib.pyplot as plt
    image,label=np.array(image),np.array(label)
    fig,ax=plt.subplots(1,2)
    ax[0].imshow(image)
    ax[1].imshow(label)
    plt.show()

    # 测试label相应颜色转换成数字标号的效果
    # label=Image.open('./2.png').convert('RGB')
    # label = image2label()(label)
    # print(label[150:160, 240:250])



