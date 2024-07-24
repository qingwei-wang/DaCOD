





from logging import root
import os
import os.path
import torch.utils.data as data
from PIL import Image


def make_dataset(root):
    image_path = os.path.join(root,'rgb')
    depth_path = os.path.join(root,'depth')
    gt_path = os.path.join(root,'gt')

    #分割对应路径下的文件名和后缀
    img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]  
    
    #返回列表，每个元素是元组，（图片，mask图）
    return [(os.path.join(image_path,img_name +'.jpg'),os.path.join(depth_path,img_name+'.png'),os.path.join(gt_path,img_name+'.png')) for img_name in img_list]   


######## 自定义 图像文件########################
class ImageFolder(data.Dataset):

    def __init__(self,root,joint_transform=None,transform=None,gt_transform=None,dp_transform=None):
        self.root = root
        self.imges = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.gt_transform = gt_transform
        self.dp_transform = dp_transform

    
    def __getitem__(self, index):
        img_path,dep_path,gt_path = self.imges[index]
        img = Image.open(img_path).convert('RGB')
        dp = Image.open(dep_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img,gt,dp = self.joint_transform(img,gt,dp)
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        if self.dp_transform is not None:
            dp = self.dp_transform(dp)

        return img , gt ,dp

    def __len__(self):
        return len(self.imges)

