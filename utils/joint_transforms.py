import random
from PIL import Image



class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,img,gt,dp):
        assert (img.size == gt.size and img.size == dp.size)
        for t in self.transforms:
            img,gt,dp = t(img,gt,dp)
        
        return img,gt,dp


class RandomHorizontallyFlip(object):
    def __call__(self,img,gt,dp):
        if random.random() < 0.5 :
            return img.transpose(Image.FLIP_LEFT_RIGHT),gt.transpose(Image.FLIP_LEFT_RIGHT),dp.transpose(Image.FLIP_LEFT_RIGHT)

        return img,gt,dp


class Resize(object):
    def __init__(self,size):
        self.size = tuple(reversed(size))  # size : (h,w)  PIL : (w,h)

    def __call__(self, img , gt, dp):
        assert img.size == gt.size
        return img.resize(self.size,Image.BILINEAR), gt.resize(self.size,Image.NEAREST),dp.resize(self.size,Image.BILINEAR)

