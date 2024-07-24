import datetime
import time
import os

import torch
from torch import  nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F



import utils.joint_transforms as joint_transforms 
from data.dataset import ImageFolder
from utils.config import depth_cod_training_root
from utils.config import backbone_path
from utils.misc import check_mkdir ,AvgMeter

from models.Depth_cod import De_cod

import utils.loss as loss
cudnn.benchmark = True  
torch.manual_seed(2021)  
device_ids = [0]         

ckpt_path = './checkpoints'     
ckpt_path2 = '/root/tf-logs/new2'     
exp_name = 'Depth_cod'

args = {
    'epoch_num': 60,                                    
    'train_batch_size': 10,                             
    'last_epoch': 0,                                    
    'lr': 1e-3,                                         
    'lr_decay': 0.9,                                    
    'weight_decay': 5e-4,                               
    'momentum': 0.9,                                    
    'snapshot': '',
    'scale': 448,                                       
    'save_point': [],
    'poly_train': True,                                 
    'optimizer': 'SGD',                                 
}

print(torch.__version__)       

#路径Path设置
check_mkdir(ckpt_path)                               #检查路径是否存在  没有就创建
check_mkdir(os.path.join(ckpt_path,exp_name))        #同上
vis_path = os.path.join(ckpt_path2,exp_name,'log')    #拼接日志文件夹的路径
check_mkdir(vis_path)                                #检查日志文件夹路径是否存在   没有就创建
log_path = os.path.join(ckpt_path,exp_name,str(datetime.datetime.now())+'.txt')

log_path = log_path.replace('-','_')
log_path = log_path.replace(':','_')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)



#Transform Data   对图像进行预处理
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),                      
    joint_transforms.Resize((args['scale'],args['scale']))         
])

img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),          
    transforms.ToTensor(),                                                                  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                     
])

gt_transform = transforms.ToTensor()                      

dp_transform = transforms.ToTensor()  



train_set = ImageFolder(depth_cod_training_root,joint_transform,img_transform,gt_transform,dp_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)


total_epoch = args['epoch_num'] * len(train_loader)



structure_loss = loss.structure_loss().cuda(device_ids[0])
bce_loss = nn.BCEWithLogitsLoss().cuda(device_ids[0])
iou_loss = loss.IOU().cuda(device_ids[0])
ce_loss = nn.CrossEntropyLoss().cuda(device_ids[0])




def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss


def cal_ual(pred):
    sigmoid_x = pred.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()

def flat(mask):
    batch_size = mask.shape[0]
    h = 28
    mask = F.interpolate(mask,size=(int(h),int(h)), mode='bilinear')
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1) 
   
    g = x @ x.transpose(-2,-1) 
    g = g.unsqueeze(1) 
    return g


def att_loss(pred,mask,p4,p5):
    g = flat(mask)
    np4 = torch.sigmoid(p4.detach())
    np5 = torch.sigmoid(p5.detach())
    p4 = flat(np4)
    p5 = flat(np5)
    w1  = torch.abs(g-p4)
    w2  = torch.abs(g-p5)
    w = (w1+w2)*0.5+1
    attbce=F.binary_cross_entropy_with_logits(pred, g,weight =w*1.0,reduction='mean')
    return attbce

def bce_iou_loss2(pred, mask):
    size = pred.size()[2:]
    mask = F.interpolate(mask,size=size, mode='bilinear')
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(net,optimizer):
    curr_iter = 1 
    start_time = time.time()

    
    for epoch in range(args['last_epoch']+1,args['last_epoch']+1+args['epoch_num']):

       
        loss_record,loss_1_record,loss_2_record,loss_3_record,loss_4_record,loss_5_record= AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter(),AvgMeter()
       
        train_iterator = tqdm(train_loader,total = len(train_loader))

        for data in train_iterator:
            
            if args['poly_train']:
                base_lr = args['lr']*(1-float(curr_iter) / float(total_epoch) ) ** args['lr_decay'] #poly策略的公式
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            img,labels,dp = data

            inputs = torch.cat((img,dp),dim=0)
            

            batch_size = inputs.size(0)

           
            inputs = Variable(inputs).cuda(device_ids[0])
            
            labels = Variable(labels).cuda(device_ids[0])

            optimizer.zero_grad()    
            
            
            predict1,predict2,predict3,predict4,predict5  = net(inputs)

            

            
            loss_1 = bce_iou_loss(predict1,labels)
            loss_2 = structure_loss(predict2, labels)
            loss_3 = structure_loss(predict3, labels)
            loss_4 = structure_loss(predict4, labels)
            loss_5 = structure_loss(predict5, labels)
            
            
            
            loss = 1 * loss_1 + 1* loss_2 +1 * loss_3 + 1 * loss_4 + 2* loss_5  


  
            loss.backward()

            optimizer.step() 

            
            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_5_record.update(loss_5.data, batch_size)
            

            if curr_iter % 10 == 0 :
                writer.add_scalar('loss',loss,curr_iter)
                writer.add_scalar('loss_1',loss_1,curr_iter)
                writer.add_scalar('loss_2',loss_2,curr_iter)
                writer.add_scalar('loss_3',loss_3,curr_iter)
                writer.add_scalar('loss_4',loss_4,curr_iter)
                writer.add_scalar('loss_5',loss_5,curr_iter)
                

            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_1_record.avg, loss_2_record.avg,
                   loss_3_record.avg, loss_4_record.avg, loss_5_record.avg)
                   
            train_iterator.set_description(log)           
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch in args['save_point']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.cuda(device_ids[0])
        
        if (epoch >= 45 and epoch < args['epoch_num'])  :
            if epoch % 5 == 0 :
                torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))


        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            return
            


def main():
    print(args)
    print(exp_name)

    net = De_cod(backbone_path).cuda(device_ids[0]).train()         

    

    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    #设置断点
    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])

        #重新加载参数权重
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

        #剩余的要执行的次数
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)

        print(total_epoch)

    #使用多gpu训练  但我们这里只设置了一个gpu
    net = nn.DataParallel(net, device_ids=device_ids)
    print("Using {} GPU(s) to Train.".format(len(device_ids)))


    #打开日志 并记录
    open(log_path, 'w').write(str(args) + '\n\n')

    #训练      传入要训练的网络 和  优化方法
    train(net, optimizer)
    writer.close()


if __name__ == '__main__':
    main()

