import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean

from utils.config import *
from utils.misc import *
from models.Depth_cod import De_cod

torch.manual_seed(2021)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

results_path = './results'
check_mkdir(results_path)
exp_name = 'Depth_cod'
args = {
    'scale': 448,
    'save_results': True
}

print(torch.__version__)

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dp_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor()])

gt_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor()])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
                       ('CAMO_depth', camo_path),
                       ('cod10k_depth_test', cod10k_path),
                       ('NC4K',NC4K_path)
                       ])


results = OrderedDict()

def main():
    net = De_cod(backbone_path).cuda(device_ids[0])

    net.load_state_dict(torch.load('checkpoints/Depth_cod/55.pth'))
    print('Load {} succeed!'.format('55.pth'))

    net.eval()
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            time_list = []
            image_path = os.path.join(root, 'rgb')
            depth_path = os.path.join(root, 'depth')
            gt_path = os.path.join(root, 'gt')

            if args['save_results']:
                check_mkdir(os.path.join(results_path, exp_name, name))
                

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
            for idx, img_name in enumerate(img_list):
                imgs = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')
                dep = Image.open(os.path.join(depth_path, img_name + '.png')).convert('RGB')
                gts = Image.open(os.path.join(gt_path, img_name + '.png')).convert('L')

                w , h = imgs.size

                img = img_transform(imgs).unsqueeze(0)
                dp = dp_transform(dep).unsqueeze(0)
                gt = dp_transform(gts).unsqueeze(0)

                fus = torch.cat((img,dp),dim=0)

                inputs = Variable(fus).cuda(device_ids[0])
                gt = Variable(gt).cuda(device_ids[0])

                start_each = time.time()
                
                _,_,_,_,prediction = net(inputs)
                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))
                

                if args['save_results']:
                    Image.fromarray(prediction).convert('L').save(os.path.join(results_path, exp_name, name, img_name + '.png'))
                    
                    
            print(('{}'.format(exp_name)))
            print("{}'s average Time Is : {:.3f} s".format(name, mean(time_list)))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))

if __name__ == '__main__':
    main()
