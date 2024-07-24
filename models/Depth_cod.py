import torch
import torch.nn as nn
import torch.nn.functional as F
import backbone.resnet as resnet
import backbone.Swin as swin



def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.ReLU,nn.ReLU,nn.AdaptiveAvgPool2d,nn.Softmax)):
            pass
        else:
            m.initialize()





########   Channel Attention Block  ########

class CA_Block(nn.Module):
    #注意力通道模块
    def __init__(self,in_dim) :
        super(CA_Block,self).__init__()
        #输入通道数
        self.channel_in = in_dim
        
        #γ  初始化为1  是一个可学习的比例参数
        self.gamma = nn.Parameter(torch.ones(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            INPUTS: 
                x : input feature map (B x C x H x W)
            RETURN:
                OUT : channel attention features
        """

        #获取批量大小  通道数   高       宽
        bacthsize , channel , height , width = x.size()

        #  Query   并且转换形状为 BxCxN  N=HxW
        query = x.view(bacthsize,channel,-1)

        #  Key    同上  并且进行转置
        key = x.view(bacthsize,channel,-1).permute(0,2,1)

        # 矩阵相乘
        energy = torch.bmm(query,key)

        #注意力通道图
        attention = self.softmax(energy)

        # Value  
        value = x.view(bacthsize,channel,-1)

        # 融合 attention 和 value
        out = torch.bmm(attention,value)

        # 转换形状
        out = out.view(bacthsize,channel,height,width)

        # 输出
        out = self.gamma * out + x

        return out


######## Spatial Attention Block ########


class SA_Block(nn.Module):
    def __init__(self , in_dim):
        super(SA_Block,self).__init__()
        
        #输入的通道数
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)

        self.key_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)

        self.value_conv = nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            INPUTS: 
                x : input feature map (B x C x H x W)
            RETURN:
                OUT : spatial attention features
        """

        batchsize,channel,height,width = x.size()

        query_cov = self.query_conv(x).view(batchsize,-1,width*height).permute(0,2,1)

        key_conv = self.key_conv(x).view(batchsize,-1,width*height)

        energy = torch.bmm(query_cov,key_conv)

        attention = self.softmax(energy)

        value_conv = self.value_conv(x).view(batchsize,-1,width*height)

        out = torch.bmm(value_conv,attention.permute(0,2,1))

        out = out.view(batchsize,channel,height,width)

        out = self.gamma*out + x

        return out


##################################################################
# ################## Context Exploration Block ####################
###################################################################
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc + self.p2_channel_reduction(x)*p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc + self.p3_channel_reduction(x)*p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc + self.p4_channel_reduction(x)*p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce

###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    #总的定位模块
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        # out = cab + sab
        map = self.map(sab)

        return sab, map

####################################################################
######################### depth positioning ########################

class d_Positioning(nn.Module):
    def __init__(self, channel):
        super(d_Positioning, self).__init__()
        self.channel = channel
        # self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        # cab = self.cab(x)
        sab = self.sab(x)
        # out = cab + sab
        map = self.map(sab)

        return sab, map





###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up2 = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU())
        #CBRU  文中的Fup

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.input_map2 = nn.Sequential(nn.Sigmoid())
        #输入

        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        #输出

        self.fp = Context_Exploration_Block(self.channel1)
        #通过CE模块

        self.fn = Context_Exploration_Block(self.channel1)
        #通过CE模块

        self.alpha = nn.Parameter(torch.ones(1))
        #参数α

        self.beta = nn.Parameter(torch.ones(1))
        #参数β

        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map,upsample = True):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        if upsample :
            up = self.up(y)
        #首先输入高级别的特征y 通过CBRU（卷积 归一化 Relu 上采样）后输出
            input_map = self.input_map(in_map)
        #输入高级别预测  通过上采样和sigmoid函数 变化成0~1之间的数输出
        else:
            up = self.up2(y)
            input_map = self.input_map2(in_map)
        f_feature = x * input_map
        #假阳性特征  由当前特征x*输入 表示

        b_feature = x * (1 - input_map)
        #假阴性特征 由当前特征x*（1-输入）表示

        fp = self.fp(f_feature)
        #将假阳性特征输入到CE上下文模块中 得到输出fpd

        fn = self.fn(b_feature)
        #将假阴性特征输入到CE上下文模块中 得到输出fnd

        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)
        #Fr = BR(Fup−αFfpd)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)
        #F′r = BR(Fr +βFf nd)

        output_map = self.output_map(refine2)
        #卷积后的输出

        return refine2, output_map


###################    Cross-Modal fusion module   ####################
class CM_Block(nn.Module):
    def __init__(self):
        super(CM_Block,self).__init__()
        
    def forward(self,x):
        # resl = []

        # for i in range(len(x)//2):
        #     part1 = x[i]
        #     part2 = x[i+len(x)//2]
        #     sum = (part1 + part2 + (part1 * part2))
        #     resl.append(sum)
        t = torch.chunk(x,2,dim=0)
        rgb = t[0]
        d = t[1]
        return rgb,d


class Grafting(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k = nn.Linear(dim, dim , bias=qkv_bias)
        self.qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(8,8,kernel_size=3, stride=1, padding=1)
        self.lnx = nn.LayerNorm(64)
        self.lny = nn.LayerNorm(64)
        self.bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, y):          #x: resnet    y:swin
        batch_size = x.shape[0]
        chanel     = x.shape[1]
        sc = x     #sc store the origin x
        x = x.view(batch_size, chanel, -1).permute(0, 2, 1)  #flatten
        sc1 = x    #sc1 store the flatten x
        x = self.lnx(x)


        y = y.view(batch_size, chanel, -1).permute(0, 2, 1)
        y = self.lny(y)
        
        B, N, C = x.shape


        y_k = self.k(y).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_qv= self.qv(x).reshape(B,N,2,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_v = x_qv[0], x_qv[1] 
        y_k = y_k[0]
        attn = (x_q @ y_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ x_v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = (x+sc1)

        x = x.permute(0,2,1)
        x = x.view(batch_size,chanel,*sc.size()[2:])
        x = self.conv2(x)+x
        return x,self.act(self.bn(self.conv(attn+attn.transpose(-1,-2))))


    def initialize(self):
        weight_init(self)




class DB1(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(DB1,self).__init__()
        self.squeeze1 = nn.Sequential(  
                    nn.Conv2d(inplanes, outplanes,kernel_size=1,stride=1,padding=0), 
                    nn.BatchNorm2d(64), 
                    nn.ReLU(inplace=True)
                )
        self.squeeze2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3,stride=1,dilation=2,padding=2), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        z = self.squeeze2(self.squeeze1(x))   
        return z,z

    def initialize(self):
        weight_init(self)

class DB2(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(DB2,self).__init__()
        self.short_cut = nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes+outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self,x,z): 
        z = F.interpolate(z,size=x.size()[2:],mode='bilinear',align_corners=True)
        p = self.conv(torch.cat((x,z),1))
        sc = self.short_cut(z)
        p  = p+sc
        p2 = self.conv2(p)
        p  = p+p2
        return p,p
    
    def initialize(self):
        weight_init(self)

class DB3(nn.Module):
    def __init__(self) -> None:
        super(DB3,self).__init__()

        self.db2 = DB2(64,64)

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.sqz_r4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )

        self.sqz_s1=nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )
    def forward(self,s,r,up):
        up = F.interpolate(up,size=s.size()[2:],mode='bilinear',align_corners=True)
        s = self.sqz_s1(s)
        r = self.sqz_r4(r)
        sr = self.conv3x3(s+r)
        out,_  =self.db2(sr,up)
        return out,out
    def initialize(self):
        weight_init(self)



class IOU(nn.Module):
    def __init__(self):
        super(IOU,self).__init__()

    
    def _iou(self,pred,gt):
        pred = torch.sigmoid(pred)
        inter = (pred * gt).sum(dim = (2 , 3))
        union = (pred + gt).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)
        
        return iou.mean()

    def forward(self, pred, gt):
        return self._iou(pred, gt)
    




###################################################################
# ########################## NETWORK ##############################
###################################################################
class De_cod(nn.Module):
    def __init__(self, backbone_path=None):
        super(De_cod, self).__init__()
        # params

        # backbone
        resnet50 = resnet.resnet50(backbone_path)
        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu)   
        self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        

        self.swin = swin.Swintransformer(224)
        self.swin.load_state_dict(torch.load('./backbone/swin_large_patch4_window7_224_22k.pth')['model'],strict=False)

        # channel reduction
        self.cr4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        # positioning
        self.positioning = Positioning(512)
        self.d_positioning = d_Positioning(512)
        #定位 输入通道数512

        # focus
        self.focus3 = Focus(256, 512)
        #输入通道数256  512

        self.focus2 = Focus(128, 256)
        #输入通道数128  256

        self.focus1 = Focus(64, 128)
        #输入通道数64  128

        self.cm = CM_Block()

        self.map = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU() ,nn.Conv2d(64, 1, 7, 1, 3)) 

        self.C4 = nn.Sequential(nn.Conv2d(768, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.C3 = nn.Sequential(nn.Conv2d(768, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.C2 = nn.Sequential(nn.Conv2d(384, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.C1 = nn.Sequential(nn.Conv2d(192, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())


        self.up_channel = nn.Conv2d(64,512,1)
        self.up_channel2 = nn.Conv2d(64,256,1)

        self.GraF   = Grafting(64,num_heads=8)

        self.d1 = DB1(512,64)
        self.d2 = DB2(512,64)
        self.d3 = DB2(64,64)
        self.d4 = DB3()

        self.itter = 0

        self.sqz_s2=nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )
        
        self.sqz_cr4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )

        self.GF   = Grafting(64,num_heads=8)
        self.conv_att = nn.Conv2d(8,1,kernel_size=3, stride=1, padding=1)

        self.linear2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.iou = IOU()

        self.itter2 = 0
        
        

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

        
    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        r0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        r1 = self.layer1(r0)  # [-1, 256, h/4, w/4]       2,256,104,104
        r2 = self.layer2(r1)  # [-1, 512, h/8, w/8]       2,512,52,52
        r3 = self.layer3(r2)  # [-1, 1024, h/16, w/16]    2,1024,26,26
        r4 = self.layer4(r3)  # [-1, 2048, h/32, w/32]    2,2048,13,13
        # #通道数在变大 图片高和宽在变小  分辨率在降低  目的是提取多级特征  

        # # channel reduction   通道减少  论文图2中的CBR
        # #将提取的多级特征进行通道缩减 
        cr4 = self.cr4(r4)           #512
        cr3 = self.cr3(r3)           #256
        cr2 = self.cr2(r2)           #128
        cr1 = self.cr1(r1)           #64

        cm4,cm4_d = self.cm(cr4)
        cm3,cm3_d = self.cm(cr3)
        cm2,cm2_d = self.cm(cr2)
        cm1,cm1_d = self.cm(cr1)





        #swin 部分需要224 * 224   采样成224 * 224
        y = F.interpolate(x, size=(224,224), mode='bilinear',align_corners=True)
        s1,s2,s3,s4 = self.swin(y)
        #s1     2,128,56,56
        #s2     2,256,28,28
        #s3     2,512,14,14
        #s4     2,512,14,14

        s1 = self.C1(s1)
        s2 = self.C2(s2)
        s3 = self.C3(s3)
        s4 = self.C4(s4)
        
        

        cs4,cs4_d = self.cm(s4)
        cs3,cs3_d = self.cm(s3)
        cs2,cs2_d = self.cm(s2)
        cs1,cs1_d = self.cm(s1)

        # d0 = self.layer0(y)  # [-1, 64, h/2, w/2]
        # d1 = self.layer1(d0)  # [-1, 256, h/4, w/4]       2,256,104,104
        # d2 = self.layer2(d1)  # [-1, 512, h/8, w/8]       2,512,52,52
        # d3 = self.layer3(d2)  # [-1, 1024, h/16, w/16]    2,1024,26,26
        # d4 = self.layer4(d3)  # [-1, 2048, h/32, w/32]    2,2048,13,13
        # # #通道数在变大 图片高和宽在变小  分辨率在降低  目的是提取多级特征  

        # # # channel reduction   通道减少  论文图2中的CBR
        # # #将提取的多级特征进行通道缩减 
        # cr4_d = self.cr4(d4)           #512
        # cr3_d = self.cr3(d3)           #256
        # cr2_d = self.cr2(d2)           #128
        # cr1_d = self.cr1(d1)           #64

        

        # positioning
        #将最深层的特征输入至定位模块 输出定位特征 和经过卷积的预测 
        positioning, predict4 = self.positioning(cs4)
        positioning_d, predict4_d = self.d_positioning(cs4_d)
        

        # focus
        #将当前级的特征 高级定位特征 高级预测 输入至聚焦模块
        #x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction
        focus3, predict3 = self.focus3(cm3, positioning, predict4  )
        focus3_d, predict3_d= self.focus3(cm3_d, positioning_d, predict4 )

        focus2, predict2 = self.focus2(cm2, focus3, predict3)
        focus2_d,predict2_d = self.focus2(cm2_d, focus3_d, predict3)
        
        focus1, predict1 = self.focus1(cm1, focus2, predict2)
        focus1_d, predict1_d = self.focus1(cm1_d, focus2_d, predict2)

        # rgb_stream = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        # depth_stream = F.interpolate(predict1_d, size=x.size()[2:], mode='bilinear', align_corners=True)

        fus = focus1 + focus1_d 
        pre = self.map(fus)

        # rescale   进行上采样 重新缩放
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict4_d = F.interpolate(predict4_d, size=x.size()[2:], mode='bilinear', align_corners=True)            #change
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3_d = F.interpolate(predict3_d, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2_d = F.interpolate(predict2_d, size=x.size()[2:], mode='bilinear', align_corners=True)
        pre = F.interpolate(pre, size=x.size()[2:], mode='bilinear', align_corners=True)


        #将融合预测结果和depth分支的结果进行 相似性度量 如果相似度太差 表明加入depth不好  就不加depth
        # a = torch.mean(torch.abs(depth_stream/255.0-pre/255.0))
        # b = torch.mean(torch.abs(depth_stream/255.0-rgb_stream/255.0))
        # c = torch.mean(torch.abs(pre/255.0-rgb_stream/255.0))
        # d = torch.mean(torch.abs(pre/255.0-gt/255.0))

        # pre = pre if a < 0.005 else rgb_stream

        # rgb_iou = self.iou(rgb_stream,gt)
        # depth_iou = self.iou(depth_stream,gt)

        # if rgb_iou > depth_iou :
        #     pre1 = pre 
        #     self.itter2 += 1
        #     itter = self.itter2
        # else:
        #     pre1 = rgb_stream
        
        


        # self.itter +=1
        # itter = self.itter 

        if self.training:
            # return predict4,rgb_stream, predict4_d,  depth_stream , pre1   #,self.conv_att(attmap),wr,ws 
            # return predict4,predict3,predict2, predict4_d,predict3_d,  predict2_d , pre
            return predict4,predict2, predict4_d ,predict2_d , pre


        # a = torch.mean(torch.abs(depth_stream/255.0-pre/255.0))
        # b = torch.mean(torch.abs(depth_stream/255.0-rgb_stream/255.0))
        # c = torch.mean(torch.abs(pre/255.0-rgb_stream/255.0))
        # d = torch.mean(torch.abs(pre/255.0-gt/255.0))

        # pre = pre if a < 0.15 else rgb_stream
        # return torch.sigmoid(predict4), torch.sigmoid(rgb_stream), torch.sigmoid(predict4_d), torch.sigmoid(depth_stream), torch.sigmoid(pre1)
        
        
        return torch.sigmoid(predict4), torch.sigmoid(predict2), torch.sigmoid(predict4_d), torch.sigmoid(predict2_d), torch.sigmoid(pre)
        # return torch.sigmoid(predict4),torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(predict4_d), torch.sigmoid(predict3_d), torch.sigmoid(predict2_d), torch.sigmoid(pre)