import random
random.seed(64)

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp_mixer_pytorch import MLPMixer

    
class QNet_CNN(nn.Module): #proposed QNet_CNN model
    def __init__(self):
        super(QNet_CNN, self).__init__()


        # Left Distortion map convolution layers
        self.conv1L = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2L = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3L = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4L = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # BN and FC for left distortion map
        self.bn1_L = nn.BatchNorm2d(32)
        self.bn2_L = nn.BatchNorm2d(32)
        self.bn3_L = nn.BatchNorm2d(64)
        # self.fc1_L = nn.Linear(256, 128) # for patch size 16
        self.fc1_L = nn.Linear(1024, 128) # for patch size 32
        # self.fc1_L = nn.Linear(4096, 128) # for patch size 64
        # self.fc1_L = nn.Linear(16384, 128) # for patch size 128


        # Right Distortion map convolution layers
        self.conv1R = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2R = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3R = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4R = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # BN and FC for right distortion map
        self.bn1_R = nn.BatchNorm2d(32)
        self.bn2_R = nn.BatchNorm2d(32)
        self.bn3_R = nn.BatchNorm2d(64)
        # self.fc1_R = nn.Linear(256, 128) # for patch size 16
        self.fc1_R = nn.Linear(1024, 128) # for patch size 32
        # self.fc1_R = nn.Linear(4096, 128) # for patch size 64
        # self.fc1_R = nn.Linear(16384, 128) # for patch size 128


        # FC : concat left and right distortion maps
        self.fc1_1 = nn.Linear(128 * 2, 32) 

        # Left image convolution layers
        self.conv1iL = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2iL = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3iL = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4iL = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # BN and FC for left image
        self.bn1_iL = nn.BatchNorm2d(32)
        self.bn2_iL = nn.BatchNorm2d(32)
        self.bn3_iL = nn.BatchNorm2d(64)
        # self.fc1_iL = nn.Linear(256, 128) # for patch size 16
        self.fc1_iL = nn.Linear(1024, 128) # for patch size 32
        # self.fc1_iL = nn.Linear(4096, 128) # for patch size 64
        # self.fc1_iL = nn.Linear(16384, 128) # for patch size 128

        # Right image convolution layers
        self.conv1iR = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2iR = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3iR = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4iR = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # BN and FC for right image
        self.bn1_iR = nn.BatchNorm2d(32)
        self.bn2_iR = nn.BatchNorm2d(32)
        self.bn3_iR = nn.BatchNorm2d(64)
        # self.fc1_iR = nn.Linear(256, 128) # for patch size 16
        self.fc1_iR = nn.Linear(1024, 128) # for patch size 32
        # self.fc1_iR = nn.Linear(4096, 128) # for patch size 64
        # self.fc1_iR = nn.Linear(16384, 128) # for patch size 128

        # FC : concat left and right images
        self.fc1_i1 = nn.Linear(128 * 2, 32)  

        # FC : concat distortion map and imag features
        self.fc_glob = nn.Linear(32*2, 1)
        
    def forward(self, xL, xR, iL, iR):

        # xL: left distortion map
        # xR: right distortion map
        # iL: left image
        # iR: right image

        x_distort_L = xL.view(-1, xL.size(-3), xL.size(-2), xL.size(-1))
        x_distort_R = xR.view(-1, xR.size(-3), xR.size(-2), xR.size(-1))

        i_distort_L = iL.view(-1, iL.size(-3), iL.size(-2), iL.size(-1))
        i_distort_R = iR.view(-1, iR.size(-3), iR.size(-2), iR.size(-1))

        
        Out_L = F.relu(self.conv1L(x_distort_L)) 
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn1_L(Out_L)

        Out_L = F.relu(self.conv2L(Out_L))
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn2_L(Out_L)


        Out_L = F.relu(self.conv3L(Out_L))
        Out_L = F.relu(self.conv4L(Out_L))

        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn3_L(Out_L)

        
        Out_L = Out_L.view(-1, self.num_flat_features(Out_L))
        Out_L = F.relu(self.fc1_L(Out_L))


        Out_iL = F.relu(self.conv1iL(i_distort_L)) 
        Out_iL = F.max_pool2d(Out_iL, (2, 2), stride=2)
        Out_iL = self.bn1_iL(Out_iL)

        Out_iL = F.relu(self.conv2iL(Out_iL))
        Out_iL = F.max_pool2d(Out_iL, (2, 2), stride=2)
        Out_iL = self.bn2_iL(Out_iL)


        Out_iL = F.relu(self.conv3iL(Out_iL))
        Out_iL = F.relu(self.conv4iL(Out_iL))

        Out_iL = F.max_pool2d(Out_iL, (2, 2), stride=2)
        Out_iL = self.bn3_iL(Out_iL)

        
        Out_iL = Out_iL.view(-1, self.num_flat_features(Out_iL))
        Out_iL = F.relu(self.fc1_iL(Out_iL))
        
        Out_R = F.relu(self.conv1R(x_distort_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn1_R(Out_R)

        Out_R = F.relu(self.conv2R(Out_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn2_R(Out_R)
      
        Out_R = F.relu(self.conv3R(Out_R))
        Out_R = F.relu(self.conv4R(Out_R))
        

        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn3_R(Out_R)
        
        Out_R = Out_R.view(-1, self.num_flat_features(Out_R))
        Out_R = F.relu(self.fc1_R(Out_R))

        Out_iR = F.relu(self.conv1iR(i_distort_R))
        Out_iR = F.max_pool2d(Out_iR, (2, 2), stride=2)
        Out_iR = self.bn1_iR(Out_iR)

        Out_iR = F.relu(self.conv2iR(Out_iR))
        Out_iR = F.max_pool2d(Out_iR, (2, 2), stride=2)
        Out_iR = self.bn2_iR(Out_iR)
      
        Out_iR = F.relu(self.conv3iR(Out_iR))
        Out_iR = F.relu(self.conv4iR(Out_iR))
        

        Out_iR = F.max_pool2d(Out_iR, (2, 2), stride=2)
        Out_iR = self.bn3_iR(Out_iR)
        
        Out_iR = Out_iR.view(-1, self.num_flat_features(Out_iR))
        Out_iR = F.relu(self.fc1_iR(Out_iR))

        cat_total = F.relu(torch.cat((Out_L, Out_R), dim=1)) # distortion map features

        Global_Quality_1 = F.relu(self.fc1_1(cat_total))

        cat_total_2 = F.relu(torch.cat((Out_iL, Out_iR), dim=1)) # image features

        Global_Quality_2 = F.relu(self.fc1_i1(cat_total_2))

        cat_total_g = F.relu(torch.cat((Global_Quality_1, Global_Quality_2), dim=1)) # concatenating distortion map and image features

        Global_Quality = F.relu(self.fc_glob(cat_total_g))


        return Global_Quality


    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class QNet_Mixer(nn.Module): # Proposed QNet_Mixer
    def __init__(self, image_size = (32, 32), channels = 1, patch_size = 8, dim = 128, depth = 8, num_classes = 1):
        super(QNet_Mixer, self).__init__()
        self.dl_mixer_model = MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
        self.dr_mixer_model = MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
        self.dl_mixer_model = torch.nn.Sequential(*list(self.dl_mixer_model.children())[:-1]) # for left distortion map
        self.dr_mixer_model = torch.nn.Sequential(*list(self.dr_mixer_model.children())[:-1]) # for right distortion map

        self.il_mixer_model = MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
        self.ir_mixer_model = MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            num_classes = num_classes
        ) 
        self.il_mixer_model = torch.nn.Sequential(*list(self.il_mixer_model.children())[:-1]) # for left image
        self.ir_mixer_model = torch.nn.Sequential(*list(self.ir_mixer_model.children())[:-1]) # for right image

        self.fc_d = nn.Linear(dim*2, 128) # FC for left and right distortion maps
        self.fc_i = nn.Linear(dim*2, 128) # FC for left and right images

        self.fc = nn.Linear(128*2, num_classes) # FC for distortion map and image features


    def forward(self, xl, xr, il, ir):

        # xL: left distortion map
        # xR: right distortion map
        # iL: left image
        # iR: right image

        xl = xl.view(-1, xl.size(-3), xl.size(-2), xl.size(-1))
        xr = xr.view(-1, xr.size(-3), xr.size(-2), xr.size(-1))

        il = il.view(-1, il.size(-3), il.size(-2), il.size(-1))
        ir = ir.view(-1, ir.size(-3), ir.size(-2), ir.size(-1))

        xlf = self.dl_mixer_model(xl)
        xrf = self.dr_mixer_model(xr)
        
        ilf = self.il_mixer_model(il)
        irf = self.ir_mixer_model(ir)
        
        xf = torch.cat((xlf, xrf), dim=1) # distortion map features
        imf = torch.cat((ilf, irf), dim=1) # image features

        xf = self.fc_d(xf)
        imf = self.fc_i(imf)
        
        
        f = torch.cat((xf, imf), dim=1) # concatenating distortion map and image features

        return self.fc(f)
    

class MessaiWD(nn.Module): # Main Messai et. al. code taken from https://github.com/o-messai/Multi-score-SIQA 
                           # and modified to take in 2 channels (image+distortion map)
    def __init__(self):
        super(MessaiWD, self).__init__()

        # conv of left view
        self.conv1L = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2L = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3L = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # conv of right view
        self.conv1R = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2R = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3R = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # conv of stereo view
        self.conv1S = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2S = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3S = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)


        # FC of letf view
        self.bn1_L = nn.BatchNorm2d(32)
        self.bn2_L = nn.BatchNorm2d(64)
        self.bn3_L = nn.BatchNorm2d(128)
        self.fc1_L = nn.Linear(2048, 1024)
        self.fc2_L = nn.Linear(1024, 512)
        self.fc3_L = nn.Linear(512, 1)

        # FC of right view
        self.bn1_R = nn.BatchNorm2d(32)
        self.bn2_R = nn.BatchNorm2d(64)
        self.bn3_R = nn.BatchNorm2d(128)
        self.fc1_R = nn.Linear(2048, 1024)
        self.fc2_R = nn.Linear(1024, 512)
        self.fc3_R = nn.Linear(512, 1)

        # FC of stereo
        self.bn1_S = nn.BatchNorm2d(32)
        self.bn2_S = nn.BatchNorm2d(64)
        self.bn3_S = nn.BatchNorm2d(128)
        self.fc1_S = nn.Linear(2048, 1024)
        self.fc2_S = nn.Linear(1024, 512)
        self.fc3_S = nn.Linear(512, 1)

        # FC of global score
        self.fc1_1 = nn.Linear(512 * 3, 1024)  
        self.fc2_1 = nn.Linear(1024, 512)
        self.fc3_1 = nn.Linear(512, 1)
        
    def forward(self, xL,xR, iL, iR):

        x_distort_L = xL.view(-1, xL.size(-3), xL.size(-2), xL.size(-1))
        x_distort_R = xR.view(-1, xR.size(-3), xR.size(-2), xR.size(-1))
        i_distort_L = iL.view(-1, iL.size(-3), iL.size(-2), iL.size(-1))
        i_distort_R = iR.view(-1, iR.size(-3), iR.size(-2), iR.size(-1))

        cat_l = F.relu(torch.cat((x_distort_L, i_distort_L), dim=1))
        cat_r = F.relu(torch.cat((x_distort_R, i_distort_R), dim=1))

     ####################################################### left view  ####################################################  
        
        Out_L = F.relu(self.conv1L(cat_l)) 
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn1_L(Out_L)

        Out_L = F.relu(self.conv2L(Out_L))
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn2_L(Out_L)


        Out_L = F.relu(self.conv3L(Out_L))

        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn3_L(Out_L)

        
        Out_L = Out_L.view(-1, self.num_flat_features(Out_L))
        Out_L = self.fc1_L(Out_L)
        Out_L = F.dropout(Out_L, p=0.5, training=True, inplace=False)

        Out_LF = self.fc2_L(Out_L)

        Out_L = F.relu(self.fc3_L(Out_LF))
                    
     ####################################################### right view  ####################################################  
        
        
        Out_R = F.relu(self.conv1R(cat_r))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn1_R(Out_R)

        Out_R = F.relu(self.conv2R(Out_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn2_R(Out_R)
      
        Out_R = F.relu(self.conv3R(Out_R))
        
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn3_R(Out_R)
        
        Out_R = Out_R.view(-1, self.num_flat_features(Out_R))
        Out_R = self.fc1_R(Out_R)
        Out_R = F.dropout(Out_R, p=0.5, training=True, inplace=False)

        Out_RF = self.fc2_R(Out_R)
        
        Out_R = F.relu(self.fc3_R(Out_RF))
                      
     ####################################################### Stereo view  ####################################################  
        
        input_s = torch.cat((cat_l, cat_r), dim=1)

        Out_S = F.relu(self.conv1S(input_s))
        Out_S = F.max_pool2d(Out_S, (2, 2), stride=2)
        Out_S = self.bn1_S(Out_S)

        Out_S = F.relu(self.conv2S(Out_S))
        Out_S = F.max_pool2d(Out_S, (2, 2), stride=2)
        Out_S = self.bn2_S(Out_S)
      
        Out_S = F.relu(self.conv3S(Out_S))
        

        Out_S = F.max_pool2d(Out_S, (2, 2), stride=2)
        Out_S = self.bn3_S(Out_S)
        
        Out_S = Out_S.view(-1, self.num_flat_features(Out_S))
        Out_S = self.fc1_S(Out_S)
        Out_S = F.dropout(Out_S, p=0.5, training=True, inplace=False)

        Out_SF = self.fc2_S(Out_S)

        Out_S = F.relu(self.fc3_S(Out_SF))


        #############################################    Concatenation  ############################

        cat_total = F.relu(torch.cat((Out_RF, Out_LF, Out_SF), dim=1))
        ################################## Quality score prediction  ##########################################

        Quality_left = Out_L
        Quality_right = Out_R
        Quality_stereo = Out_S 

        fc1_1 = self.fc1_1(cat_total)
        fc2_1 = F.relu(self.fc2_1(fc1_1))
        Global_Quality = F.relu(self.fc3_1(fc2_1))                        # Global Quality score prediction

        return Global_Quality, Quality_left, Quality_right, Quality_stereo
    
    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class ZhouWD(nn.Module): # Main Zhou et. al. architecture taken from https://github.com/weizhou-geek/Stereoscopic-Image-Quality-Assessment-Network
                         # and modified to take in 2 channels (image+distortion map)
    def __init__(self):
        super(ZhouWD, self).__init__()
        # Left Image Layers
        self.left_conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.left_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.left_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.left_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.left_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.left_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.left_conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.left_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.left_fc6 = nn.Linear(128 * 4 * 4, 512)
        self.left_fc7 = nn.Linear(512, 512)
        
        # Right Image Layers
        self.right_conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.right_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.right_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.right_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.right_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.right_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.right_conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.right_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.right_fc6 = nn.Linear(128 * 4 * 4, 512)
        self.right_fc7 = nn.Linear(512, 512)
        
        # Fusion Layers
        self.fusion1_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion1_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fusion1_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fusion1_conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fusion1_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion1_fc6 = nn.Linear(128 * 4 * 4, 512)
        self.fusion1_fc7 = nn.Linear(512, 512)
        
        self.fusion2_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion2_fc6 = nn.Linear(128 * 4 * 4, 512)
        self.fusion2_fc7 = nn.Linear(512, 512)
        
        self.fc8 = nn.Linear(2048, 1024)
        self.fc9 = nn.Linear(1024, 1)
        
        # ELU and Dropout layers
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(p=0.35)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, left_image, right_image, left_dmap, right_dmap):

        left_dmap = left_dmap.view(-1, left_dmap.size(-3), left_dmap.size(-2), left_dmap.size(-1))
        right_dmap = right_dmap.view(-1, right_dmap.size(-3), right_dmap.size(-2), right_dmap.size(-1))
        left_image = left_image.view(-1, left_image.size(-3), left_image.size(-2), left_image.size(-1))
        right_image = right_image.view(-1, right_image.size(-3), right_image.size(-2), right_image.size(-1))

        left_image = F.relu(torch.cat((left_dmap, left_image), dim=1))
        right_image = F.relu(torch.cat((right_dmap, right_image), dim=1))

        left_conv1 = self.elu(self.left_conv1(left_image))
        left_pool1 = self.left_pool1(left_conv1)
        left_conv2 = self.elu(self.left_conv2(left_pool1))
        left_pool2 = self.left_pool2(left_conv2)
        left_conv3 = self.elu(self.left_conv3(left_pool2))
        left_conv4 = self.elu(self.left_conv4(left_conv3))
        left_conv5 = self.elu(self.left_conv5(left_conv4))
        left_pool5 = self.left_pool5(left_conv5)
        left_flat6 = left_pool5.view(-1, 128 * 4 * 4)
        left_fc6 = self.elu(self.left_fc6(left_flat6))
        left_fc6 = self.dropout1(left_fc6)
        left_fc7 = self.elu(self.left_fc7(left_fc6))
        left_drop7 = self.dropout(left_fc7)
        
        # Right image
        right_conv1 = self.elu(self.right_conv1(right_image))
        right_pool1 = self.right_pool1(right_conv1)
        right_conv2 = self.elu(self.right_conv2(right_pool1))
        right_pool2 = self.right_pool2(right_conv2)
        right_conv3 = self.elu(self.right_conv3(right_pool2))
        right_conv4 = self.elu(self.right_conv4(right_conv3))
        right_conv5 = self.elu(self.right_conv5(right_conv4))
        right_pool5 = self.right_pool5(right_conv5)
        right_flat6 = right_pool5.view(-1, 128 * 4 * 4)
        right_fc6 = self.elu(self.right_fc6(right_flat6))
        right_fc6 = self.dropout1(right_fc6)
        right_fc7 = self.elu(self.right_fc7(right_fc6))
        right_drop7 = self.dropout(right_fc7)
        
        # Fusion1
        fusion1_add_conv2 = left_conv2 + right_conv2
        fusion1_subtract_conv2 = left_conv2 - right_conv2
        fusion1_concat_conv2 = torch.cat((fusion1_add_conv2, fusion1_subtract_conv2), dim=1)
        fusion1_elu2 = self.elu(fusion1_concat_conv2)
        fusion1_pool2 = self.fusion1_pool2(fusion1_elu2)
        fusion1_conv3 = self.elu(self.fusion1_conv3(fusion1_pool2))
        fusion1_conv4 = self.elu(self.fusion1_conv4(fusion1_conv3))
        fusion1_conv5 = self.elu(self.fusion1_conv5(fusion1_conv4))
        fusion1_pool5 = F.max_pool2d(fusion1_conv5, kernel_size=2, stride=2)
        fusion1_flat5 = fusion1_pool5.view(-1, 128 * 4 * 4)
        fusion1_fc6 = self.elu(self.fusion1_fc6(fusion1_flat5))
        fusion1_fc6 = self.dropout1(fusion1_fc6)
        fusion1_fc7 = self.elu(self.fusion1_fc7(fusion1_fc6))
        fusion1_drop7 = self.dropout(fusion1_fc7)
        
        # Fusion2
        fusion2_add_conv5 = left_conv4 + right_conv4
        fusion2_subtract_conv5 = left_conv4 - right_conv4
        fusion2_concat_conv5 = torch.cat((fusion2_add_conv5, fusion2_subtract_conv5), dim=1)
        fusion2_elu5 = self.elu(fusion2_concat_conv5)
        fusion2_pool5 = self.fusion2_pool5(fusion2_elu5)
        fusion2_flat6 = fusion2_pool5.view(-1, 128 * 4 * 4)
        fusion2_fc6 = self.elu(self.fusion2_fc6(fusion2_flat6))
        fusion2_fc6 = self.dropout1(fusion2_fc6)
        fusion2_fc7 = self.elu(self.fusion2_fc7(fusion2_fc6))
        fusion2_drop7 = self.dropout(fusion2_fc7)
        
        # Fusion3
        fusion3_drop7 = torch.cat((left_drop7, right_drop7, fusion1_drop7, fusion2_drop7), dim=1)
        
        # Final FC layers
        fc8 = self.fc8(fusion3_drop7)
        fc9 = self.fc9(fc8)
        
        return fc9
    
class Messai(nn.Module): # Messai et. al. code taken from https://github.com/o-messai/Multi-score-SIQA 
    def __init__(self):
        super(Messai, self).__init__()
        # conv of left view
        self.conv1L = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2L = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3L = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # conv of right view
        self.conv1R = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2R = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3R = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # conv of stereo view
        self.conv1S = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2S = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3S = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # FC of letf view
        self.bn1_L = nn.BatchNorm2d(32)
        self.bn2_L = nn.BatchNorm2d(64)
        self.bn3_L = nn.BatchNorm2d(128)
        self.fc1_L = nn.Linear(2048, 1024) # patch size 32 
        self.fc2_L = nn.Linear(1024, 512)
        self.fc3_L = nn.Linear(512, 1)

        # FC of right view
        self.bn1_R = nn.BatchNorm2d(32)
        self.bn2_R = nn.BatchNorm2d(64)
        self.bn3_R = nn.BatchNorm2d(128)
        self.fc1_R = nn.Linear(2048, 1024) # patch size 32 
        self.fc2_R = nn.Linear(1024, 512)
        self.fc3_R = nn.Linear(512, 1)

        # FC of stereo
        self.bn1_S = nn.BatchNorm2d(32) 
        self.bn2_S = nn.BatchNorm2d(64)
        self.bn3_S = nn.BatchNorm2d(128)
        self.fc1_S = nn.Linear(2048, 1024) # patch size 32 
        self.fc2_S = nn.Linear(1024, 512)
        self.fc3_S = nn.Linear(512, 1)

        # FC of global score
        self.fc1_1 = nn.Linear(512 * 3, 1024)  
        self.fc2_1 = nn.Linear(1024, 512)
        self.fc3_1 = nn.Linear(512, 1)
        
    def forward(self, xL,xR):

        x_distort_L = xL.view(-1, xL.size(-3), xL.size(-2), xL.size(-1))
        x_distort_R = xR.view(-1, xR.size(-3), xR.size(-2), xR.size(-1))

     ####################################################### left view  ####################################################  
        
        Out_L = F.relu(self.conv1L(x_distort_L)) 
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)#32x16×16
        Out_L = self.bn1_L(Out_L)

        Out_L = F.relu(self.conv2L(Out_L))#64x8×8
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn2_L(Out_L)


        Out_L = F.relu(self.conv3L(Out_L))#128x8×8

        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)#128x4×4
        Out_L = self.bn3_L(Out_L)

        
        Out_L = Out_L.view(-1, self.num_flat_features(Out_L))
        Out_L = self.fc1_L(Out_L)#512
        Out_L = F.dropout(Out_L, p=0.5, training=True, inplace=False)

        Out_LF = self.fc2_L(Out_L)#512
        #Out_LF = F.dropout(Out_L, p=0.0, training=True, inplace=False)

        Out_L = F.relu(self.fc3_L(Out_LF))
                    
     ####################################################### right view  ####################################################  
        
        
        Out_R = F.relu(self.conv1R(x_distort_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn1_R(Out_R)

        Out_R = F.relu(self.conv2R(Out_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn2_R(Out_R)
      
        Out_R = F.relu(self.conv3R(Out_R))
        
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn3_R(Out_R)
        
        Out_R = Out_R.view(-1, self.num_flat_features(Out_R))
        Out_R = self.fc1_R(Out_R)
        Out_R = F.dropout(Out_R, p=0.5, training=True, inplace=False)

        Out_RF = self.fc2_R(Out_R)
        #Out_R = F.dropout(Out_R, p=0.0, training=True, inplace=False)
        
        Out_R = F.relu(self.fc3_R(Out_RF))
                      
     ####################################################### Stereo view  ####################################################  
        
        input_s = torch.cat((x_distort_L, x_distort_R), dim=1)

        Out_S = F.relu(self.conv1S(input_s))
        Out_S = F.max_pool2d(Out_S, (2, 2), stride=2)
        Out_S = self.bn1_S(Out_S)

        Out_S = F.relu(self.conv2S(Out_S))
        Out_S = F.max_pool2d(Out_S, (2, 2), stride=2)
        Out_S = self.bn2_S(Out_S)
      
        Out_S = F.relu(self.conv3S(Out_S))
        

        Out_S = F.max_pool2d(Out_S, (2, 2), stride=2)
        Out_S = self.bn3_S(Out_S)
        
        Out_S = Out_S.view(-1, self.num_flat_features(Out_S))
        Out_S = self.fc1_S(Out_S)
        Out_S = F.dropout(Out_S, p=0.5, training=True, inplace=False)

        Out_SF = self.fc2_S(Out_S)
        #Out_S = F.dropout(Out_S, p=0.0, training=True, inplace=False)

        Out_S = F.relu(self.fc3_S(Out_SF))


        #############################################    Concatenation  ############################

        cat_total = F.relu(torch.cat((Out_RF, Out_LF, Out_SF), dim=1))
        #fc_total = cat_total.view(-1, self.num_flat_features(cat_total))
        ################################## Quality score prediction  ##########################################

        Quality_left = Out_L
        Quality_right = Out_R
        Quality_stereo = Out_S 

        fc1_1 = self.fc1_1(cat_total)
        fc2_1 = F.relu(self.fc2_1(fc1_1))
        Global_Quality = F.relu(self.fc3_1(fc2_1))                        # Global Quality score prediction

        return Global_Quality, Quality_left, Quality_right, Quality_stereo
    
    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class Zhou(nn.Module): # Zhou et. al. architecture taken from https://github.com/weizhou-geek/Stereoscopic-Image-Quality-Assessment-Network
    def __init__(self):
        super(Zhou, self).__init__()
        # Left Image Layers
        self.left_conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.left_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.left_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.left_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.left_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.left_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.left_conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.left_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.left_fc6 = nn.Linear(128 * 4 * 4, 512)
        self.left_fc7 = nn.Linear(512, 512)
        
        # Right Image Layers
        self.right_conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.right_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.right_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.right_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.right_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.right_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.right_conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.right_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.right_fc6 = nn.Linear(128 * 4 * 4, 512)
        self.right_fc7 = nn.Linear(512, 512)
        
        # Fusion Layers
        self.fusion1_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion1_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fusion1_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fusion1_conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fusion1_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion1_fc6 = nn.Linear(128 * 4 * 4, 512)
        self.fusion1_fc7 = nn.Linear(512, 512)
        
        self.fusion2_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion2_fc6 = nn.Linear(128 * 4 * 4, 512)
        self.fusion2_fc7 = nn.Linear(512, 512)
        
        self.fc8 = nn.Linear(2048, 1024)
        self.fc9 = nn.Linear(1024, 1)
        
        # ELU and Dropout layers
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(p=0.35)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, left_image, right_image, left_dmap, right_dmap):

        left_dmap = left_dmap.view(-1, left_dmap.size(-3), left_dmap.size(-2), left_dmap.size(-1))
        right_dmap = right_dmap.view(-1, right_dmap.size(-3), right_dmap.size(-2), right_dmap.size(-1))
        left_image = left_image.view(-1, left_image.size(-3), left_image.size(-2), left_image.size(-1))
        right_image = right_image.view(-1, right_image.size(-3), right_image.size(-2), right_image.size(-1))

        left_image = F.relu(torch.cat((left_dmap, left_image), dim=1))
        right_image = F.relu(torch.cat((right_dmap, right_image), dim=1))

        left_conv1 = self.elu(self.left_conv1(left_image))
        left_pool1 = self.left_pool1(left_conv1)
        left_conv2 = self.elu(self.left_conv2(left_pool1))
        left_pool2 = self.left_pool2(left_conv2)
        left_conv3 = self.elu(self.left_conv3(left_pool2))
        left_conv4 = self.elu(self.left_conv4(left_conv3))
        left_conv5 = self.elu(self.left_conv5(left_conv4))
        left_pool5 = self.left_pool5(left_conv5)
        left_flat6 = left_pool5.view(-1, 128 * 4 * 4)
        left_fc6 = self.elu(self.left_fc6(left_flat6))
        left_fc6 = self.dropout1(left_fc6)
        left_fc7 = self.elu(self.left_fc7(left_fc6))
        left_drop7 = self.dropout(left_fc7)
        
        # Right image
        right_conv1 = self.elu(self.right_conv1(right_image))
        right_pool1 = self.right_pool1(right_conv1)
        right_conv2 = self.elu(self.right_conv2(right_pool1))
        right_pool2 = self.right_pool2(right_conv2)
        right_conv3 = self.elu(self.right_conv3(right_pool2))
        right_conv4 = self.elu(self.right_conv4(right_conv3))
        right_conv5 = self.elu(self.right_conv5(right_conv4))
        right_pool5 = self.right_pool5(right_conv5)
        right_flat6 = right_pool5.view(-1, 128 * 4 * 4)
        right_fc6 = self.elu(self.right_fc6(right_flat6))
        right_fc6 = self.dropout1(right_fc6)
        right_fc7 = self.elu(self.right_fc7(right_fc6))
        right_drop7 = self.dropout(right_fc7)
        
        # Fusion1
        fusion1_add_conv2 = left_conv2 + right_conv2
        fusion1_subtract_conv2 = left_conv2 - right_conv2
        fusion1_concat_conv2 = torch.cat((fusion1_add_conv2, fusion1_subtract_conv2), dim=1)
        fusion1_elu2 = self.elu(fusion1_concat_conv2)
        fusion1_pool2 = self.fusion1_pool2(fusion1_elu2)
        fusion1_conv3 = self.elu(self.fusion1_conv3(fusion1_pool2))
        fusion1_conv4 = self.elu(self.fusion1_conv4(fusion1_conv3))
        fusion1_conv5 = self.elu(self.fusion1_conv5(fusion1_conv4))
        fusion1_pool5 = F.max_pool2d(fusion1_conv5, kernel_size=2, stride=2)
        fusion1_flat5 = fusion1_pool5.view(-1, 128 * 4 * 4)
        fusion1_fc6 = self.elu(self.fusion1_fc6(fusion1_flat5))
        fusion1_fc6 = self.dropout1(fusion1_fc6)
        fusion1_fc7 = self.elu(self.fusion1_fc7(fusion1_fc6))
        fusion1_drop7 = self.dropout(fusion1_fc7)
        
        # Fusion2
        fusion2_add_conv5 = left_conv4 + right_conv4
        fusion2_subtract_conv5 = left_conv4 - right_conv4
        fusion2_concat_conv5 = torch.cat((fusion2_add_conv5, fusion2_subtract_conv5), dim=1)
        fusion2_elu5 = self.elu(fusion2_concat_conv5)
        fusion2_pool5 = self.fusion2_pool5(fusion2_elu5)
        fusion2_flat6 = fusion2_pool5.view(-1, 128 * 4 * 4)
        fusion2_fc6 = self.elu(self.fusion2_fc6(fusion2_flat6))
        fusion2_fc6 = self.dropout1(fusion2_fc6)
        fusion2_fc7 = self.elu(self.fusion2_fc7(fusion2_fc6))
        fusion2_drop7 = self.dropout(fusion2_fc7)
        
        # Fusion3
        fusion3_drop7 = torch.cat((left_drop7, right_drop7, fusion1_drop7, fusion2_drop7), dim=1)
        
        # Final FC layers
        fc8 = self.fc8(fusion3_drop7)
        fc9 = self.fc9(fc8)
        
        return fc9
    
class QNet_CNNWOD(nn.Module): # Proposed QNet_CNN model without distortion maps
    def __init__(self):
        super(QNet_CNNWOD, self).__init__()


        # For left image
        self.conv1L = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2L = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3L = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4L = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.bn1_L = nn.BatchNorm2d(32)
        self.bn2_L = nn.BatchNorm2d(32)
        self.bn3_L = nn.BatchNorm2d(64)
        # self.fc1_L = nn.Linear(256, 128) # for patch size 16
        self.fc1_L = nn.Linear(1024, 128) # for patch size 32
        # self.fc1_L = nn.Linear(4096, 128) # for patch size 64
        # self.fc1_L = nn.Linear(16384, 128) # for patch size 128

        # For right image
        self.conv1R = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2R = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3R = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4R = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.bn1_R = nn.BatchNorm2d(32)
        self.bn2_R = nn.BatchNorm2d(32)
        self.bn3_R = nn.BatchNorm2d(64)
        # self.fc1_R = nn.Linear(256, 128) # for patch size 16
        self.fc1_R = nn.Linear(1024, 128) # for patch size 32
        # self.fc1_R = nn.Linear(4096, 128) # for patch size 64
        # self.fc1_R = nn.Linear(16384, 128) # for patch size 128

        self.fc1_1 = nn.Linear(128 * 2, 1)  # left and right FC
        
    def forward(self, xL,xR):

        # xL: Left image
        # xR: right image

        x_distort_L = xL.view(-1, xL.size(-3), xL.size(-2), xL.size(-1))
        x_distort_R = xR.view(-1, xR.size(-3), xR.size(-2), xR.size(-1))

        Out_L = F.relu(self.conv1L(x_distort_L)) 
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn1_L(Out_L)

        Out_L = F.relu(self.conv2L(Out_L))
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn2_L(Out_L)


        Out_L = F.relu(self.conv3L(Out_L))
        Out_L = F.relu(self.conv4L(Out_L))

        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn3_L(Out_L)

        
        Out_L = Out_L.view(-1, self.num_flat_features(Out_L))
        Out_L = F.relu(self.fc1_L(Out_L))
        
        Out_R = F.relu(self.conv1R(x_distort_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn1_R(Out_R)

        Out_R = F.relu(self.conv2R(Out_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn2_R(Out_R)
      
        Out_R = F.relu(self.conv3R(Out_R))
        Out_R = F.relu(self.conv4R(Out_R))
        

        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn3_R(Out_R)
        
        Out_R = Out_R.view(-1, self.num_flat_features(Out_R))
        Out_R = F.relu(self.fc1_R(Out_R))

        cat_total = F.relu(torch.cat((Out_L, Out_R), dim=1)) # concatenate left and right features

        Global_Quality = F.relu(self.fc1_1(cat_total))
        return Global_Quality


    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

class QNet_MixerWOD(nn.Module): # Proposed QNet_Mixer model without distortion maps
    def __init__(self, image_size = (32, 32), channels = 1, patch_size = 8, dim = 128, depth = 8, num_classes = 1):
        super(QNet_MixerWOD, self).__init__()
        self.il_mixer_model = MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
        self.ir_mixer_model = MLPMixer(
            image_size = image_size,
            channels = channels,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            num_classes = num_classes
        )
        self.il_mixer_model = torch.nn.Sequential(*list(self.il_mixer_model.children())[:-1]) # for left image
        self.ir_mixer_model = torch.nn.Sequential(*list(self.ir_mixer_model.children())[:-1]) # for right image

        self.fc = nn.Linear(dim*2, num_classes) # FC for concatenated left and right


    def forward(self, il, ir):

        # iL: Left image
        # iR: right image

        il = il.view(-1, il.size(-3), il.size(-2), il.size(-1))
        ir = ir.view(-1, ir.size(-3), ir.size(-2), ir.size(-1))

        ilf = self.il_mixer_model(il)
        irf = self.ir_mixer_model(ir)
        imf = torch.cat((ilf, irf), dim=1) #concatenated left and right features

        return self.fc(imf)
      