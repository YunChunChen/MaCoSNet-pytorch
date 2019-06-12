from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
from geotnf.transformation import GeometricTnf


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature,norm)


class FeatureExtraction(torch.nn.Module):

    def __init__(self, 
                 train_fe=False, 
                 feature_extraction_cnn='resnet-101', 
                 normalization=True, 
                 last_layer='', 
                 use_cuda=True):

        super(FeatureExtraction, self).__init__()

        self.normalization = normalization

        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1',
                                     'bn1',
                                     'relu',
                                     'maxpool',
                                     'layer1',
                                     'layer2',
                                     'layer3',
                                     'layer4']
            if last_layer=='':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1,
                                  self.model.bn1,
                                  self.model.relu,
                                  self.model.maxpool,
                                  self.model.layer1,
                                  self.model.layer2,
                                  self.model.layer3,
                                  self.model.layer4]
            
            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx+1])

        if feature_extraction_cnn == 'resnet101_v2':
            self.model = models.resnet101(pretrained=True)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.children())[:-3])

        if not train_fe:
            # freeze parameters
            for param in self.model.parameters():
                param.requires_grad = False

        if use_cuda:
            self.model = self.model.cuda()
        
    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization:
            features = featureL2Norm(features)
        return features
    

class FeatureCorrelation(torch.nn.Module):

    def __init__(self,
                 shape='3D',
                 normalization=True):

        super(FeatureCorrelation, self).__init__()

        self.normalization = normalization
        self.shape=shape
        self.ReLU = nn.ReLU()
    

    def forward(self, feature_A, feature_B):

        b,c,h,w = feature_A.size()

        if self.shape=='3D':
            # reshape features for matrix multiplication
            feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
            feature_B = feature_B.view(b,c,h*w).transpose(1,2)
            # perform matrix mult.
            feature_mul = torch.bmm(feature_B,feature_A)

            # batch x (h_A x w_A) x h_B x w_B
            correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2) 
            

        elif self.shape=='4D':
            # reshape features for matrix multiplication
            feature_A = feature_A.view(b,c,h*w).transpose(1,2) # size [b,c,h*w]
            feature_B = feature_B.view(b,c,h*w) # size [b,c,h*w]
            # perform matrix mult.
            feature_mul = torch.bmm(feature_A,feature_B)
            # indexed [batch, row_A, col_A, row_B, col_B]
            correlation_tensor = feature_mul.view(b,h,w,h,w).unsqueeze(1)
        
        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))
            
        return correlation_tensor


class FeatureRegression(nn.Module):

    def __init__(self, 
                 output_dim=6, 
                 use_cuda=True, 
                 batch_normalization=True, 
                 kernel_sizes=[7,5], 
                 channels=[128,64] ,
                 feature_size=15):

        super(FeatureRegression, self).__init__()

        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):

            if i==0:
                ch_in = feature_size*feature_size
            else:
                ch_in = channels[i-1]

            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size, padding=0))

            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*nn_modules)        
        self.linear = nn.Linear(ch_out * k_size * k_size, output_dim)

        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    
class CNNGeometric(nn.Module):

    def __init__(self, output_dim=6, 
                 feature_extraction_cnn='vgg', 
                 feature_extraction_last_layer='',
                 return_correlation=False,  
                 fr_feature_size=15,
                 fr_kernel_sizes=[7,5],
                 fr_channels=[128,64],
                 feature_self_matching=False,
                 normalize_features=True, 
                 normalize_matches=True, 
                 batch_normalization=True, 
                 train_fe=False,use_cuda=True):
        
        super(CNNGeometric, self).__init__()

        self.use_cuda = use_cuda
        self.feature_self_matching = feature_self_matching
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=self.use_cuda)
        
        self.FeatureCorrelation = FeatureCorrelation(shape='3D',normalization=normalize_matches)        
        

        self.FeatureRegression = FeatureRegression(output_dim,
                                                   use_cuda=self.use_cuda,
                                                   feature_size=fr_feature_size,
                                                   kernel_sizes=fr_kernel_sizes,
                                                   channels=fr_channels,
                                                   batch_normalization=batch_normalization)


        self.ReLU = nn.ReLU(inplace=True)
    
    # used only for foward pass at eval and for training with strong supervision
    def forward(self, tnf_batch): 
        # feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['image_A'])
        feature_B = self.FeatureExtraction(tnf_batch['image_B'])
        # feature correlation
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        # regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)
        
        if self.return_correlation:
            return (theta,correlation)
        else:
            return theta


class WeakMatchNet(CNNGeometric):

    def __init__(self, 
                 fr_feature_size=15,
                 fr_kernel_sizes=[7,5],
                 fr_channels=[128,64],
                 feature_extraction_cnn='vgg', 
                 feature_extraction_last_layer='',
                 return_correlation=False,                  
                 normalize_features=True,
                 normalize_matches=True, 
                 batch_normalization=True, 
                 train_fe=False,
                 use_cuda=True,
                 s1_output_dim=6,
                 s2_output_dim=18):

        super(WeakMatchNet, self).__init__(output_dim=s1_output_dim, 
                                           fr_feature_size=fr_feature_size,
                                           fr_kernel_sizes=fr_kernel_sizes,
                                           fr_channels=fr_channels,
                                           feature_extraction_cnn=feature_extraction_cnn,
                                           feature_extraction_last_layer=feature_extraction_last_layer,
                                           return_correlation=return_correlation,
                                           normalize_features=normalize_features,
                                           normalize_matches=normalize_matches,
                                           batch_normalization=batch_normalization,
                                           train_fe=train_fe,
                                           use_cuda=use_cuda)

        if s1_output_dim==6:
            self.geoTnf = GeometricTnf(geometric_model='affine', 
                                       use_cuda=use_cuda)
        else:
            tps_grid_size = np.sqrt(s2_output_dim/2)
            self.geoTnf = GeometricTnf(geometric_model='tps', 
                                       tps_grid_size=tps_grid_size, 
                                       use_cuda=use_cuda)
        
        self.FeatureRegression2 = FeatureRegression(output_dim=s2_output_dim,
                                                    use_cuda=use_cuda,
                                                    feature_size=fr_feature_size,
                                                    kernel_sizes=fr_kernel_sizes,
                                                    channels=fr_channels,
                                                    batch_normalization=batch_normalization)        
 
    def forward(self, batch, training=False): 

        if not training:
            """ Affine """
            img_A, img_B = batch['image_A'], batch['image_B']
            f_A, f_B = self.get_feature2(img_A, img_B)
            corr_AB = self.get_corr1(f_A, f_B)
            aff_AB = self.get_affine1(corr_AB)

            """ Tps """
            img_Awrp = self.warp_image(img_A, aff_AB)
            f_Awrp = self.get_feature1(img_Awrp)
            corr_Awrp_B = self.get_corr1(f_Awrp, f_B) 
            tps_Awrp_B = self.get_tps1(corr_Awrp_B)

            return aff_AB, tps_Awrp_B

        else:
            """ Affine """
            img_A, img_B, img_C = batch['image_A'], batch['image_B'], batch['image_C']

            f_A, f_B, f_C = self.get_feature3(img_A, img_B, img_C)

            corr_AB, corr_BA = self.get_corr2(f_A, f_B)
            corr_BC, corr_CB = self.get_corr2(f_B, f_C)
            corr_CA, corr_AC = self.get_corr2(f_C, f_A)

            aff_AB, aff_BA = self.get_affine2(corr_AB, corr_BA)
            aff_BC, aff_CB = self.get_affine2(corr_BC, corr_CB)
            aff_CA, aff_AC = self.get_affine2(corr_CA, corr_AC)

            aff_dict = {
                'aff_AB': aff_AB,
                'aff_BA': aff_BA,
                'aff_BC': aff_BC,
                'aff_CB': aff_CB,
                'aff_CA': aff_CA,
                'aff_AC': aff_AC,
            }

            corr_dict = {
                'corr_AB': corr_AB,
                'corr_BA': corr_BA,
                'corr_BC': corr_BC,
                'corr_CB': corr_CB,
                'corr_CA': corr_CA,
                'corr_AC': corr_AC,
            }

            """ Tps """
            img_Awrp_B = self.warp_image(img_A, aff_AB) # should better align img_B
            img_Bwrp_A = self.warp_image(img_B, aff_BA) # should better align img_A

            img_Bwrp_C = self.warp_image(img_B, aff_BC) # should better align img_C
            img_Cwrp_B = self.warp_image(img_C, aff_CB) # should better align img_B

            img_Cwrp_A = self.warp_image(img_C, aff_CA) # should better align img_A
            img_Awrp_C = self.warp_image(img_A, aff_AC) # should better align img_C

            f_Awrp_B, f_Bwrp_A = self.get_feature2(img_Awrp_B, img_Bwrp_A)
            f_Bwrp_C, f_Cwrp_B = self.get_feature2(img_Bwrp_C, img_Cwrp_B)
            f_Cwrp_A, f_Awrp_C = self.get_feature2(img_Cwrp_A, img_Awrp_C)

            corr_Awrp_B = self.get_corr1(f_Awrp_B, f_B)
            corr_Bwrp_A = self.get_corr1(f_Bwrp_A, f_A)

            corr_Bwrp_C = self.get_corr1(f_Bwrp_C, f_C)
            corr_Cwrp_B = self.get_corr1(f_Cwrp_B, f_B)

            corr_Awrp_C = self.get_corr1(f_Awrp_C, f_C)
            corr_Cwrp_A = self.get_corr1(f_Cwrp_A, f_A)

            tps_Awrp_B, tps_Bwrp_A = self.get_tps2(corr_Awrp_B, corr_Bwrp_A)
            tps_Bwrp_C, tps_Cwrp_B = self.get_tps2(corr_Bwrp_C, corr_Cwrp_B)
            tps_Cwrp_A, tps_Awrp_C = self.get_tps2(corr_Cwrp_A, corr_Awrp_C)

            tps_dict = {
                'tps_Awrp_B': tps_Awrp_B,
                'tps_Bwrp_A': tps_Bwrp_A,
                'tps_Bwrp_C': tps_Bwrp_C,
                'tps_Cwrp_B': tps_Cwrp_B,
                'tps_Cwrp_A': tps_Cwrp_A,
                'tps_Awrp_C': tps_Awrp_C,
            }

            return aff_dict, tps_dict, corr_dict

    def get_feature1(self, img_A): 
        f_A = self.FeatureExtraction(img_A)
        return f_A

    def get_feature2(self, img_A, img_B): 
        f_A = self.get_feature1(img_A)
        f_B = self.get_feature1(img_B)
        return f_A, f_B

    def get_feature3(self, img_A, img_B, img_C): 
        f_A = self.get_feature1(img_A)
        f_B = self.get_feature1(img_B)
        f_C = self.get_feature1(img_C)
        return f_A, f_B, f_C

    def get_corr1(self, f_A, f_B): 
        corr_AB = self.FeatureCorrelation(f_A, f_B) # (h_A x w_A) x h_B x w_B -> for T_AB
        return corr_AB

    def get_corr2(self, f_A, f_B): 
        corr_AB = self.get_corr1(f_A, f_B)
        corr_BA = self.get_corr1(f_B, f_A)
        return corr_AB, corr_BA

    def get_affine1(self, corr_AB): 
        aff_AB = self.FeatureRegression(corr_AB) # warp img_A to align img_B -> T_AB
        return aff_AB

    def get_affine2(self, corr_AB, corr_BA): 
        aff_AB = self.get_affine1(corr_AB)
        aff_BA = self.get_affine1(corr_BA)
        return aff_AB, aff_BA

    def get_tps1(self, corr_AB): 
        tps_AB = self.FeatureRegression2(corr_AB)
        return tps_AB

    def get_tps2(self, corr_AB, corr_BA): 
        tps_AB = self.FeatureRegression2(corr_AB)
        tps_BA = self.FeatureRegression2(corr_BA)
        return tps_AB, tps_BA

    def warp_image(self, img, theta): 
        warped_img = self.geoTnf(img, theta)
        return warped_img


def conv(in_channel, 
         out_channel, 
         kernel_size=3, 
         stride=1, 
         dilation=1, 
         bias=False, 
         transposed=False):

    if transposed:
        layer = nn.ConvTranspose2d(in_channel, 
                                   out_channel, 
                                   kernel_size=kernel_size, 
                                   stride=stride, 
                                   padding=1, 
                                   output_padding=1, 
                                   dilation=dilation, 
                                   bias=bias)
        w = torch.Tensor(kernel_size, kernel_size)
        center = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
        for y in range(kernel_size):
            for x in range(kernel_size):
                w[y, x] = (1 - abs((x - center) / stride)) * (1 - abs((y - center) / stride))
        layer.weight.data.copy_(w.div(in_channel).repeat(out_channel, in_channel, 1, 1))
    else:
        padding = (kernel_size + 2 * (dilation - 1)) // 2
        layer = nn.Conv2d(in_channel, 
                          out_channel, 
                          kernel_size=kernel_size, 
                          stride=stride, 
                          padding=padding, 
                          dilation=dilation, 
                          bias=bias)
    if bias:
        nn.init.constant(layer.bias, 0)
    return layer


def bn(channel):
    layer = nn.BatchNorm2d(channel)
    nn.init.constant(layer.weight, 1)
    nn.init.constant(layer.bias, 0)
    return layer


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = conv(1249, 512, stride=2, transposed=True)
        self.bn1 = bn(512)
        self.relu1 = nn.ReLU()
        self.deconv2 = conv(512, 256, stride=2, transposed=True)
        self.bn2 = bn(256)
        self.relu2 = nn.ReLU()
        self.deconv3 = conv(256, 64, stride=2, transposed=True)
        self.bn3 = bn(64)
        self.relu3 = nn.ReLU()
        self.deconv4 = conv(64, 1, stride=2, transposed=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, features):
        _, x3, x2, x1 = features
        deconv1 = self.deconv1(data)
        bn1 = self.bn1(deconv1)
        relu1 = self.relu1(bn1)
        deconv2 = self.deconv2(relu1+x3)
        bn2 = self.bn2(deconv2)
        relu2 = self.relu2(bn2)
        deconv3 = self.deconv3(relu2+x2)
        bn3 = self.bn3(deconv3)
        relu3 = self.relu3(bn3)
        deconv4 = self.deconv4(relu3+x1)

        return deconv4


class WeakCosegNet(CNNGeometric):

    def __init__(self, 
                 train_fe=False,
                 normalize_features=True,
                 normalize_matches=True,
                 feature_extraction_cnn='vgg', 
                 feature_extraction_last_layer='',
                 use_cuda=True):

        super(WeakCosegNet, self).__init__(use_cuda=use_cuda)

        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
                                                   feature_extraction_cnn=feature_extraction_cnn,
                                                   last_layer=feature_extraction_last_layer,
                                                   normalization=normalize_features,
                                                   use_cuda=use_cuda)

        self.FeatureCorrelation = FeatureCorrelation(shape='3D',
                                                     normalization=normalize_matches) 

        self.Decoder = Decoder()

        if use_cuda:
            self.Decoder = self.Decoder.cuda()


    def forward(self, batch): 

        img_A, img_B = batch['image_A'], batch['image_B']

        features_A = self.get_features(img_A)
        festures_B = self.get_features(img_B)

        f_A = features_A[-1]
        f_B = features_B[-1]

        corr_AB, corr_BA = self.get_corr(f_A, f_B)

        C_A = torch.cat((f_A, corr_BA), dim=1)
        C_B = torch.cat((f_B, corr_AB), dim=1)

        mask_A = self.get_mask(C_A, features_A)
        mask_B = self.get_mask(C_B, features_B)

        mask_dict = {
            'mask_A': mask_A,
            'mask_B': mask_B,
        }

        return mask_dict

    def get_feature(self, data):
        x = data
        f = []
        for idx, module in self.FeatureExtraction.model._modules.items():
            x = module(x)
            if idx in ['2', '4', '5', '6']:
                print('size:', x.size())
                f.append(x)
        return f

    def get_corr(self, f_A, f_B): 
        corr_AB = self.FeatureCorrelation(f_A, f_B) # (h_A x w_A) x h_B x w_B
        corr_BA = self.FeatureCorrelation(f_B, f_A) # (h_B x w_B) x h_A x w_A
        return corr_AB, corr_BA

    def get_mask(self, C, features): 
        mask = self.Decoder(C, features)
        return mask
