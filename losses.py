"""
Loss functions for low light image enhancement and denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from skimage.metrics import structural_similarity as ssim
import numpy as np

class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss"""
    
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def _gaussian(self, window_size, sigma=1.5):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
        
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    
    def __init__(self, feature_layers=[3, 8, 15, 22]):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG19
        vgg = torchvision.models.vgg19(pretrained=True).features
        
        self.features = nn.ModuleList()
        self.feature_layers = feature_layers
        
        for i, layer in enumerate(vgg):
            if i in feature_layers:
                self.features.append(layer)
            if i == max(feature_layers):
                break
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred, target):
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        
        pred_features = []
        target_features = []
        
        for layer in self.features:
            pred_norm = layer(pred_norm)
            target_norm = layer(target_norm)
            pred_features.append(pred_norm)
            target_features.append(target_norm)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += F.mse_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)

class GradientLoss(nn.Module):
    """Gradient loss to preserve edges"""
    
    def __init__(self):
        super(GradientLoss, self).__init__()
        
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, pred, target):
        # Convert to grayscale for gradient computation
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        
        target_grad_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        
        # Compute gradient magnitude
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        return F.mse_loss(pred_grad, target_grad)

class CharbonnierLoss(nn.Module):
    """Charbonnier loss (smooth L1)"""
    
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return loss.mean()

class TotalVariationLoss(nn.Module):
    """Total Variation loss for smoothness"""
    
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
    
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class CombinedLoss(nn.Module):
    """Combined loss function for low light enhancement"""
    
    def __init__(self, l1_weight=1.0, ssim_weight=0.1, perceptual_weight=0.1, 
                 gradient_weight=0.1, tv_weight=0.01):
        super(CombinedLoss, self).__init__()
        
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.gradient_weight = gradient_weight
        self.tv_weight = tv_weight
        
        # Initialize loss functions
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss()
        self.gradient_loss = GradientLoss()
        self.tv_loss = TotalVariationLoss()
        self.charbonnier_loss = CharbonnierLoss()
    
    def forward(self, pred, target):
        # Basic losses
        l1 = self.l1_loss(pred, target)
        charbonnier = self.charbonnier_loss(pred, target)
        
        # Advanced losses
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        gradient = self.gradient_loss(pred, target)
        tv = self.tv_loss(pred)
        
        # Combine losses
        total_loss = (self.l1_weight * l1 + 
                     self.ssim_weight * ssim + 
                     self.perceptual_weight * perceptual + 
                     self.gradient_weight * gradient + 
                     self.tv_weight * tv)
        
        return {
            'total': total_loss,
            'l1': l1,
            'charbonnier': charbonnier,
            'ssim': ssim,
            'perceptual': perceptual,
            'gradient': gradient,
            'tv': tv
        }

def create_loss_function(config):
    """Create and return the loss function"""
    return CombinedLoss(
        l1_weight=config.L1_WEIGHT,
        ssim_weight=config.SSIM_WEIGHT,
        perceptual_weight=config.PERCEPTUAL_WEIGHT
    )
