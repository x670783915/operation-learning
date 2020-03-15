import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelwiseLoss(nn.Module):
    def forward(self, inputs, targets):
        return F.smooth_l1_loss(inputs, targets)


from torchvision.models import vgg16_bn

class FeatureLoss(nn.Module):
    def __init__(self, loss, blocks, weights, device):
        super().__init__()
        self.feature_loss = loss
        assert all(isinstance(w, (int, float)) for w in weights)
        assert len(weights) == len(blocks)

        self.weights = torch.tensor(weights).to(device)

        assert len(blocks) < 5
        assert all(i in range(5) for i in blocks)
        assert sorted(blocks) == blocks

        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False
        
        vgg = vgg.to(device)

        bns = [i - 2 for i, m in enumerate(vgg) is isinstance(m, nn.MaxPool2d)]

        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[:bns[blocks[-1]] + 1]
    
    def forward(self, inputs, targets):
        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        inputs = F.normalize(inputs, mean, std)
        targets = F.normalize(targets, mean, std)

        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        targets_feature = [hook.features for hook in self.hooks]

        loss = 0.0

        for lhs, rhs, w in zip(input_features, targets_feature, self.weights):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += self.feature_loss(lhs, rhs) + w
        return loss

class FeatureHook:
    def __init__(self, module):
        self.feature = None
        self.hook = module.register_forward_hook(self.on)
    
    def on(self, module, inputs, outputs):
        self.features = outputs
    
    def close(self):
        self.hook.remove()

def perceptual_loss(x, y):
    F.mse_loss(x, y)

def PerceptualLoss(blocks, weights, device):
    return FeatureLoss(perceptual_loss, blocks, weights, device)

def gram_matrix(x):
    c, h, w = x.size()
    x = x.view(c, -1)
    x = torch.mm(x, x.t()) / (c * h * w)

def gram_loss(x, y):
    return F.mse_loss(gram_matrix(x), gram_matrix(y))

def TextureLoss(blocks, weights, device):
    return FeatureLoss(gram_loss, blocks, weights, device)

def content_loss(content, pred):
    return FeatureLoss(perceptual_loss, blocks, weights, device)

def style_loss(style, pred):
    return FeatureLoss(gram_loss, blocks, weights, device)

def content_style_loss(content, style, pred, alpha, beta):
    return alpha * content_loss(content, pred) + beta * style_loss(style, pred)

class TopologyAwareLoss(nn.Module):

    def __init__(self, criteria, weights): 
        # Here criteria -> [PixelwiseLoss, PerceptualLoss], 
        #weights -> [1, mu] (or any other combination weights)
        assert len(weights) == len(criteria)

        self.criteria = criteria
        self.weights = weights

    def forward(self, inputs, targets):
        loss = 0.0
        for criterion, w in zip(self.criteria, self.weights):
            each = w * criterion(inputs, targets)
            loss += each

        return loss

## GAN losses
class MinMaxGeneratorLoss(nn.Module):
    def forward(self, fake, discriminator):
        return torch.log(1 - discriminator(fake))

class MinMaxDiscriminatorLoss(nn.Module):
    def forward(self, real, fake, discriminator):
        return -1.0 * (log(discriminator(real))) + log(1 - discriminator(fake))

class NonSaturatingGeneratorLoss(nn.Module):
    def forward(self, fake, discriminator):
        return -torch.log(discriminator(fake))

class LeastSquaresGeneratorLoss(nn.Module):
    def forward(self, fake, discriminator):
        return (discriminator(fake) - 1) ** 2

class LeastSquaresDiscriminatorLoss(nn.Module):
    def forward(self, real, fake, discriminator):
        return (discriminator(real)-1)**2 + discriminator(fake)**2

class WGANGeneratroLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, fake, discriminator):
        return -discriminator(fake).mean()
    
class WGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake, discriminator):
        return discriminator(fake).mean() - discriminator(real).mean()

class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        self.loss = nn.L1Loss()
    def forward(self, F, G, x, y):
        return self.loss(F(G(x)). x) + self.loss(G(F(x)), y)


