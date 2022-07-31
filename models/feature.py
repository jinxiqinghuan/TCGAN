import torch
import torchvision.models as models
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
# device = torch.device("cpu")
vgg = models.vgg19(pretrained=True).features.to(device)

# vgg = torch.nn.DataParallel(vgg)

# 设置参数不需要优化了
for param in vgg.parameters():
    param.requires_grad_(False)


def get_features(x, model, layers=None):
    """
    实现论文Gatys et al (2016)
    """

    # 选择提取特征和风格特征需要的特征层
    if layers is None:
        layers = {'0': 'conv1_1',
                '5':  'conv2_1',
                '10': 'conv3_1',
                '19': 'conv4_1',
                '21': 'conv4_2',
                '28': 'conv5_1'}
    features = {}

    x = torch.cat((x, x, x), 1)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    #print(features)
    return features

# https://blog.csdn.net/cheetah023/article/details/107686572
def gram_matrix(tensor, normalize=True):
    """
    格拉姆矩阵
    """
    n, c, h, w = tensor.size()
    tensor1 = tensor.view(n, c, h*w)
    tensor2 = tensor1.permute(0, 2, 1)
    gram = torch.matmul(tensor1, tensor2)
    if normalize:
        gram = gram / (c*h*w)
    return gram 


def feature_loss(x, x_hat, batch_size = 16):
    style_weights = {
                'conv1_1': 1.,
                'conv2_1': 0.75,
                'conv3_1': 0.2,
                'conv4_1': 0.2,
                'conv4_2': 0.2,
                'conv5_1': 0.2
                }

    # 内容和风格权重
    content_weight = 1  # alpha
    style_weight = 1e6  # beta

    x_features = get_features(x, vgg)
    x_hat_features = get_features(x_hat, vgg)

    x_grams = {layer: gram_matrix(x_features[layer]) for layer in x_features}  

    content_loss = torch.mean((x_hat_features['conv4_2'] - x_features['conv4_2'])**2)

    style_loss  = 0

    for layer in style_weights:
        # get the "target" style representation for the layer
        x_hat_feature = x_hat_features[layer]
        _, d, h, w = x_hat_feature.shape

        # 计算目标图像的格拉姆矩阵
        x_hat_gram = gram_matrix(x_hat_feature)

        # Get the "style" style represtation
        x_gram = x_grams[layer]
        # 计算一层的风格损失， 适当加权
        layer_style_loss = style_weights[layer] * torch.mean((x_hat_gram - x_gram) ** 2)
        style_loss += layer_style_loss / ((d * h * w)**2)
    
    feature_loss = content_weight * content_loss + style_weight * style_loss

    return feature_loss


    # x_gram = {layer: gram_matrix(x_features[layer]) for layer in x_features}
    # for i in x_gram.values():
    #     print(i.shape)
    
    

    


    # # target = x_hat.clone().requires_grad_(True).to(device)
    # for i,j in zip(x_features.values(), x_hat_features.values()):
    #     loss = loss_style(i, j)
    #     # tmp = loss_style(i, j)
    # return loss  




def main():
    x = torch.randn((6, 1, 255, 255)).to(device)
    x_hat = torch.randn((6, 1, 255, 255)).to(device)
    tmp = feature_loss(x, x_hat, batch_size=6)
    print(tmp)

    # f = Feature(x, x_hat)
    # tmp = f.style_loss()
    # print(tmp)


if __name__ == '__main__':
    main()


# class Feature():
#     def __init__(self, x, x_hat):
#         super(Feature, self).__init__()




# class Feature():
#     def __init__(self, x, x_hat):
#         super(Feature, self).__init__()
#         self.x = x
#         self.x_hat = x_hat

#         self.vgg = models.vgg19(pretrained=True).features.to(device)

#         # 设置参数不需要优化了
#         for param in self.vgg.parameters():
#             param.requires_grad_(False)


#         self.loss_style = torch.nn.MSELoss()

#     def get_features(x, model, layers=None):
#         """
#         实现论文Gatys et al (2016)
#         """

#         # 选择提取特征和风格特征需要的特征层
#         if layers is None:
#             layers = {'0': 'conv1_1',
#                     '5':  'conv2_1',
#                     '10': 'conv3_1',
#                     '19': 'conv4_1',
#                     '21': 'conv4_2',
#                     '28': 'conv5_1'}
#         features = {}
#         for name, layer in model._modules.items():
#             x = layer(x)
#             if name in layers:
#                 features[layers[name]] = x
#         #print(features)
#         return features

#     def gram_matrix(tensor):
#         """
#         格拉姆矩阵
#         """
#         _, d, h, w = tensor.size()
#         tensor = tensor.view(d, h*w)
#         gram = torch.mm(tensor, tensor.t())
#         return gram 
    
#     def style_loss(self):
#         x_features = self.get_features(self.x, self.vgg)
#         x_hat_features = self.get_features(self.x_hat, self.vgg)
#         loss = 0
#         for i,j in zip(x_features.values(), x_hat_features.values()):
#             loss = self.loss_style(i, j)
#             print(loss)
#         return loss
