import torch
import numpy as np
import cv2
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
# im = cv2.imread('./carnivaldolls.bmp',0)
# im = torch.from_numpy(im).float()
# h,w = im.size()
# im = im.reshape([1,1,h,w])
# def Edge(im):
#     conv_op = nn.Conv2d(1,1,3,bias=False)
#     sobel_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=np.float32)
#     sobel_kernel = sobel_kernel.reshape((1,1,3,3))
#     conv_op.weight.data = torch.from_numpy(sobel_kernel)
#     edge_x = conv_op(Variable(im))
#     edge_x = edge_x.squeeze()
#     return edge_x

# edge = Edge(im)

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        Laplacian = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        Laplacian_3C = torch.Tensor(1, 3, 3, 3)
        Laplacian_3C[:, 0:3, :, :] = Laplacian
        self.conv_la = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_la.weight = torch.nn.Parameter(Laplacian_3C)
    def forward(self, X, Y):
        X_la = self.conv_la(X)
        Y_la = self.conv_la(Y)        
        # compute gradient of Y
        self.conv_la.train(False)
        loss = F.mse_loss(X_la, Y_la, size_average=True)

        return loss
if __name__=='__main__':
 	criterion = GradientLoss()
    # loss = criterion(output,target)
   
    
