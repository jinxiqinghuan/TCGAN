import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class Soble_Net(nn.Module):
    def __init__(self):
        super(Soble_Net, self).__init__()
        self.sobel_conv = nn.Conv2d(3, 3, 3, padding=(0, 0), bias=False)
    def forward(self, x):
        x = self.sobel_conv(x)
        return x
sobel_net = Soble_Net()

conv_rgb_core_original = [
                        [[0,0,0],[0,1,0], [0,0,0],
                         [0,0,0],[0,0,0], [0,0,0],
                         [0,0,0],[0,0,0], [0,0,0]
                        ],
                        [[0,0,0],[0,0,0], [0,0,0],
                         [0,0,0],[0,1,0], [0,0,0],
                         [0,0,0],[0,0,0], [0,0,0]
                        ],
                        [[0,0,0],[0,0,0], [0,0,0],
                         [0,0,0],[0,0,0], [0,0,0],
                         [0,0,0],[0,1,0], [0,0,0]
                        ]]
conv_rgb_core_sobel = [
                        [[-1,-1,-1],[-1,8,-1], [-1,    -1,    -1],
                         [0,0,0],[0,0,0], [0,0,0],
                         [0,0,0],[0,0,0], [0,0,0]
                        ],
                        [[0,0,0],[0,0,0], [0,0,0],
                         [-1,-1,-1],[-1,8,-1], [-1,    -1,    -1],
                         [0,0,0],[0,0,0], [0,0,0]
                        ],
                        [[0,0,0],[0,0,0], [0,0,0],
                         [0,0,0],[0,0,0], [0,0,0],
                         [-1,-1,-1],[-1,8,-1], [-1,    -1,    -1],
                        ]]
conv_rgb_core_sobel_vertical = [
                        [[-1,0,1],[-2,0,2], [-1,    0,    1],
                         [0,0,0],[0,0,0], [0,0,0],
                         [0,0,0],[0,0,0], [0,0,0]
                        ],
                        [[0,0,0],[0,0,0], [0,0,0],
                         [-1,0,1],[-2,0,2], [-1,    0,    1],
                         [0,0,0],[0,0,0], [0,0,0]
                        ],
                        [[0,0,0],[0,0,0], [0,0,0],
                         [0,0,0],[0,0,0], [0,0,0],
                         [-1,0,1],[-2,0,2], [-1,    0,    1],
                        ]]
conv_rgb_core_sobel_horizontal = [
                        [[1,2,1],[0,0,0], [-1, -2, -1],
                         [0,0,0],[0,0,0], [0,0,0],
                         [0,0,0],[0,0,0], [0,0,0]
                        ],
                        [[0,0,0],[0,0,0], [0,0,0],
                         [1,2,1],[0,0,0], [-1, -2, -1],
                         [0,0,0],[0,0,0], [0,0,0]
                        ],
                        [[0,0,0],[0,0,0], [0,0,0],
                         [0,0,0],[0,0,0], [0,0,0],
                         [1,2,1],[0,0,0], [-1, -2, -1],
                        ]]


def sobel_model(model=sobel_net, kernel=conv_rgb_core_sobel):
    sobel_kernel = np.array(kernel, dtype='float32')
    sobel_kernel = sobel_kernel.reshape((3, 3, 3, 3))
    model.sobel_conv.weight.data = torch.from_numpy(sobel_kernel)
    return model


def sobel_test():
    input = torch.ones(1, 3, 256, 256)
    input.cuda()
    
    
    
    
if __name__ == '__main__':
    sobel_test()




# def main(net):
#     def sobel(net, kernel):
#         sobel_kernel = np.array(kernel,    dtype='float32')
#         sobel_kernel = sobel_kernel.reshape((3,    3,    3,    3))
#         net.conv1.weight.data = torch.from_numpy(sobel_kernel)
#         return net



#     net = sobel(net, conv_rgb_core_sobel)
#     fake_img, real_img = 0
#     fake_img, real_img = 0
#     fake_sobel_loss = net(fake_img)
#     real_sobel_loss = net(real_img)

#     sobel_loss = torch.abs(fake_sobel_loss, real_sobel_loss)


# params = list(net.parameters())
# img = cv2.imread(r"/mnt/4t/ljt/project/pet_ct_/img.png")
# input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# input_tensor = (input_img.astype(np.float32) - 127.5) / 128 # to [-1, 1]
# input_tensor = torch.Tensor(input_tensor).permute((2, 0, 1))
# print(input_tensor.shape)
# input_tensor = input_tensor.unsqueeze(0)
# print("input shape:", input_tensor.shape)
# sobel(net, conv_rgb_core_sobel)
# out = net(input_tensor).detach().numpy()[0].transpose([1,2,0])
# sobel(net, conv_rgb_core_sobel_vertical)
# out_v = net(input_tensor).detach().numpy()[0].transpose([1,2,0])
# sobel(net, conv_rgb_core_sobel_horizontal)
# out_h = net(input_tensor).detach().numpy()[0].transpose([1,2,0])
# print("out shape: {}, tensor:{}".format(out.shape, out))
# print(out.shape, out.max(), out.min())
# plt.figure()
# plt.figure()
# plt.subplot(1, 5, 1)
# input = input_tensor.numpy()[0].transpose((1,2,0))
# print(input.max(), input.min())
# plt.imshow(input_img)
# plt.subplot(1, 5, 2)
# print(out.max(), out.min())
# # out = np.sqrt(np.square(out))
# # out = out * 255.0 / out.max()
# # out = out.astype(np.uint8)
# # print(out.max(), out.min())
# plt.imshow(out)
# plt.subplot(1, 5, 3)
# out = np.abs(out_v)
# # out = out * 255.0 / out.max()
# # plt.imshow(out.astype(np.uint8))
# plt.imshow(out)
# plt.subplot(1, 5, 4)
# out = np.abs(out_h)
# # out = out * 255.0 / out.max()
# # plt.imshow(out.astype(np.uint8))
# plt.imshow(out)
# plt.subplot(1, 5, 5)
# out = np.sqrt(np.square(out_v) + np.square(out_h))
# # out = out * 255.0 / out.max()
# # plt.imshow(out.astype(np.uint8))
# plt.imshow(out)
# plt.show()



