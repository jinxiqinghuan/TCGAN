import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        # print(self.files[0])
        # if mode == "train":
        #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
        # if mode == "test":
        #     self.files = sorted(glob.glob(os.path.join(root, mode)))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)]).convert('L')
        # 三个方向的label设置
        img_path = self.files[index % len(self.files)]
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))


        # if np.random.random() < 0.5:
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")


        # img_A = np.array(img_A)
        # img_B = np.array(img_B)
    
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B, "path": img_path}

    def __len__(self):
        return len(self.files)




class ImageDataset_3c(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        # print(self.files[0])
        # if mode == "train":
        #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
        # if mode == "test":
        #     self.files = sorted(glob.glob(os.path.join(root, mode)))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        # 三个方向的label设置
        img_path = self.files[index % len(self.files)]
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))


        # if np.random.random() < 0.5:
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")


        # img_A = np.array(img_A)
        # img_B = np.array(img_B)
    
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B, "path": img_path}

    def __len__(self):
        return len(self.files)



class ImageDataset_Way2(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.im1_list = sorted(glob.glob(os.path.join(root, "images") + "/*.*"))
        self.im2_list = sorted(glob.glob(os.path.join(root, "gts") + "/*.*"))
        assert len(self.im1_list) == len(self.im2_list)
        
        # print(self.files[0])
        # if mode == "train":
        #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
        # if mode == "test":
        #     self.files = sorted(glob.glob(os.path.join(root, mode)))
    def __getitem__(self, index):

        im1 = Image.open(self.im1_list[index]).convert('RGB')
        im2 = Image.open(self.im2_list[index]).convert('RGB')
        
        
        # 三个方向的label设置
        # img_path = self.files[index % len(self.files)]
        # if np.random.random() < 0.5:
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB"
    
        img_A = self.transform(im1)
        img_B = self.transform(im2)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.im1_list)



class ImageDataset_Multi_Task(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.root = root
        self.mode = mode
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        # print(self.files[0])
        # if mode == "train":
        #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
        # if mode == "test":
        #     self.files = sorted(glob.glob(os.path.join(root, mode)))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])#.convert('RGB')
        # 三个方向的label设置
        img_path = self.files[index % len(self.files)]
        if "xy_pet_ct" in img_path:
            label = 0
        if "xz_pet_ct" in img_path:
            label = 1
        if "yz_pet_ct" in img_path:
            label = 2
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))


        # if np.random.random() < 0.5:
        #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
        #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")


        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B, "label": label}

    def __len__(self):
        return len(self.files)




















# import glob
# import random
# import os
# import numpy as np

# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as transforms


# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None, mode="train"):
#         self.transform = transforms.Compose(transforms_)

#         self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
#         # print(self.files[0])
#         # if mode == "train":
#         #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
#         # if mode == "test":
#         #     self.files = sorted(glob.glob(os.path.join(root, mode)))

#     def __getitem__(self, index):
#         # print(index)

#         img = Image.open(self.files[index % len(self.files)]) # 3.convert('RGB')
#         w, h = img.size
#         img_A = img.crop((0, 0, w / 2, h))
#         img_B = img.crop((w / 2, 0, w, h))


#         # if np.random.random() < 0.5:
#         #     img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
#         #     img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        

#         img_A = self.transform(img_A)
#         img_B = self.transform(img_B)

#         return {"A": img_A, "B": img_B}

#     def __len__(self):
#         return len(self.files)


# class ImageDataset_Multi_Task(Dataset):
#     def __init__(self, root, transforms_=None, mode="train"):
#         self.root = root
#         self.mode = mode
#         self.transform = transforms.Compose(transforms_)

#         self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
#         # print(self.files[0])
#         # if mode == "train":
#         #     self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
#         # if mode == "test":
#         #     self.files = sorted(glob.glob(os.path.join(root, mode)))

#     def __getitem__(self, index):

#         img = Image.open(self.files[index % len(self.files)]).convert('RGB')
#         # 三个方向的label设置
#         img_path = self.files[index % len(self.files)]
#         if "xy_pet_ct" in img_path:
#             label = 0
#         if "xz_pet_ct" in img_path:
#             label = 1
#         if "yz_pet_ct" in img_path:
#             label = 2
#         w, h = img.size
#         img_A = img.crop((0, 0, w / 2, h))
#         img_B = img.crop((w / 2, 0, w, h))


#         if np.random.random() < 0.5:
#             img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
#             img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")


#         img_A = self.transform(img_A)
#         img_B = self.transform(img_B)

#         return {"A": img_A, "B": img_B, "label": label}

#     def __len__(self):
#         return len(self.files)
