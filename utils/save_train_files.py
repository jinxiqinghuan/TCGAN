#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: 保存训练的相关文件 
@Date:2021/12/01 20:33:42
@Author: ljt
'''

import os
import time
sys_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
exp_name = "{}".format(time.strftime("%Y-%m-%d", time.localtime()))
description = "transformV3 效果都不如原始模型，特此归档"
home_path = "/root/cloud_hard_drive/project/pet2ct"
save_path = "/root/cloud_hard_drive/project/pet2ct/exp_data/{}".format(exp_name)

os.makedirs(save_path, exist_ok=False)

os.system('echo {} >> {}{}'.format(sys_time, save_path, "/description.txt"))
os.system('echo {} >> {}{}'.format(description, save_path, "/description.txt"))

os.chdir(home_path)
os.system('pwd')
os.system("mv images runs saved_models {}".format(save_path))
os.system("cp train.py {}".format(save_path))
os.system('du -sh {}'.format(save_path))


