import numpy as np
from numpy.core.numerictypes import maximum_sctype



def mmm_value(img, print_=False):
    """
    @description: 输入图片或者图像数组，返回最大值，最小值和平均值
    ---------
    @param:
    -------
    @Returns:
    -------
    """
    
    
    image_array = np.array(img)
    
    max_value = image_array.max()
    min_value = image_array.min()
    mean_value = image_array.mean()
    if print_:
        print("最大值：%f, 最小值：%f, 平均值：%f " %(max_value, min_value, mean_value))
    return max_value, min_value, mean_value

