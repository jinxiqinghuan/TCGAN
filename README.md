# TCGAN

The implemetation of the paper "TCGAN"

After the paper is accepted, we will continue to improve the content of the document, including the ideas of the paper and specific experimental methods.



## 1.Methods



![image-20220731214338901](README.assets/image-20220731214338901.png)



![image-20220731214554081](README.assets/image-20220731214554081.png)



## 2. Usage



### 2.1 Train

```shell
python train.py　--path /data/root --n_epochs 0 --ever_hum 2000 --dataset_name exp --batch_size 12 --lr 2e-4 --n_cpu 1 --img_height 256 --img_width 256 --channels 3 --checkpoint_interval 100 --lambda_pixel 50
```

### 2. Test

```shell
python test.py
```









