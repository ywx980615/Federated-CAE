# Federated-CAE
This project combined federated learning and CAE encoder. 

Federated learning code credit to https://github.com/shaoxiongji/federated-learning 

CAE code credit to https://github.com/alexandru-dinu/cae

PSNR code credit to https://blog.csdn.net/u010886794/article/details/84784453

SSIM code credit to https://blog.csdn.net/weixin42096901/article/details/90172534


## Requirements
Python==3.8.3 other toolkit See [requirement.txt](https://github.com/ywx980615/Federated-CAE/blob/master/requirements.txt)

## Dataset
Training Dataset is [Youtube frame](https://drive.google.com/open?id=1wbwkpz38stSFMwgEKhoDCQCMiLLFVC4T)

Testing Dataset is [Kodak24](http://r0k.us/graphics/kodak/) 


## Run
First need to seperate the dataset by using [cut.ipynb](https://github.com/ywx980615/Federated-CAE/blob/master/utils/cut.ipynb)

Then need to set the configure in [train.yaml](https://github.com/ywx980615/Federated-CAE/blob/master/configs/train.yaml) and [test.yaml](https://github.com/ywx980615/Federated-CAE/blob/master/configs/test.yaml).

Running the [main.py](https://github.com/ywx980615/Federated-CAE/blob/master/main.py) and [test.py](https://github.com/ywx980615/Federated-CAE/blob/master/src/test.py) to train and test the model.

Draw the loss rate epoch plot by [draw.ipynb](https://github.com/ywx980615/Federated-CAE/blob/master/utils/draw.ipynb)

Test the compression effect by [PSNR.py](https://github.com/ywx980615/Federated-CAE/blob/master/utils/PSNR.py) [MSSIM.py](https://github.com/ywx980615/Federated-CAE/blob/master/utils/MSSIM.py) 


## Result
The CAE result stored in [here](https://github.com/ywx980615/Federated-CAE/tree/master/CAE_OUT), federated combine CAE result stored in [here](https://github.com/ywx980615/Federated-CAE/tree/master/FL_CAE_out).

![](https://github.com/ywx980615/Federated-CAE/blob/master/SHOW.png)

