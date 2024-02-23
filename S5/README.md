## S5(Session 5: Structure the python modules properly)
>Simple network to train the MNIST dataset and understand the concept of convolutions, receptive feilds concepts.

### 1. model.py
> In this file the model architecure is defined.

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #

            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
----------------------------------------------------------------

> Total params: 593,200     
> Trainable params: 593,200     
> Non-trainable params: 0

### 2. utils.py
> This module have dataset loaders, training and testing functions amd other utility funstions for visualization.

### 3. S5.ipynb
> This python notebook will call the other module files and train the model.