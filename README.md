## Description

Based on the cutting-edge technologies related to **GAN**, such as makeup transfer, hairstyle transfer and virtual try-on, we have built a **virtual modeling room** **application** with complete functions and flexible operation, integrating various functions such as **virtual makeup**, **virtual hair change** and **virtual clothes try-on**. The application allows users to upload model photos freely and combine corresponding functions to meet the personalized needs of them for their modeling preview.

## Installation

We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). All dependencies for defining the environment are provided in `environment/environment.yaml`.

Run:

```
cd environment
conda env create -f environment.yml
```

You can download all weights from [Baidu Netdisk](https://pan.baidu.com/s/1HWfCEoo3Hen_NkuZ_-UavA) (extracted code: hw96) and put them in ```./checkpoint```.

## Getting Started

It's very easy to use this app. Just run ```python app.py``` to start the app, then you can enjoy your modeling time!

<img src="./images/virtual makeup.png" alt="image" style="zoom: 67%;" />

<img src="./images/virtual hair change.png" alt="image" style="zoom:67%;" />

<img src="./images/virtual makeup and hair change.png" alt="image" style="zoom:67%;" />

<img src="./images/virtual try-on.png" alt="image" style="zoom:67%;" />

## Acknowledgements

This project is built up on [SSAT](https://github.com/Snowfallingplum/SSAT), [Barbershop ](https://github.com/ZPdesu/Barbershop)and  [dressing-in-order](https://github.com/cuiaiyu/dressing-in-order). Please be aware of their licenses when using the code.

Thanks a lot for the great work to the pioneer researchers!