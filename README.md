# fakeglyph

![samples.gif](https://github.com/kumatheworld/fakeglyph/assets/14884102/5efcf9a5-4102-478c-84e1-8fc162bec663)

This repository lets you train generative models that try to produce letters.

## Setup

* Create a Python environment where `python>=3.10`.
* Clone the repository by `git clone https://github.com/kumatheworld/fakeglyph.git`.
* Install the required packages by `pip install -r requirements.txt`.
* Set the device. Change the `device` field of [configs/train.yaml](configs/train.yaml) to `cuda`, `cpu` or whatever device you want to use.
* Define the letter set and font file to create the dataset. Edit [configs/data/cjk.yaml](configs/data/cjk.yaml) and [fakeglyph/data/charset.py](fakeglyph/data/charset.py) accordingly. The default dataset is made of ~20K Chinese characters, but you can define your own letter set with any font.

## Train models

The first time you run the script, you'll need some time to generate the dataset, which will be cached under [datasets/](datasets/). You can easily change the model architecture by editting config files under [configs/model/](configs/model/).

### Train (β-)VAE

Run `python main.py train model=vae`. Run `python main.py train model=vae model.beta=1 model.reduction=batchmean` instead to train the vanilla VAE.

### Train GAN

Run `python main.py train model=gan`.
