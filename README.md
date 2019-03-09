# DeepLabv3

In this repository we reproduce the DeepLabv3 paper which can be found here: [Rethinking Atrous Convolutions](https://arxiv.org/pdf/1706.05587.pdf)
The DeepLabv3 model expects the feature extracting architecture to be ResNet50 or ResNet101 so this repository will also contain the code of the ResNet50 and ResNet101 architecture.
We will also release colab notebook and pretrained models.

## How to use

0. This repository comes in with a handy notebook which you can use with Colab. <br/>
You can find a link to the notebook here: [
DeepLabv3](https://github.com/AvivSham/DeepLabv3.ipynb) <br/>
Open it in colab: [Open in Colab](https://colab.research.google.com/github/AvivSham/DeepLabv3/blob/master/DeepLabv3.ipynb)

---


0. Clone the repository and cd into it
```
git clone https://github.com/AvivSham/DeepLabv3.git
cd DeepLabv3/
```

1. Use this command to train the model
```
python3 init.py --mode train -iptr path/to/train/input/set/ -lptr /path/to/label/set/ --cuda False -nc <number_of_classes>
```

2. Use this command to test the model
```
python3 init.py --mode test -m /path/to/model.pth -i /path/to/image.png -nc <number_of_classes>
```

3. Use `--help` to get more commands
```
python3 init.py --help
```

---


0. If you want to download the cityscapes dataset
```
sh ./datasets/dload.sh cityscapes <your_username> <your_password>
```

1. If you want to download the PASCAL VOC 2012 datasets
```
sh ./datasets/dload.sh pascal
```

## Results

### Pascal VOC 2012

### CityScapes

## References
1. [Rethinking Atrous Convolutions](https://arxiv.org/pdf/1706.05587.pdf)
2. [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

## License

The code in this repository is free to use and to modify with proper linkage back to this repository.
