# CIFAR-10 CNN Architectures: Baseline vs AlexNet vs VGG vs ResNet

A PyTorch notebook that implements four convolutional neural networks from scratch, trains them under identical settings on CIFAR-10, and compares how architectural ideas (depth, dropout, batch norm, skip connections) affect accuracy and overfitting.

## Results

All models: 10 epochs, Adam (lr = 0.001), batch size 100, no data augmentation.

| Model | Parameters | Train Acc (%) | Test Acc (%) |
|---|---|---|---|
| Baseline CNN | 1.97M | 97.32 | 71.76 |
| AlexNet (CIFAR version) | 35.86M | 85.87 | 76.14 |
| VGG-style (+ BatchNorm) | 9.23M | 96.79 | 81.80 |
| ResNet-18 | 11.17M | 96.96 | **82.95** |

**Takeaways**

- **Baseline CNN** has the largest train/test gap (~25.5 points). Two conv layers plus a big FC head memorize the training set.
- **AlexNet** with Dropout closes the gap the most (~9.7 points), but it is the largest model by far (35.9M params, mostly in the FC layers) and trains slowest to fit.
- **VGG-style** gets a big jump from stacked 3×3 convs and BatchNorm, with a single-Linear head keeping parameters low.
- **ResNet-18** performs best. Skip connections plus global average pooling give the highest test accuracy with a tiny FC head.

Results will vary slightly between runs (random init and shuffling).

## Models

| Model | Key ideas | Adaptation for 32×32 images |
|---|---|---|
| **CNN** | 2 conv layers + 3-layer FC head | Baseline from a previous lab |
| **AlexNet** | 5 conv layers, wider channels (up to 384), ReLU, Dropout(0.5) in FC head | 3×3 filters instead of 11×11 stride-4 |
| **VGG** | Uniform stacked 3×3 convs, channels 64→128→256→512, BatchNorm | Single Linear head instead of 3 huge FC layers |
| **ResNet-18** | `BasicBlock` with skip connections, BatchNorm, global average pooling | 3×3 stem with stride 1 (no 7×7 conv or initial max-pool) |

## Getting started

### Requirements

- Python 3.9+
- PyTorch and torchvision
- numpy, matplotlib, pandas
- torchinfo (installed in the notebook via `pip`)

```bash
pip install torch torchvision numpy matplotlib pandas torchinfo jupyter
```

### Run

```bash
git clone <your-repo-url>
cd <your-repo-name>
jupyter notebook cnn_architectures_alexnet_vgg_resnet.ipynb
```

Run all cells in order. The notebook:

1. Downloads CIFAR-10 (from a Zenodo mirror, MD5-verified) and builds data loaders
2. Visualizes sample images
3. Defines the four models
4. Calls one `train_and_test()` function per model, which trains, evaluates on train and test sets, and logs results
5. Prints a final comparison table

A GPU is recommended. The notebook automatically uses CUDA if available, and also runs on Google Colab.

## Project structure

```
.
├── cnn_architectures_alexnet_vgg_resnet.ipynb   # Full code, training, and comparison
└── README.md
```

## Possible extensions

- Add data augmentation (random crop, horizontal flip) to reduce overfitting
- Use a learning-rate scheduler and train for more epochs
- Add deeper ResNets (34/50) or other architectures (DenseNet, MobileNet)
- Plot training and validation curves per epoch

## Acknowledgements

- Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (Krizhevsky, 2009)
- Architectures based on AlexNet (Krizhevsky et al., 2012), VGG (Simonyan & Zisserman, 2014), and ResNet (He et al., 2015)

## License

MIT (add a `LICENSE` file if you'd like to use this)
