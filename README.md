Here's how you could structure the markdown for your project description to make it more professional and visually appealing:

# Image-Classification-Framework-in-Pytorch

- **Student ID**: 437451
- **Student Name**: Julian Gerald Dcruz
- **Course Code**: N-AAI-DAV-23-A23
- **Deep Learning Assignment**

## Installation

To set up the necessary environment, run the following command:

```
conda create -n pytorch_env -f environment.yml
```

## Usage

To run the project, use:

```
python main.py config.json
```

## Configuration Options

### Using FCN

To use the Fully Convolutional Network (FCN), modify `config.json` as follows:

```json
{
    "hyperparameters": {
        "network": "fcn"
        // Everything else remains the same
    }
}
```

### Using Custom CNN

For a custom CNN, edit `config.json` like this:

```json
{
    "hyperparameters": {
        "network": "deep_cnn"
        // Everything else remains the same
    }
}
```

### Using Pretrained ResNet18

To use a pretrained ResNet18 model, change `config.json` to:

```json
{
    "hyperparameters": {
        "network": "pretrained"
        // Everything else remains the same
    },
    "networks": {
        "pretrained": {
            "name": "resnet18"
            // Everything else remains the same
        }
    }
}
```

### Using Custom LeNet-5

For LeNet-5, update `config.json` accordingly:

```json
{
    "hyperparameters": {
        "network": "lenet"
        // Everything else remains the same
    }
}
```

## WANDB Report

The extensive Wandb report can be found [here](https://wandb.ai/juliangeralddcruz/MNIST_Classification/reports/MNIST-Classification---Vmlldzo2Nzc2MDM0?accessToken=gwapdhbigkw7e6h35v0ij2xsptv2vcn8jri5fnl82jv6yekpeer2wi1ly5wiw6jp).