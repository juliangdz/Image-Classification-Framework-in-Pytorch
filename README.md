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

The extensive Wandb report can be found [here](https://api.wandb.ai/links/juliangeralddcruz/uuvl61p8).