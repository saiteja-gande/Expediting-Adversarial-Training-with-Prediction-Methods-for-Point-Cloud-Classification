# StudienArbeit: Expediting Adversarial Training with Prediction Methods for PointNet and DGCNN

This project aims to expedite adversarial training with prediction methods to enhance the resilience of PointNet and DGCNN architectures against adversarial attacks on point cloud data.

## Requirements

To run this project, you will need the following dependencies:

*   python==3.9.15
*   numpy>=1.23.0
*   Pillow>=9.3.0
*   torch==1.13.1
*   torchvision==0.14.1
*   pytorch-lightning==1.8.6
*   tensorboard==2.11.2

## Usage

To use the code with the default settings, simply run the following command:
```
python main.py
```

You can also customize the settings by using the following command line arguments:

*   `--model`: Choose between PointNet and DGCNN models (default: PointNet).
*   `--epochs`: Set the number of epochs to train (default: 50).
*   `--Tnet`: Choose whether to use Tnet or not (default: True).
*   `--training`: Choose between 'fgsm_attack', 'pgd_linf', or 'mixed' types of training (default: None).
*   `--attack`: Choose between 'fgsm' or 'pgd' adversarial attacks (default: None).
*   `--prediction`: Choose whether to use prediction or not (default: False).
*   `--learning_rate`: Set the learning rate (default: 0.01).
*   `--step_size`: Set the step size for adversarial attacks (default: 0.001).
*   `--steps`: Set the number of steps for a PGD attack (default: 5).
*   `--epsilon`: Set the epsilon value for adversarial attacks (default: 0.1).
*   `--num-workers`: Set the number of workers (default: 0).
*   `--batch-size`: Set the batch size (default: 32).
*   `--delta`: Set the value of the multiplication factor (Gamma value in the report) of the prediction method (default:1).

## Contents

This repository contains the following files:

- `attacks.py`: Implementation of adversarial attacks used in this project.
- `dataloader.py`: Data loader class with necessary augmentations.
- `models.py`: Implementation of PointNet and DGCNN architectures.

The dataset used in this project is [ModelNet10](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip).

## Report

The detailed information and analysis of this project can be found at - [Report](https://github.com/saiteja1012/Expediting-Adversarial-Training-with-Prediction-Methods-for-Point-Cloud-Classification/files/12395324/Studienarbeit_Saiteja_Gande.pdf)
