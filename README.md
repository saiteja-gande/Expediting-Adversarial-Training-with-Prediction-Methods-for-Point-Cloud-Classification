# StudienArbeit on Expediting-Adversarial-Training-with-Prediction-Methods-for-PointNet-and-DGCNN
Expediting Adversarial Training with Prediction Methods to Enhance PointNet and DGCNN Architectures’ Resilience against Adversarial Attacks on Point Cloud Data

## Requirements

*   python==3.9.15
*   numpy>=1.23.0
*   Pillow>=9.3.0
*   torch==1.13.1
*   torchvision==0.14.1
*   pytorch-lightning==1.8.6
*   tensorboard==2.11.2

## Usage
To use the code with the default setting, simply run the command 
'python main.py'
To use different settings following command line arguments are available to use and can be used accordingly:

*   --model: To change the Model between PointNet and DGCNN (default: PointNet).
*   --epochs: To change the number of epochs to train (default: 50).
*   --Tnet : To change between with or without Tnet (default: True)
*   --training : To change between 'fgsm_attack', 'pgd_linf','mixed' types of training (default: None)
*   --attack : To change between 'fgsm', 'pgd' adversarial attacks
*   --prediction : Use if we want prediction or not (default: False)
*   --learning_rate : To change the learning rate (default: 0.01)
*   --step_size : To change the step size for adversarial attacks (default: 0.001)
*   --steps : To change the number of steps for a PGD attack (default: 5)
*   --epsilon : To change the epsilon value for adversarial attacks (default: 0.1)
*   --num-workers : To change the number of workers (default: 0)
*   --batch-size : To change the batch size (default: 32)
*   --delta : To change the value of the multiplication factor of the prediction Method (default: 1)

## contents

* This directory includes the implementation of Adversarial attacks used for the project in `attacks.py`.
* The `dataloader.py` contains the data loader class along with necessary augmentations.
* The `models.py` consists of implemented PointNet and DGCNN architectures.
* The dataset used for this project is [ModelNet10](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip) 
