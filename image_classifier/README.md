
## Project: Image Classifier

### Goal 

The goal of this project is to train an image classifier on the flower dataset from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html to recognize different species of flowers (102 categories).
In this project, the first part is developing code for an image classifier built with PyTorch, then converting it into a command line application in the second part.


### Software

This project uses Python 3.6.3 and Pytorch, a Python-based deep learning research platform. 

All libraries and versions can be found in image_classifier_report.html. 

### Files

cat_to_name.json - maps the integer encoded categories to the actual names of the flowers
predict.py - application to predict class of the flower in the image from checkpoint file
train.py -  application to train a new network on dataset and save checkpoint
checkpoint5_densenet121.pth - example of checkpoint file
image_classifier_project.ipynb - jupyter notebook for part 1
image_classifier_report.html - html version of jupyter notebook

Not included is the actual dataset divided into train, test, and valid folders. 

### Code

Code for Part 1 of this project is provided in image_classifier_project.ipynb and image_classifier_report.html.
Code for Part 2 of this project is provided in predict.py and train.py.     

### Run

The application for part 2 includes two files, predict.py and train.py. 
Make sure you have the data available on your computer.

To train a new network with train.py, run the following in the terminal:
```
python train.py <data_directory DIR> --epochs <INT> --arch <STRING> --learning_rate <FLOAT> --hidden_units <LIST OF INT> --save_dir <DIR> --gpu
```
Example:
Train a network with gpu enabled on data in flowers directory with a vgg19 architecture, learning rate of 0.003, 10 epochs, and hidden layers with one layer containing 1000 nodes and another layer with 500 nodes. Save the checkpoint in a folder called 'checkpoints'. 
```
python train.py flowers --arch vgg19 --epochs 10 --hidden_units 1000 500 --learning_rate 0.003 --save_dir checkpoints --gpu
```
It is highly recommended that you enable gpu for training the model. 

To predict the class of an image with a trained network with predict.py, run the following in the terminal:
```
python predict.py <path_to_image> <path_to_checkpoint> --category_names <path_to_category_names> --top_k <int> --gpu
```
Example (using cpu):
```
python predict.py flowers/test/10/image_07090.jpg checkpoint5_densenet121.pth --category_names cat_to_name.json --top_k 7
```
### References

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad

https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
