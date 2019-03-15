#!/usr/bin/env python3
import torch
import json
import os
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
from pathlib import Path
from collections import OrderedDict
import operator
import importlib
import argparse
import sys
import time
import copy

"""
Purpose: Train a new network on a dataset and save the model as a checkpoint. Accuracy and loss on training, validation, and testing sets are printed out. 

Models available to use are 'alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169', 'densenet161', 'densenet201', 'inception_v3', 'squeezenet1_0', and 'squeezenet1_1'. Unlike the other models, squeezenet models contain a convolutional layer in its classifier. For this application, user-inputted hidden units will be ignored if squeezenet models are selected.   

Referred to https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html to create models based on architecture. 

To use: 
python train.py <data_directory DIR> --epochs <INT> --arch <STRING> --learning_rate <FLOAT> --hidden_units <LIST OF INT> --save_dir <DIR> --gpu

Example:
Train a network with gpu enabled on data in flowers directory with a densenset121 architecture, learning rate of 0.003, 20 epochs, and hidden layers with one layer containing 1024 nodes and another layer with 500 nodes. Save the checkpoint in a folder called 'checkpoints'. 

python train.py flowers --epochs 20 --arch densenet121 --learning_rate 0.0003 --hidden_units 1024 500 --save_dir checkpoints --gpu 
 
"""
__author__ = "Sara Jones"
__author_email__ = "jonessarae@gmail.com"
__doc__ = "Train a new network on a dataset and save the model as a checkpoint."
__date_modified__ = "3/15/19"

def make_classifier(arch, hidden_layers, model, model_type):
    """
    Creates the classifier and replaces it in the pretrained model. If the pretrained model does not have classifier, the fc (last) layer is replaced. Dropout rate is set at 0.2 for all models except squeezenet, which is 0.5 (default). For every hidden layer other than the last layer, ReLU activation and dropout are performed. The last layer undergoes log softmax activation. If the user did not provide hidden units, an automatic hidden unit is created. Hidden units are ignored for squeezenet models. 
    
    Arguments:
        arch - architecture for pretrained model; string
        hidden_layers - list of hidden units to be used for hidden layers; int
        model - pretrained model
        model_type - architecture type; string
        
    Returns:
        model - model with classifier
    """
    # Number of flower categories
    output_size = 102  
    
    # Determine input_size from model type
    if model_type == 'alexnet':
        input_size = model.classifier[1].in_features
    elif model_type == 'resnet':
        input_size = model.fc.in_features
    elif model_type == 'vgg':
        input_size = model.classifier[0].in_features
    elif model_type == 'densenet':
        input_size = model.classifier.in_features
    elif model_type == 'squeezenet':
        input_size = model.classifier[1].in_channels
    elif model_type == 'inception':
        input_size = model.fc.in_features
        # Need to adjust auxillary output
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, output_size)
   
    # If the user didn't give any input for --hidden_units, make default hidden layer
    if hidden_layers is None:
        hidden_layers = [int((input_size + output_size)/2)]
        
    # Make sure that user-inputted hidden units are above zero
    if all(i > 0 for i in hidden_layers) is False:
        print('Please ensure hidden units are above 0.')
        print('Exiting...')
        sys.exit(0)     
     
    # Dropout rate
    drop = 0.2
    # Length of hidden layers
    len_hl = len(hidden_layers)
    # Initialize OrderedDict
    network = OrderedDict()
    # First layer
    network['fc1'] = nn.Linear(input_size, hidden_layers[0])
    
    if len_hl > 1:
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    
         # Use for tracking layers
        x=1
        # Number of hidden layers
        y=len_hl

        # Iterate through only the nn.Linear objects, which starts at y, and add layers to OrderedDict
        for each in hidden_layers[len_hl:]:
            # Add ReLU activation
            network['relu'+str(x)] = nn.ReLU()
            # Add dropout 
            network['drop'+str(x)] = nn.Dropout(drop)
            # Add hidden layer 
            network['fc'+str(x+1)] = hidden_layers[y]
            x+=1
            y+=1
            if y==len(hidden_layers):
                network['relu'+str(x)] = nn.ReLU()
                network['drop'+str(x)] = nn.Dropout(drop)
    else:
        network['relu'] = nn.ReLU()
        network['drop'] = nn.Dropout(drop) 
        
    # Add hidden layer with number of categories to classify        
    network['fc'+str(len_hl+1)] = nn.Linear(hidden_layers[len_hl-1], output_size) 
    # Add log softmax activation
    network['output'] = nn.LogSoftmax(dim=1)
        
    # Create classifier
    classifier = nn.Sequential(network)
    
    if model_type in ['inception', 'resnet']:
        # Set classifier for model
        model.fc = classifier
    elif model_type == 'squeezenet':
        # reinitialize the Conv2d layer 
        model.classifier[1] = nn.Conv2d(input_size, output_size, kernel_size=(1,1), stride=(1,1))
        # In forward pass, there is a view function call which depends on the final output class size
        # https://discuss.pytorch.org/t/fine-tuning-squeezenet/3855/6
        model.num_classes = output_size
    else:
        model.classifier = classifier      

    print(model)
        
    return model    
    
def set_premodel(arch = 'densenet121'):
    """
    Load in pretrained model and freeze parameters. Default architecture is densenet121. 
    
    Argument:
        arch - architecture of pretrained model; string      
        
    Returns:
        model - pytorch model
    """
    # List of pretrained models 
    pre_models = ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19',
             'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
             'densenet121', 'densenet169', 'densenet161', 'densenet201', 'inception_v3', 'squeezenet1_0', 'squeezenet1_1']
    
    if arch in pre_models:
        # Create an instance of the pretrained model
        model = getattr(models, arch)(pretrained=True)
    else:
        print('\n{} not found. Please refer to the following list for available pretrained models.\n'.format(arch))
        print(pre_models)
        sys.exit(0)
    
    # Freeze parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad=False  
    
    return model

def train_model(model, epochs, device, criterion, dataloaders, optimizer, dataset_sizes, model_type):
    """
    Trains the model and prints the training loss and accuracy and validation loss and accuracy.
    The best model is used for validation of test set and saved as checkpoint.
    
    Arguments:
        model - pretrained model with classifier
        epochs - number of epochs; int        
        device - cuda or gpu
        criterion - negative log likelihood loss function 
        dataloaders - dataset; dict 
        optimizer - adam optimizer
        model_type - architecture type; string
        dataset_sizes = number of images in sets; dict
        
    Returns:
        model - trained
    """
    # Track time to train model
    start_time = time.time()

    # For storing the weights of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Loop through epochs
    for epoch in range(epochs):
    
        # Track epochs
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
    
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':      
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluation mode
                model.eval()
            
            running_loss = 0.0
            running_accuracy = 0
        
            # Loop through data
            for inputs, labels in dataloaders[phase]:           
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
            
                # Zero the parameter gradients
                optimizer.zero_grad()
                        
                # Perform forward pass and calculate loss (log probabilities) 
                # Set gradients only during training phase
                with torch.set_grad_enabled(phase == 'train'):
                    if model_type is 'inception' and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model.forward(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model.forward(inputs)
                        loss = criterion(outputs, labels)
                        
                    # Obtain predictions
                    _,preds = torch.max(outputs,1)
        
                    if phase == 'train':
                        # Backpropagate through model
                        loss.backward()
                        # Update classifier parameters
                        optimizer.step()
                
                # Stats per batch
                running_loss += loss.item() * inputs.size(0)
                running_accuracy += torch.sum(preds == labels.data)
         
            # Stats per epoch
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_accuracy = running_accuracy.double()/dataset_sizes[phase]        
      
            print('{} loss: {:.3f} accuracy: {:3f}'.format(phase,epoch_loss,epoch_accuracy))
        
            # deep copy the model
            if phase == 'valid' and epoch_accuracy > best_acc:
                best_acc = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())            
        
        print() #add newline

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

def test_model(dataloaders, model, device, criterion):
    """
    Performs validation on the test set and prints loss and accuracy. 
    
    Arguments:
        dataloaders - test set data; dataloader
        model - trained model
        device - cuda or gpu
        criterion - negative log likelihood loss function  
    """
    test_loss = 0
    accuracy = 0

    # Set model to evaluation mode
    model.eval()
    # Set it to device
    model.to(device)
    
    # Turn off gradients
    with torch.no_grad():
        # Loop through test set
        for inputs, labels in dataloaders['test']:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            # Perform forward pass and get log probabilities
            logps = model.forward(inputs)
            # Calculate loss
            loss = criterion(logps, labels)                    
            test_loss += loss.item()                    
            # Get probabilities
            probs = torch.exp(logps)
            # Determine top probabilities and predicted classes
            top_p, top_class = probs.topk(1, dim=1)
            # Check if predicted classes match the labels
            equals = top_class == labels.view(*top_class.shape)
            # Calculate accuracy
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    # Average loss and accuracy for test set, divided by number of batches
    avg_test_loss = test_loss/len(dataloaders['test'])
    avg_test_acc = accuracy/len(dataloaders['test'])
    
    print() # Add newline
    
    print(f"Test loss: {avg_test_loss:.3f}.. "
          f"Test accuracy: {avg_test_acc:.3f}") 

def save_checkpoint(save_dir, arch, epochs, model, optimizer, criterion, image_datasets, name, model_type):
    """
    Save the checkpoint.
    
    Arguments:
        save_dir - name of directory to save checkpoint; os.PathLike object or string
        arch - name of architecture used for model; string
        epochs - number of epochs to run; int
        model - trained model
        optimizer - adam optimizer
        criterion - negative log likelihood loss function 
        image_datasets - transformed images; dict
        name - name of checkpoint file; string
        model_type - architecture type; string
    """
    if model_type in ['inception','resnet']:
        classifier = model.fc
    else:
        classifier = model.classifier
        
    # Keep track of our mapping of flower class values to the flower indices
    model.class_to_idx = image_datasets['train'].class_to_idx

    # Create checkpoint
    checkpoint = {'arch': arch,
                 'epoch': epochs,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': criterion,
                 'class_to_idx': model.class_to_idx,
                 'classifier': classifier
                 }
    
    if save_dir is not None:
        try:
            # Create target Directory
            os.mkdir(save_dir)
            print("Directory " , save_dir ,  " created ") 
            path_to_checkpoint = save_dir + '/' + name    
            # Save checkpoint under user-inputted directory
            torch.save(checkpoint, path_to_checkpoint)
            print('Checkpoint saved at:')
            print(path_to_checkpoint)
        except FileExistsError:
            print("Directory " , save_dir ,  " already exists")
            path_to_checkpoint = save_dir + '/' + name    
            # Save checkpoint under user-inputted directory
            torch.save(checkpoint, path_to_checkpoint)
            print('Checkpoint saved at:')
            print(path_to_checkpoint)
    else:
        torch.save(checkpoint, name)    
        print('Checkpoint saved as {}'.format(name))
              
def load_images(data_directory, data_transforms):
    """
    Loads datasets with ImageFolder and performs transformations of the data.    
    
    Arguments:
        data_directory - directory containing data; os.PathLike object or string
        data_transforms - transforms for data set; dict   
        
    Returns:
        image_datasets - transformed images; dict
        dataset_sizes - number of images per set; dict
    """
    # Define directories
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Create dictionary linking directories
    dirs = {'train': train_dir, 
            'valid': valid_dir, 
            'test' : test_dir}    
    
    try:
        # Load the datasets with ImageFolder
        image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x]) for x in ['train', 'valid', 'test']}
        # Get the number of images for each dataset
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    except IOError as e:
        print(e)
        print('Make sure your data directory contains folders for train, valid, and test respectively.')
        sys.exit(0)
        
    return image_datasets, dataset_sizes

def define_transforms(arch):
    """
    Define transforms for the training, validation, and testing sets. Two sets of transforms based on argument arch.   
    
    Argument:
        arch - architecture; string
        
    Returns:
        data_transforms - transforms for data; dict
    """

    # inception_v3 requires 299 x 299
    if arch == 'inception_v3':
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
               transforms.Resize(350),
               transforms.CenterCrop(299),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(350),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
        }
    else:
        # All other pretrained models require 224 x 224
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
               transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
               transforms.Normalize([0.485, 0.456, 0.406], 
                                    [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
        }
    
    return data_transforms
    
def define_dataloaders(image_datasets):
    """
    Define dataloaders. Batches data and shuffles the data.
    
    Argument:
        image_datasets - datasets of transformed images; dict
        
    Returns:
        dataloaders; dict
    """
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}
    return dataloaders

def main(args):
    
    # Create dictionary of main types of models to versions
    pretrained_model = {'alexnet': ['alexnet'],
                        'vgg': ['vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19', 'vgg19_bn'],
                        'resnet':['resnet18','resnet34','resnet50','resnet101','resnet152'],
                        'inception':['inception_v3'],
                        'squeezenet':['squeezenet1_0', 'squeezenet1_1'],
                        'densenet': ['densenet121', 'densenet169', 'densenet161','densenet201']
                       }
    
    # Get key from pretrained_model
    model_type = [k for k,v in pretrained_model.items() if args.arch in v][0]
    
    # Define the loss function
    if model_type == 'squeezenet':
        # Notify user that hidden units will be ignored for squeezenet models
        print('\nAny user-inputted hidden units will be ignored for squeezenet models.\n')           
        # Define the loss function for squeezenet
        # This criterion combines log_softmax (not in classifier) and nll_loss in a single function
        criterion = nn.CrossEntropyLoss()
    else:
        # Define the loss function for all other models
        criterion = nn.NLLLoss()
        
    # Define transforms
    transformations = define_transforms(args.arch)
    
    # Load images and transforms them, also get number of images per set
    images, sizes = load_images(args.data_directory, transformations)
    
    # Define dataloaders
    datasets = define_dataloaders(images)
    
    # Load in pretrained model and freeze parameters of pretrained model
    pmodel = set_premodel(arch = args.arch)   
    
    # Create classifier and add it to model
    model = make_classifier(args.arch, args.hidden_units, pmodel, model_type)
    
    # Define the optimizer and train the classifier parameters only
    # Apply absolute power on learning_rate to avoid negative numbers
    if model_type in ['inception', 'resnet']:
        optimizer = optim.Adam(model.fc.parameters(), lr=abs(args.learning_rate))
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=abs(args.learning_rate))
        
    # Check if user used --gpu option
    if args.gpu is True:
        # Check if cuda is available
        if torch.cuda.is_available():
            # Set device to cuda
            device = torch.device('cuda')           
        else:
            print('Use cpu or ensure gpu is available for use.')
            print('Exiting...')
            sys.exit(0)
    else:
        # Set device to cpu
        device = torch.device('cpu')
        
    # Move model to whichever device is available
    model.to(device)   
    
    # Train model
    tmodel = train_model(model, args.epochs, device, criterion, datasets, optimizer, sizes, model_type)
    
    # Perform validation on test set
    test_model(datasets, tmodel, device, criterion)  
    
    # Name checkpoint file, includes epoch, learning rate, and architecture as part of name
    name = 'checkpoint_{}_{}_{}.pth'.format(args.epochs,args.learning_rate,args.arch) 
    
    # Save checkpoint
    save_checkpoint(args.save_dir, args.arch, args.epochs, tmodel, optimizer, criterion, images, name, model_type)   
    
if __name__ == "__main__":
    
    # Create arguments
    p = argparse.ArgumentParser(description=__doc__, prog='train.py', usage='%(prog)s <data_directory DIR> [options]', add_help=True)
    p.add_argument('data_directory', help='path to directory containing data')
    p.add_argument('--epochs', default=10, help='number of epochs for training model, default is 10', type=int)
    p.add_argument('--arch', default='densenet121', help='architecture of pretrained model, default is densenet121', type=str.lower) 
    p.add_argument('--learning_rate', default=0.01, help='learning rate, default is 0.01', type=float)
    p.add_argument('--hidden_units', nargs='+', help='hidden units', type=int) 
    p.add_argument('--save_dir', help='directory for saving checkpoints')
    p.add_argument('--gpu', action='store_true', default=False, help='enable gpu')  

    # Check number of arguments entered by user, must have image and checkpoint files
    if len(sys.argv) < 2:
        print('Please enter data directory.\n')
        p.print_help()        
        sys.exit(0)
    else:
        # Set arguments
        args = p.parse_args()
        print(args)
        main(args)