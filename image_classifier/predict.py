#!/usr/bin/env python3
import torch
import json
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

"""
Purpose: Use a trained network for inference by passing a single image into the network and predicting the class of the flower in the image. 

Models available to use are 'alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169', 'densenet161', 'densenet201', 'inception_v3', 'squeezenet1_0', and 'squeezenet1_1'. 

To use: 
python predict.py <path_to_image> <path_to_checkpoint> --category_names <path_to_category_names> --top_k <int> --gpu

Example (using cpu):
python predict.py flowers/test/10/image_07090.jpg checkpoint5_densenet121.pth --category_names cat_to_name.json --top_k 7
"""
__author__ = "Sara Jones"
__author_email__ = "jonessarae@gmail.com"
__doc__ = "Rebuilds a trained network from a checkpoint file, passes an image into the network, and predicts the class of the flower in the image."
__date_modified__ = "3/15/19"

def load_checkpoint(path_to_checkpoint, enable_gpu, pretrained_models):
    """
    Loads a checkpoint and rebuilds the model. Also returns whether model uses inception_v3 architecture and architecture type (model_type)
    
    Arguments:
        path_to_checkpoint - path to checkpoint file; os.PathLike object or string
        enable_gpu - if true, use cuda, else use cpu; boolean
        pretrained_models - architecture type; dict
        
    Returns:
        model - pytorch model
        inception - boolean
        model_type - architecture type; string
    """
    # Set device to cpu or cuda and load checkpoint
    if enable_gpu is True:
        # Check if cuda is available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            try:
                checkpoint = torch.load(path_to_checkpoint)
            except pickle.UnpicklingError as e:
                print(e)
                print('Error loading checkpoint. Check that you are providing a checkpoint file.')
                sys.exit(0)
        else:
            print('Use cpu or ensure gpu is available for use.')
            print('Exiting...')
            sys.exit(0)
    else:
        device = torch.device('cpu')
        try:
            checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
        except pickle.UnpicklingError as e:
            print(e)
            print('Error loading checkpoint. Check that you are providing a checkpoint file.')
            sys.exit(0)
    
    # Check if pretrained model is inception_v3
    if checkpoint['arch'] == 'inception_v3':
        inception = True
    else:
        inception = False

    # Create an instance of the pretrained model
    model = getattr(models, checkpoint['arch'])(pretrained=True)

    # Get architecture type
    model_type = [k for k,v in pretrained_models.items() if checkpoint['arch'] in v][0]
    
    # Create attributes of model from checkpoint
    if model_type in ['inception', 'resnet']:
        model.fc = checkpoint['classifier']
        optimizer = optim.Adam(model.fc.parameters())
        if model_type == 'inception':
            model.AuxLogits.fc = nn.Linear(768, 102)
    else:
        model.classifier = checkpoint['classifier']
        optimizer = optim.Adam(model.classifier.parameters()) 
        if model_type == 'squeezenet':
            model.num_classes = 102
            
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    criterion = checkpoint['loss']
    model.class_to_idx = checkpoint['class_to_idx']

    # Put model on device
    model.to(device)
    
    return model, inception, model_type

def process_image(path_to_image, is_inception):
    """ 
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    
    Arguments:
        path_to_image - path to image file; os.PathLike object or string
        is_inception - checks if inception_v3 was used for pretrained model; boolean
    
    Returns:
        np_final_image - numpy array
    """  
    try:
        # Open image as PIL image
        im = Image.open(path_to_image)
    except IOError as e:
        print(e)
        print('Error loading image file. Check that you are providing an image file.')
        sys.exit(0)
       
    # Get width and height of PIL image
    width, height = im.size
    # Calculate aspect ratio
    aspect_ratio = width/height
    
    if is_inception is True:
        # Make shortest side 350 pixels, keeping aspect ratio
        if width < height:
            im.resize((350,int(350*aspect_ratio**-1)))
        else:
            im.resize((int(350*aspect_ratio),350))
        
        # Center crop to 299 x 299 
        left = int((width - 299)/2)
        top = int((height - 299)/2)
        right = int((width + 299)/2)
        bottom = int((height + 299)/2)
        im = im.crop((left, top, right, bottom))
        
    else:
        # Make shortest side 256 pixels, keeping aspect ratio
        if width < height:
            im.resize((256,int(256*aspect_ratio**-1)))
        else:
            im.resize((int(256*aspect_ratio),256))
        
        # Center crop to 224 x 224 
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2
        im = im.crop((left, top, right, bottom))
    
    # Convert PIL image to numpy array
    np_image = np.array(im)
    
    # Scale color channels to floats 0–1
    np_image_scaled = np.array(np_image)/255
    
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_norm_image = (np_image_scaled - mean)/std
    
    # Reorder dimensions so that color channel is first, retain order of other two
    np_final_image = np.transpose(np_norm_image, [2,0,1])

    return np_final_image 

def predict(path_to_image, cat_to_name, model, inception, topk, enable_gpu, model_type):
    """ 
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Arguments:
        path_to_image - path to image file; os.PathLike object or string
        cat_to_name - dictionary of encoded categories (integer) to flower names (string)
        model - PyTorch model
        inception - checks if pretrained model is inception; boolean
        topk - returns the top k classes; integer
        enable_gpu - if true, use cuda, else use cpu; boolean
        model_type - architecture type; string
    Returns:
        top_probs - list of top probabilities; float
        top_classes - list of top classes; string
    """ 
    # Set device to cpu or cuda and convert model to cuda or cpu
    if enable_gpu is True:
        # Check if cuda is available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model.cuda()
        else:
            print('Use cpu or ensure gpu is available for use.')
            print('Exiting...')
            sys.exit(0)
    else:
        device = torch.device('cpu')
        model.cpu()
   
    # Set model to evaluation mode
    model.eval()
    
    # Preprocess image
    img = process_image(path_to_image, is_inception=inception)
    
    # Convert numpy array image to tensor
    timg = torch.from_numpy(img).type(torch.FloatTensor)

    # Add batch of size 1 to image
    timg.unsqueeze_(0)

    # Calculate the class log probabilities for img
    with torch.no_grad():
        timg = timg.to(device)
        output = model.forward(timg)

    # Calculate class probabilities for img
    if model_type == 'squeezenet':
        ps = nn.functional.softmax(output)
    else:
        ps = torch.exp(output)
    
    # Determine top probabilities and predicted classes 
    top_probs, top_indices = ps.topk(topk, dim=1)
    top_probs = top_probs.cpu().detach().numpy().tolist()[0] 
    top_indices = top_indices.cpu().detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_indices]    
    
    return top_probs, top_classes

def load_json_labels(path_to_json):
    """
    Loads a json file and maps the integer encoded categories to the actual names of the flowers.
    
    Argument:
        path_to_json - path to json labels file; file object
           
    Returns:
        cat_to_name - dictionary of integer encoded categories to flower names or None.    
    """
    if path_to_json is not None:        
        try:
            with open(path_to_json, 'r') as f:
                try:
                    cat_to_name = json.load(f)
                except (UnicodeDecodeError, json.decoder.JSONDecodeError) as e:
                    print(e)
                    print('The category_names file is not in json format. Proceeding without the file.')
                    cat_to_name = None
        except IOError as e:
            print(e)
            print('Check that you provided a category_names file. Proceeding without the file.')
            cat_to_name = None
    else:
        cat_to_name = None
     
    return cat_to_name

def main(args):
   
    # Dictionary of pretrained models to main architecture type
    pretrained_models = {'alexnet': ['alexnet'],
                        'vgg': ['vgg11','vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19', 
                        'vgg19_bn'],
                         'squeezenet' : ['squeezenet1_0', 'squeezenet1_1'],
                        'resnet':['resnet18','resnet34','resnet50','resnet101','resnet152'],
                        'inception':['inception_v3'],
                        'densenet': ['densenet121', 'densenet169', 'densenet161','densenet201']
                       }  
  
    # Load model 
    model, inception, model_type = load_checkpoint(args.checkpoint, args.gpu, pretrained_models)
   
    # Load json file to get flower names
    categories = load_json_labels(args.category_names)
        
    # Obtain top k classes and their probabilities
    probs, classes = predict(args.image, categories, model, inception, args.top_k, args.gpu, model_type)
    
    # Get name of image 
    img_path = Path(args.image)
    if categories is not None:
        name = categories[img_path.parts[-2]]
        flowers = [categories[cat] for cat in classes] 
    else:
        name = img_path.parts[-2]
        flowers = classes
        
    # Print stats
    print('\nActual flower type for image is {}.\n'.format(name))
    print('Top predicted flower type is {} and its probability is {:.3f}.\n'.format(flowers[0], probs[0]))
    print('Top {} classes and their probabilities:'.format(args.top_k))
    for prob, flower in zip(probs, flowers):
        print('{}: {:.3f}'.format(flower, prob))
    
if __name__ == "__main__":

    # Create arguments
    p = argparse.ArgumentParser(description=__doc__, prog='predict.py', usage='%(prog)s <image FILE> <checkpoint FILE> [options]', add_help=True)
    p.add_argument('image', help='path to image file')
    p.add_argument('checkpoint', help='path to checkpoint file')
    p.add_argument('--category_names', help='path to json file of categories')
    p.add_argument('--top_k', default=5, help='number of the top classes to print out, default is 5', type=int)
    p.add_argument('--gpu', action='store_true', default=False, help='enable gpu')  

    # Check number of arguments entered by user, must have image and checkpoint files
    if len(sys.argv) < 3:
        print('Please enter required image and checkpoint files.\n')
        p.print_help()        
        sys.exit(0)
    else:
        # Set arguments
        args = p.parse_args()
        print(args)
        main(args)
 
