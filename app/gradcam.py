############### FOR SUPERVISED ###################################

import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse


pytorch_labels = {'1': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5, '15': 6, '16': 7, '17': 8, '18': 9, '19': 10, '2': 11, '20': 12, '21': 13, '22': 14, '23': 15, '24': 16, '25': 17, '26': 18, '27': 19, '28': 20, '29': 21, '3': 22, '30': 23, '31': 24, '32': 25, '33': 26, '34': 27, '35': 28, '36': 29, '37': 30, '39': 31, '4': 32, '40': 33, '41': 34, '42': 35, '43': 36, '44': 37, '46': 38, '47': 39, '48': 40, '5': 41, '6': 42, '7': 43, '8': 44, '9': 45}


reverse_map = {}
for k,v in pytorch_labels.items():
    reverse_map[v] = int(k) 
actual_names = ['None',
 'Anorak',
 'Blazer',
 'Blouse',
 'Bomber',
 'Button-Down',
 'Cardigan',
 'Flannel',
 'Halter',
 'Henley',
 'Hoodie',
 'Jacket',
 'Jersey',
 'Parka',
 'Peacoat',
 'Poncho',
 'Sweater',
 'Tank',
 'Tee',
 'Top',
 'Turtleneck',
 'Capris',
 'Chinos',
 'Culottes',
 'Cutoffs',
 'Gauchos',
 'Jeans',
 'Jeggings',
 'Jodhpurs',
 'Joggers',
 'Leggings',
 'Sarong',
 'Shorts',
 'Skirt',
 'Sweatpants',
 'Sweatshorts',
 'Trunks',
 'Caftan',
 'Cape',
 'Coat',
 'Coverup',
 'Dress',
 'Jumpsuit',
 'Kaftan',
 'Kimono',
 'Nightdress',
 'Onesie',
 'Robe',
 'Romper',
 'Shirtdress',
 'Sundress']

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers, arch):
        self.model = model
        self.arch = arch
        if self.arch == "supervised":
            self.feature_extractor = FeatureExtractor(self.model.features, target_layers)
        elif self.arch == "rotation":
            self.feature_extractor = FeatureExtractor(self.model._feature_blocks, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        if self.arch == "supervised":
            output = output.view(output.size(0), -1)	
            output = self.model.classifier(output)
        return target_activations, output

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

def show_cam_on_image(img, mask, path, arch):
    print("Call came to show_cam")
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    image_path = "./app/static/images/cam_{}.jpg".format(arch)
    print(image_path)
    cv2.imwrite(image_path, np.uint8(255 * cam))

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda, arch): 
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.arch = arch
	#if self.cuda:
	#   self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, self.arch)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
                
        print(reverse_map[index])
        predicted_category = actual_names[reverse_map[index]].strip().lower()      
    
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        if self.arch == "supervised":
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        elif self.arch == "rotation":
            self.model._feature_blocks.zero_grad()

        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam,predicted_category

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda, arch):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.arch = arch
	#if self.cuda:
	#   self.model = model.cuda()

	# replace ReLU with GuidedBackpropReLU
        if self.arch == "supervised":
            for idx, module in self.model.features._modules.items():
                if module.__class__.__name__ == 'ReLU':
                    self.model.features._modules[idx] = GuidedBackpropReLU()
        elif self.arch == "rotation":
            for idx, module in self.model._feature_blocks._modules.items():
                if module.__class__.__name__ == 'ReLU':
                    self.model._feature_blocks._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        print("INDEX {}".format(index))
        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

	# self.model.features.zero_grad()
	# self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=True,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default=DATA_DIR+'/test/1/Hooded_Cotton_Canvas_Anorak_img_00000005.jpg',
	                    help='Input image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for acceleration")
	else:
	    print("Using CPU for computation")

	return args
