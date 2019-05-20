from collections import OrderedDict

import torch
from torch import nn
from torchvision import models, transforms, datasets

model_files = {
  "supervised":"imagenet_pretrained_supervised2_alexnet_71_checkpoint.pth",
  "rotation":"imagenet_pretrained_rotnet2_alexnet_44_checkpoint.pth"
}


def create_model(arch,path):
    model = None
    out_categories = 46 #hardcoded at the moment

    model_path = path/"pretrained_models"/model_files[arch]
    checkpoint = torch.load(model_path, map_location="cpu")

    new_checkpoint = {}
    for key,value in checkpoint.items():
        new_checkpoint[key.replace('module.','')] = value

    if arch == "supervised":
        model = models.alexnet(pretrained=False)

        model.classifier = nn.Sequential(OrderedDict([
                            ('fc1',nn.Linear(9216, 512)),
                            ('relu1',nn.ReLU()),
                            ('dropout1',nn.Dropout(0.2)),
                            ('fc2',nn.Linear(512,128)),
                            ('relu2',nn.ReLU()),
                            ('dropout2',nn.Dropout(0.1)),
                            ('fc3',nn.Linear(128,out_categories))
                        ]))
    elif arch == "rotation":
        from rotnet_alexnet import create_model,Flatten  

        model = create_model({'num_classes':4})

        model._feature_blocks[8] = nn.Sequential(OrderedDict([
                          ('0',Flatten()),
                          ('1',nn.Linear(9216, 512)),
                          ('2',nn.ReLU()),
                          ('3',nn.Dropout(0.2)),
                          ('4',nn.Linear(512, 128)),
                          ('5',nn.ReLU()),
                          ('6',nn.Dropout(0.1))
                      ]))

        model._feature_blocks[9] = nn.Sequential(OrderedDict([
                             ('0',nn.Linear(128,out_categories))
                            ]))

    model.load_state_dict(new_checkpoint)

    return model
