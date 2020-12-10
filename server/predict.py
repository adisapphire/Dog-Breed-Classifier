import argparse
import torch
import os
import io
import PIL.Image as Image
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision.models as models
import numpy as np
from model import BreedClassifier
import json



from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
VGG16 = models.vgg16(pretrained=True)


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")
    
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)
    
    print("model_info: {}".format(model_info))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    map_location=lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    model_path = os.path.join(model_dir, 'model.pth')
    
    print("cuda info: {}".format(map_location))
    
    
    model = BreedClassifier()    
        
    if model_info['change_model']==7:
        
        model = model.TransferLearning()
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f,map_location=map_location))
            
        model.to(device).eval()
        
    else:

        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f,map_location=map_location))
        
        model.to(device).eval()
        
    print("Done loading model.")    
    return model


def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'application/x-image':
        data = Image.open(io.BytesIO(serialized_input_data)).convert('RGB')
        print(data)
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)


def dog_detector(image):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        image = image.cuda()
        VGG16.cuda()
    # predict
    ret = VGG16(image)
    idx =  torch.max(ret,1)[1].item()
    
    return idx >= 151 and idx <= 268

def predict_fn(input_data, model):
    print('Inferring breed of input data.')
    class_names = ['Affenpinscher',
                     'Afghan hound',
                     'Airedale terrier',
                     'Akita',
                     'Alaskan malamute',
                     'American eskimo dog',
                     'American foxhound',
                     'American staffordshire terrier',
                     'American water spaniel',
                     'Anatolian shepherd dog',
                     'Australian cattle dog',
                     'Australian shepherd',
                     'Australian terrier',
                     'Basenji',
                     'Basset hound',
                     'Beagle',
                     'Bearded collie',
                     'Beauceron',
                     'Bedlington terrier',
                     'Belgian malinois',
                     'Belgian sheepdog',
                     'Belgian tervuren',
                     'Bernese mountain dog',
                     'Bichon frise',
                     'Black and tan coonhound',
                     'Black russian terrier',
                     'Bloodhound',
                     'Bluetick coonhound',
                     'Border collie',
                     'Border terrier',
                     'Borzoi',
                     'Boston terrier',
                     'Bouvier des flandres',
                     'Boxer',
                     'Boykin spaniel',
                     'Briard',
                     'Brittany',
                     'Brussels griffon',
                     'Bull terrier',
                     'Bulldog',
                     'Bullmastiff',
                     'Cairn terrier',
                     'Canaan dog',
                     'Cane corso',
                     'Cardigan welsh corgi',
                     'Cavalier king charles spaniel',
                     'Chesapeake bay retriever',
                     'Chihuahua',
                     'Chinese crested',
                     'Chinese shar-pei',
                     'Chow chow',
                     'Clumber spaniel',
                     'Cocker spaniel',
                     'Collie',
                     'Curly-coated retriever',
                     'Dachshund',
                     'Dalmatian',
                     'Dandie dinmont terrier',
                     'Doberman pinscher',
                     'Dogue de bordeaux',
                     'English cocker spaniel',
                     'English setter',
                     'English springer spaniel',
                     'English toy spaniel',
                     'Entlebucher mountain dog',
                     'Field spaniel',
                     'Finnish spitz',
                     'Flat-coated retriever',
                     'French bulldog',
                     'German pinscher',
                     'German shepherd dog',
                     'German shorthaired pointer',
                     'German wirehaired pointer',
                     'Giant schnauzer',
                     'Glen of imaal terrier',
                     'Golden retriever',
                     'Gordon setter',
                     'Great dane',
                     'Great pyrenees',
                     'Greater swiss mountain dog',
                     'Greyhound',
                     'Havanese',
                     'Ibizan hound',
                     'Icelandic sheepdog',
                     'Irish red and white setter',
                     'Irish setter',
                     'Irish terrier',
                     'Irish water spaniel',
                     'Irish wolfhound',
                     'Italian greyhound',
                     'Japanese chin',
                     'Keeshond',
                     'Kerry blue terrier',
                     'Komondor',
                     'Kuvasz',
                     'Labrador retriever',
                     'Lakeland terrier',
                     'Leonberger',
                     'Lhasa apso',
                     'Lowchen',
                     'Maltese',
                     'Manchester terrier',
                     'Mastiff',
                     'Miniature schnauzer',
                     'Neapolitan mastiff',
                     'Newfoundland',
                     'Norfolk terrier',
                     'Norwegian buhund',
                     'Norwegian elkhound',
                     'Norwegian lundehund',
                     'Norwich terrier',
                     'Nova scotia duck tolling retriever',
                     'Old english sheepdog',
                     'Otterhound',
                     'Papillon',
                     'Parson russell terrier',
                     'Pekingese',
                     'Pembroke welsh corgi',
                     'Petit basset griffon vendeen',
                     'Pharaoh hound',
                     'Plott',
                     'Pointer',
                     'Pomeranian',
                     'Poodle',
                     'Portuguese water dog',
                     'Saint bernard',
                     'Silky terrier',
                     'Smooth fox terrier',
                     'Tibetan mastiff',
                     'Welsh springer spaniel',
                     'Wirehaired pointing griffon',
                     'Xoloitzcuintli',
                     'Yorkshire terrier']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     standard_normalization])
    
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(input_data)[:3,:,:].unsqueeze(0)
    image.to(device)
    with torch.no_grad():
        result = model(image)
    
    idx = torch.argmax(result.cpu())
    print(idx)
    print(class_names[idx])
    
    if dog_detector(image) is True:
        prediction = class_names[idx]
        return "Dogs Detected! It looks like a {0}".format(prediction)
    else:
        return "I don't know what it is but It looks like a {0}".format(prediction)
