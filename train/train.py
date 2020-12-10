import argparse
import torch
import os
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from model import BreedClassifier
import json

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
    

def _get_loader(batch_size, data_dir,worker_class):
    train_dir = os.path.join(data_dir, 'train/')
    valid_dir = os.path.join(data_dir, 'valid/')
    standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         standard_normalization]),
                       'val': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         standard_normalization])
                      }
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_data = datasets.ImageFolder(valid_dir, transform=data_transforms['val'])
    
    trainloader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size, 
                                           num_workers=worker_class,
                                           shuffle=True)
    validationloader = torch.utils.data.DataLoader(val_data,
                                           batch_size=batch_size, 
                                           num_workers=worker_class,
                                           shuffle=False)
    loaders= {
        'train': trainloader,
        'valid': validationloader,
    }
    
    return loaders

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path, last_validation_loss=None):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    if last_validation_loss is not None:
        valid_loss_min = last_validation_loss
    else:
        valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            # initialize weights to zero
            optimizer.zero_grad()
            
            output = model(data)
            # calculate loss
            loss = criterion(output, target)
            
            # back prop
            loss.backward()
            
            # grad
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            
    # return trained model
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    
    # training paramas
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--worker-class', type=int, default=2, metavar='N',
                        help='worker class (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--change-model', type=int, default=7, metavar='N',
                        help='change model (default: 7)')
    
    
    #sagemaker paramas
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    
    
    args = parser.parse_args()
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    
    train_loader = _get_loader(args.batch_size, args.data_dir,args.worker_class)
    
    model = BreedClassifier()
    if args.change_model == 7:
        model = model.TransferLearning()
        model.to(device)
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr = args.lr)
    
        train(args.epochs, train_loader, model, optimizer, criterion, device, args.model_dir+'/model.pth')
        
    else:    
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = args.lr)

        train(args.epochs, train_loader, model, optimizer, criterion, device, args.model_dir+'/model.pth')
        
        
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
        'change_model': args.change_model
        }
        torch.save(model_info, f)