import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BreedClassifier(nn.Module):
    def __init__(self):
        super(BreedClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        # pool
        self.pool = nn.MaxPool2d(2, 2)
        
        # fully-connected
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 133) 
        
        # drop-out
        self.dropout = nn.Dropout(0.3)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        # flatten
        x = x.view(-1, 512)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
    def TransferLearning(self):
#         model = models.resnet50(pretrained=True)
#         for param in model.parameters():
#             param.requires_grad = False
#         model.fc = nn.Linear(2048, 133, bias=True)
#         for param in model.fc.parameters():
#             param.requires_grad = True
#         return model
        model = models.vgg19(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
            
        n_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_inputs, 133)
        return model
        
        
    