import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class QFunction(nn.Module):
    def __init__(self, n_actions = 30):
        super(QFunction,self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained = True)
        self.fc_1 = nn.Linear(1000,512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(512, n_actions)

        self.ad_fc_1 = nn.Linear(64*64, 512)
        self.ad_fc_2 = nn.Linear(512,256)
    def forward(self, observation):

        out = self.resnet(observation[:,:3])
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        addtional_observation = observation[:,3]
        addtional_observation = addtional_observation.reshape((addtional_observation.shape[0],-1))
        ad_out = F.relu(self.ad_fc_1(addtional_observation))
        ad_out = F.relu(self.ad_fc_2(ad_out))

        out = torch.cat((out,ad_out),dim = 1)
        out = F.relu(self.fc_3(out))

        return out