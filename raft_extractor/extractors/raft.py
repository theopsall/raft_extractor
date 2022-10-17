import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.utils import flow_to_image


class Raft(nn.Module):
    def __init__(self, output_layers:list, flatten: bool = False,*args) -> None:
        super(Raft, self).__init__(*args)
        self.device = "cpu"
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_large(weights=self.weights, progress=False).to(self.device)
        self.model.eval()
        self.output_layers = output_layers
        #print(self.output_layers)
        self.selected_out = OrderedDict()
        self.fhooks = []





        for name,layer in self.model.named_modules():
            if name in self.output_layers:
                res = re.search(r"\.\d+", name)
                if res.group(0) is not None:
                    new_name = name.replace(f'{res.group(0)}', f'[{res.group(0).replace(".","")}]')
                else :
                    new_name = name
                layer = eval(f'self.model.{new_name}')
                self.fhooks.append(layer.register_forward_hook(self.forward_hook(name)))
        
                
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook
    
    def extract(self):
        pass

    def forward(
        self, img1_batch: torch.Tensor, img2_batch: torch.Tensor
    ) -> torch.Tensor:

        list_of_flows = self.model(
            img1_batch.to(self.device), img2_batch.to(self.device)
        )
        return flow_to_image(list_of_flows[-1]), self.selected_out
