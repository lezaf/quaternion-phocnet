import argparse

import torch
from   src.models.quaternion_phocnet_2 import QuaternionPHOCNet
from   src.utils.save_load             import my_torch_load


# Process cmd arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model_path', '-mp', action='store', required=True,
                    help='Path to the model: <model_path>.pt')

args = parser.parse_args()

# Count parameters
qcnn = QuaternionPHOCNet(n_out=252,
                         in_channels=4,
                         gpp_type='spp',
                         pooling_levels=3)

my_torch_load(qcnn, args.model_path)

model_total_params = sum(p.numel() for p in qcnn.parameters() if p.requires_grad)
print('Number of trainable parameters: ', model_total_params)
