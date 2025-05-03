import os
import torch
import argparse
import cv2
import numpy as np
from options.test_options import TestOptions  # Ensure this file exists and is correctly set up
from datasets import VITONDataset  # Ensure correct dataset import
from networks import ALIASGenerator  # Ensure this is correctly implemented

# Initialize test options
opt = TestOptions().parse()
opt.ngf = 64  # Ensure ngf is defined
opt.norm_G = "batch"  # Ensure norm_G is defined
opt.semantic_nc = 13  # Ensure semantic_nc is defined
print("Options Loaded:", vars(opt))

# Load dataset
dataset = VITONDataset(opt, dataset_list=opt.dataset_list)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
print(f">> Dataset Loaded: {len(dataset)} samples.")

# Initialize model
model = ALIASGenerator(opt, input_nc=opt.inputA_nc)
model.eval()
print(">> Model initialized successfully!")

# Ensure output directory exists
output_dir = os.path.join(opt.results_dir, opt.name)
os.makedirs(output_dir, exist_ok=True)

# Process each image
for i, data in enumerate(dataloader):
    input_tensor = data['cloth'].cuda()  # Ensure correct key for input
    person_img = data['person'].cpu().numpy().transpose(0, 2, 3, 1)  # Convert to NumPy image
    
    with torch.no_grad():
        output = model(input_tensor)
    
    warped_cloth = output.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to NumPy
    warped_cloth = (warped_cloth * 255).astype(np.uint8)  # Rescale to 0-255
    
    # Overlay warped cloth on person image
    overlay_result = cv2.addWeighted(person_img[0], 0.6, warped_cloth[0], 0.4, 0)
    
    # Save output
    result_path = os.path.join(output_dir, f"result_{i}.png")
    cv2.imwrite(result_path, overlay_result)
    print(f">> Saved overlay result: {result_path}")

print(">> All images processed successfully!")
