import torch
import os
from datasets import VITONDataset
from networks import GMM, ALIASGenerator
from utils import load_checkpoint
from PIL import Image

class Opt:
    def __init__(self):
        self.inputA_nc = 3
        self.inputB_nc = 3
        self.load_width = 192
        self.load_height = 256
        self.dataset_dir = "datasets/custom_inputs/"
        self.dataset_list = "datasets/custom_inputs/test_pairs.txt"
        self.checkpoint_dir = "checkpoints/"
        self.results_dir = "results/custom_test/"

opt = Opt()
os.makedirs(opt.results_dir, exist_ok=True)

# ✅ Load dataset
test_dataset = VITONDataset(opt.dataset_dir, opt.dataset_list)

# ✅ Load models
gmm = GMM(opt, opt.inputA_nc, opt.inputB_nc)
alias_generator = ALIASGenerator()

# ✅ Load checkpoints
load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, "gmm_final.pth"))
load_checkpoint(alias_generator, os.path.join(opt.checkpoint_dir, "alias_final.pth"))

gmm.eval()
alias_generator.eval()

# ✅ Process images
for image_name, cloth_name in test_dataset.samples:
    image_path = os.path.join(opt.dataset_dir, "image", image_name)
    cloth_path = os.path.join(opt.dataset_dir, "cloth", cloth_name)

    if not os.path.exists(image_path) or not os.path.exists(cloth_path):
        print(f"Skipping {image_name} - Missing file")
        continue

    data = test_dataset.__getitem__(test_dataset.samples.index([image_name, cloth_name]))

    with torch.no_grad():
        warped_cloth = gmm(data)
        final_output = alias_generator(data, warped_cloth)

    output_path = os.path.join(opt.results_dir, f"{image_name}_result.jpg")
    final_output.save(output_path)
    print(f"Saved output to: {output_path}")

print("🎉 All images processed successfully!")
