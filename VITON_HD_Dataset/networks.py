import torch
import torch.nn as nn

class ALIASNorm(nn.Module):
    def __init__(self, norm_type, input_nc, semantic_nc):
        print(f"🛠️ Debug: norm_type received in ALIASNorm = '{norm_type}'")  # Debugging print
        if not norm_type.startswith('alias'):
            raise ValueError(f"'{norm_type}' is not a recognized parameter-free normalization type in ALIASNorm")
        super(ALIASNorm, self).__init__()
        # Define normalization layers here...

class ALIASResBlock(nn.Module):
    def __init__(self, opt, input_nc, output_nc, use_mask_norm=True):
        super(ALIASResBlock, self).__init__()
        self.norm_0 = ALIASNorm(opt.norm_G, input_nc, opt.semantic_nc)

class ALIASGenerator(nn.Module):
    def __init__(self, opt, input_nc):
        super(ALIASGenerator, self).__init__()
        nf = opt.ngf  # ✅ Ensure ngf is defined
        self.head_0 = ALIASResBlock(opt, nf * 16, nf * 16)
        self.up_2 = ALIASResBlock(opt, nf * 4 + 16, nf * 2, use_mask_norm=False)

class GMM(nn.Module):
    def __init__(self, opt, inputA_nc, inputB_nc):
        super(GMM, self).__init__()
        self.regression = FeatureRegression(input_nc=(opt.ngf // 64) * (opt.ngf // 64))

class FeatureRegression(nn.Module):
    def __init__(self, input_nc):
        super(FeatureRegression, self).__init__()
        self.fc = nn.Linear(input_nc, input_nc)

print(">> networks.py loaded successfully!")
