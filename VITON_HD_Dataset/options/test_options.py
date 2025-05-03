import argparse

class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Test Options for VITON-HD")
        self.parser.add_argument('--name', type=str, default='vitonhd_test', help='Experiment name')
        self.parser.add_argument('--dataset_mode', type=str, default='test', help='Dataset mode')
        self.parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
        self.parser.add_argument('--dataset_list', type=str, required=True, help='Path to dataset list file')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='Directory to save results')
        self.parser.add_argument('--inputA_nc', type=int, default=3, help='Number of input channels')

    def parse(self):
        opt = self.parser.parse_args()
        return opt
