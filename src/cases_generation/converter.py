import torch
import onnx
from .torch_model import TorchModel
import json
import argparse
import sys
from pathlib import Path

class Converter(object):

    def __init__(self, dir_torch, dir_json, dir_onnx):
        super().__init__()
        self.dir_torch=dir_torch
        self.dir_json=dir_json
        self.dir_onnx=dir_onnx
        
    def convert(self, version=12):

        with open(self.dir_json, 'r') as f:
            model_info = json.load(f)

        model_torch = TorchModel(model_info)
        model_torch.load_state_dict(torch.load(self.dir_torch))
        input_shape = model_info['model_structure']['0']['args']['shape']
        # Batch size equals 1 (TODO: Dynamic shape for batch size)
        input_shape.insert(0, 1)
        dummy_input = torch.ones(input_shape)

        torch.onnx.export(
            model_torch, dummy_input, self.dir_onnx,
            export_params=True,
            output_names=["out"],
            opset_version=version
        )

    

if __name__ == '__main__':
    # Obtain the parameters
    parse = argparse.ArgumentParser()
    parse.add_argument("--json_path", type=str)
    parse.add_argument("--torch_path", type=str)
    parse.add_argument("--output_dir", type=str)
    flags, _ = parse.parse_known_args(sys.argv[1:])
    onnx_path = Path(flags.output_dir) / f'model.onnx'

    converter = Converter(flags.torch_path, flags.json_path, onnx_path)

    try:
        converter.convert()

    except Exception:
        import traceback

        log_dir = Path(flags.output_dir).parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        with (log_dir / 'conversion.log').open(mode='a', encoding='utf-8') as f:
            f.write(f"[ERROR] Fail when converting PyTorch model to ONNX\n")
            traceback.print_exc(file=f)
            f.write("\n\n")

        sys.exit(-1)
            
