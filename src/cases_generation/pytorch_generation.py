import torch
import argparse
import sys
import json
from pathlib import Path
from .torch_model import TorchModel



if __name__ == '__main__':
    # Obtain the parameters
    parse = argparse.ArgumentParser()
    parse.add_argument("--json_path", type=str)
    parse.add_argument("--output_dir", type=str)
    flags, _ = parse.parse_known_args(sys.argv[1:])

    try:
        with open(flags.json_path, 'r') as f:
            model_info = json.load(f)

        model = TorchModel(model_info)

        # Save the model
        model_path = Path(flags.output_dir) / f'torch.pt'
        torch.save(model.state_dict(), model_path)

        
    except Exception:
        import traceback

        log_dir = Path(flags.output_dir).parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        with (log_dir / 'generation.log').open(mode='a', encoding='utf-8') as f:
            f.write(f"[ERROR] Fail when generating model with pyTorch\n")
            traceback.print_exc(file=f)
            f.write("\n\n")

        sys.exit(-1)