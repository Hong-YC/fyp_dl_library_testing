import argparse
import sys
import json
from pathlib import Path
import numpy as np
from src.cases_generation.torch_model import TorchModel
import warnings

warnings.filterwarnings("ignore")

# Save the output of models
def save_output(output, output_dir):
    save_path = Path(output_dir) / "output.npy"
    np.save(save_path, output)
        

def __get_pytorch_output(model_dir: str, model_info_path: str, training_instances_path: str, output_dir: str):
    import torch

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    #prepare pytorch model
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    model = TorchModel(model_info)
    torch_model_path = str(Path(model_dir) / 'torch.pt')
    model.load_state_dict(torch.load(torch_model_path))
    model = model.to(device)

    #prepare input
    training_inputs = [*np.load(training_instances_path).values()]
    training_input = torch.from_numpy(training_inputs[0]).to(torch.float32)
    training_input = training_input.to(device)

    # Run inference
    O_torch=list(model(training_input).values())[0].detach().cpu().numpy()
    # print(O_torch)

    # Save the output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_output(O_torch, output_dir)


def __get_onnx_output(model_dir: str, model_info_path: str, training_instances_path: str, output_dir: str):
    import onnxruntime
    #prepare onnx model
    onnx_model_path = str(Path(model_dir) / 'model.onnx')
    sess = onnxruntime.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    #prepare input
    training_inputs = [*np.load(training_instances_path).values()]
    training_input = np.float32(training_inputs[0])

    # Run inference
    result = sess.run(None, {input_name: training_input})
    O_onnx=result[0]
    # print(O_onnx)

    # Save the output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_output(O_onnx, output_dir)


if __name__ == "__main__":
    # obtain parameters
    parse = argparse.ArgumentParser()
    parse.add_argument("--backend", type=str)
    parse.add_argument("--model_dir", type=str)
    parse.add_argument("--model_info_path", type=str)
    parse.add_argument("--training_instances_path", type=str)
    parse.add_argument("--outputs_dir", type=str)
    flags, _ = parse.parse_known_args(sys.argv[1:])

    try:
        if flags.backend == "pytorch":
            __get_pytorch_output(flags.model_dir, flags.model_info_path, flags.training_instances_path, flags.outputs_dir)
        elif flags.backend == "onnx":
            __get_onnx_output(flags.model_dir, flags.model_info_path, flags.training_instances_path, flags.outputs_dir)
        # elif flags.backend == "tensorflow":
        #     __get_tensorflow_output()

    except Exception:
        import traceback

        # Create Log file
        log_dir = Path(flags.outputs_dir).parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        with (log_dir / 'detection.log').open(mode='a', encoding='utf-8') as f:
            f.write(f"[ERROR] Crash when training model with {flags.backend}\n")
            traceback.print_exc(file=f)
            f.write("\n\n")

        sys.exit(-1)