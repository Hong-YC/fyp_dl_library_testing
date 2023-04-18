from typing import List, Optional
from pathlib import Path
import json

from utils.cmd_process import CmdProcess



class ModelGenerator(object):

    def __init__(self, db_manager, selector, timeout):
        super().__init__()
        self.__db_manager = db_manager
        self.__selector = selector
        self.__timeout = timeout
    
    def generate(self, json_path: str, exp_dir: str, model_id: int):
        """
        Return libraries that is successfully generated
        """
        model_dir = Path(exp_dir) / 'models'
        
        
        # Generate model using Pytorch
        p = CmdProcess(f"python -m src.cases_generation.pytorch_generation"
                f" --json_path {str(json_path)}"
                f" --output_dir {str(model_dir)}")

        generate_status = p.run(self.__timeout)
        # generate_status: Success = 0, Fail = -1

        if generate_status:  # Generation Fail
            self.__db_manager.update_model_generate_fail_backends(model_id, ['pytorch'])
            # No model success
            return []

        torch_dir = model_dir/"torch.pt"
        
        p = CmdProcess(f"python -m src.cases_generation.converter"
                f" --json_path {str(json_path)}"
                f" --torch_path {str(torch_dir)}"
                f" --output_dir {str(model_dir)}")
        
        generate_status = p.run(self.__timeout)

        # generate_status: Success = 0, Fail = -1
        if generate_status:  # Generation Fail
            self.__db_manager.update_model_generate_fail_backends(model_id, ['onnx'])
            # ONNX conversion fail, only pytorch success
            return ["pytorch"]
        else:
            self.__update_selected_layers_cnt(str(json_path)) 

        # Both pytorch and onnx success
        return ["pytorch", "onnx", "tensorflow"]

    def __update_selected_layers_cnt(self, json_path):
        with open(json_path, 'r') as f:
            model_info = json.load(f)
            for layer_info in model_info['model_structure'].values():
                self.__selector.update(name=layer_info['type'])


if __name__ == '__main__':
    pass