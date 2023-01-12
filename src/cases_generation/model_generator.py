from typing import List, Optional
from pathlib import Path
import json

from utils.cmd_process import CmdProcess



class ModelGenerator(object):

    def __init__(self, weight_val_range: tuple, db_manager, selector, timeout):
        super().__init__()
        self.__weight_val_range = weight_val_range
        self.__db_manager = db_manager
        self.__selector = selector
        self.__timeout = timeout
    
    def generate(self, json_path: str, exp_dir: str, model_id: int):
        model_dir = Path(exp_dir) / 'models'

        # Generate model by Pytorch
        p = CmdProcess(f"python -m /data/fyp23-dlltest/Hong/fyp_dl_library_testing/pytorch_generation.py"
                f" --backend {bk}"
                f" --json_path {str(json_path)}"
                f" --weight_minv {self.__weight_val_range[0]}"
                f" --weight_maxv {self.__weight_val_range[1]}"
                f" --output_dir {str(model_dir)}")

        generate_status = p.run(self.__timeout)
