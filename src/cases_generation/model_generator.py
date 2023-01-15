from typing import List, Optional
from pathlib import Path
import json

from utils.cmd_process import CmdProcess



class ModelGenerator(object):

    def __init__(self, timeout):
        super().__init__()
        # Hong: add database and selector support
        # self.__db_manager = db_manager
        # self.__selector = selector
        self.__timeout = timeout
    
    def generate(self, json_path: str, exp_dir: str):
        model_dir = Path(exp_dir) / 'models'

        # Generate model by Pytorch
        p = CmdProcess(f"python -m src.cases_generation.pytorch_generation"
                f" --json_path {str(json_path)}"
                f" --output_dir {str(model_dir)}")

        generate_status = p.run(self.__timeout)
