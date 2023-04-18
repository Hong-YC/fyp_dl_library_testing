from utils.cmd_process import CmdProcess
from typing import List, Optional
import numpy as np
from pathlib import Path

class Inferencer(object):

    def __init__(self, db_manager, timeout):
        super().__init__()
        self.__db_manager = db_manager
        self.__timeout = timeout

    def inference(self, model_id: int, exp_dir: str, ok_backends: List[str]):
        """
        Run inference on each ok_backend
        """
        model_dir = Path(exp_dir) / 'models'
        training_inputs_path = Path(exp_dir) / 'dataset' / 'inputs.npz'

        outputs_dir = Path(exp_dir) / 'output'
        outputs_dir.mkdir(parents=True, exist_ok=True)

        crash_backends, nan_backends, inf_backends = [], [], []
        backends_output = {}

        cmd_processes = {
            bk: CmdProcess(f"python -m src.incons_detection.inference"
                           f" --backend {bk}"
                           f" --model_dir {str(model_dir)}"
                           f" --model_info_path {str(model_dir / 'model.json')}"
                           f" --training_instances_path {str(training_inputs_path)}"
                           f" --outputs_dir {str(outputs_dir / bk)}")
                           
            for bk in ok_backends
        }

        status = {}
        # Run each backend sequentially
        for bk, p in cmd_processes.items():  
            inference_status = p.run(self.__timeout)

            print(f"{bk}_status: {inference_status}")
            status[bk] = inference_status
            if inference_status:  # output crash
                crash_backends.append(bk)

            else:
                output_data = np.load(str(outputs_dir / bk/'output.npy'))
                backends_output[bk] = output_data

                if self.__check(output_data, np.isnan):  # If output nan
                    nan_backends.append(bk)
                if self.__check(output_data, np.isinf):  # If output inf
                    inf_backends.append(bk)


        # Record abnormality into the database
        if crash_backends:  # exist crash
            self.__db_manager.update_model_crash_backends(model_id, crash_backends)
        if nan_backends:  # exist nan
            self.__db_manager.update_model_nan_backends(model_id, nan_backends)
        if inf_backends:  # exist inf
            self.__db_manager.update_model_inf_backends(model_id, inf_backends)

        return status, backends_output, [bk for bk in ok_backends if bk not in crash_backends]

    def __check(self, w, f):
        if f(w).any():
            return True
        
        return False


