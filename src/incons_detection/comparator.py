import numpy as np
from pathlib import Path
import json
from itertools import combinations


class Comparator(object):

    def __init__(self, db_manager):
        super().__init__()
        self.__db_manager = db_manager
        self.__log_dir = Path(".")

    def compare(self, model_id: int, exp_dir: str, backends_outputs: dict, ok_backends: list):
        self.__log_dir = Path(exp_dir) / 'logs'

        # Load Model information
        json_path = Path(exp_dir) / 'models' / 'model.json'
        with open(str(json_path), 'r') as f:
            model_info = json.load(f)
        model_structure = model_info['model_structure']
        output_id_list = model_info['output_id_list']
        
        for bk1, bk2 in combinations(ok_backends, 2):
            outputs1, outputs2 = backends_outputs.get(bk1, None), backends_outputs.get(bk2, None)
            # Calculate model_output_delta
            model_output_delta = None
            try:
                model_output_delta = self.__model_output_delta(outputs1, outputs2)
                
                print(f"model_output_delta between {bk1} and {bk2}: {model_output_delta}")
            except Exception:
                import traceback
                # Create log file
                self.__log_dir.mkdir(parents=True, exist_ok=True)
                with (self.__log_dir / 'comparation.log').open(mode='a', encoding='utf-8') as f:
                    f.write(f"[ERROR] Crash when calculate inconsistencies between {bk1} and {bk2}\n")
                    traceback.print_exc(file=f)
                    f.write("\n\n")
 
            # Record into database
            incons_id = self.__db_manager.add_training_incons(model_id, f'{bk1}_{bk2}', model_output_delta)

            

    def __norm_delta(self, x, y, mode="max"):
        if x is None or y is None:
                return None
        # infinite norm
        if mode=="max":
            return float(np.max(np.abs(x-y)))
        # l2 norm
        elif mode=="l2":
            return float(np.sum((x-y)*(x-y)))
        else:
            return None
    

    def __model_output_delta(self, outputs1, outputs2):
        if outputs1 is None or outputs2 is None:
            return None

        return self.__norm_delta(outputs1, outputs2)