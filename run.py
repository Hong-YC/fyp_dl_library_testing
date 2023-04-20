from pathlib import Path
from typing import List, Optional
import datetime

from utils.utils import get_HH_mm_ss
from utils.db_manager import DbManager
from utils.selection import Roulette
from src.cases_generation.model_info_generator import ModelInfoGenerator
from src.cases_generation.model_generator import ModelGenerator
from src.cases_generation.data_generator import DataGenerator
from src.incons_detection.inferencer import Inferencer
from src.incons_detection.comparator import Comparator

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

class TrainingDebugger(object):
    def __init__(self, config: dict, use_heuristic: bool = True, generate_mode: str = 'template', timeout: float = 60):
        super().__init__()
        self.__output_dir = config['output_dir']
        self.__db_manager = DbManager(config['db_path'])
        # Roulette selector
        from utils.utils import layer_types, layer_conditions
        self.__selector = Roulette(layer_types=layer_types,
                                   layer_conditions=layer_conditions,
                                   use_heuristic=use_heuristic)
        self.__model_info_generator = ModelInfoGenerator(config['model'], generate_mode, self.__db_manager, self.__selector)
        self.__model_generator = ModelGenerator(self.__db_manager, self.__selector, timeout)
        self.__training_data_generator = DataGenerator(config['training_data'])
        self.__inferencer = Inferencer(self.__db_manager, timeout)
        self.__comparator = Comparator(self.__db_manager)

    def run_generation(self):
        """
        Generate models and data randomly
        """
        # Generate random Pytorch model
        print('Model generation start...')
        json_path, model_input_shapes, model_output_shapes, model_id, exp_dir = self.__model_info_generator.generate(save_dir=self.__output_dir)
        ok_backends = self.__model_generator.generate(json_path=json_path,
                                                      exp_dir=exp_dir,
                                                      model_id=model_id)
        

        print(f'Model generation complete: model_id={model_id} generation_ok_backends={ok_backends}')

        if len(ok_backends) >= 2:  
            print("Generate training data")
            self.__training_data_generator.generate(input_shapes=model_input_shapes,
                                                        output_shapes=model_output_shapes,
                                                        exp_dir=exp_dir)
            print("Data generation complete")
        

        return model_id, exp_dir, ok_backends
    
    def run_detection(self, model_id: int, exp_dir: str, ok_backends: List[str]):
        """
        Run differential testing on a model
        """
        
        if len(ok_backends) >= 2:
            # Inference Stage
            print('Start Inference...')
            status, backends_outputs, ok_backends = self.__inferencer.inference(model_id=model_id,
                                                                                                                                       exp_dir=exp_dir,
                                                                                                                                       ok_backends=ok_backends)
            print(f'Inference end: ok_backends={ok_backends}')

            self.__db_manager.record_status(model_id, status)

        if len(ok_backends) >= 2:
            # Comparator stage
            print('Compare start...')
            self.__comparator.compare(model_id=model_id,
                                    exp_dir=exp_dir,
                                    backends_outputs=backends_outputs,
                                    ok_backends=ok_backends)
            print('Compare end.')

        return ok_backends


def main(testing_config):
    config = {
        'model': {
            'var': {
                'tensor_dimension_range': (2, 5),
                'tensor_element_size_range': (5, 20),
                'weight_value_range': (-10.0, 10.0),
                'small_value_range': (0, 1),
                'vocabulary_size': 1001,
            },
            'node_num_range': (5, 5),
        },
        'training_data': {
            'instance_num': 1,
            'element_val_range': (0, 100),
        },
        'db_path': str(Path.cwd() / testing_config['data_dir'] / f'{testing_config["dataset_name"]}.db'),
        'output_dir': str(Path.cwd() / testing_config['data_dir'] / f'{testing_config["dataset_name"]}_output'),
        'report_dir': str(Path.cwd() / testing_config['data_dir'] / f'{testing_config["dataset_name"]}_report'),
        'distance_threshold': 0,
    }

    CASE_NUM = testing_config["case_num"]
    TIMEOUT = testing_config["timeout"]  # Second
    USE_HEURISTIC = bool(testing_config["use_heuristic"])  
    GENERATE_MODE = testing_config["generate_mode"]  # seq\merging\dag\template

    debugger = TrainingDebugger(config, USE_HEURISTIC, GENERATE_MODE, TIMEOUT)
    start_time = datetime.datetime.now()


    for i in range(CASE_NUM):
            print(f"######## Round {i} ########")
            try:
                print("------------- generation -------------")
                model_id, exp_dir, ok_backends = debugger.run_generation()
                print("------------- detection -------------")
                ok_backends = debugger.run_detection(model_id, exp_dir, ok_backends)
            except Exception:
                import traceback
                traceback.print_exc()
        

    end_time = datetime.datetime.now()
    time_delta = end_time - start_time
    h, m, s = get_HH_mm_ss(time_delta)
    print(f"Finish execution: Time used: {h} hour,{m} min,{s} sec")

if __name__ == '__main__':
    import json
    with open(str("testing_config.json"), "r") as f:
        testing_config = json.load(f)
    main(testing_config)
    