from src.cases_generation.model_generator import ModelGenerator
from utils.db_manager import DbManager
from utils.selection import Roulette
from pathlib import Path
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'


if __name__ == '__main__':
    db_manager = DbManager(str(Path.cwd() / 'data' / 'dummy.db'))
    

    from utils.utils import layer_types, layer_conditions
    selector =  Roulette(layer_types=layer_types,
                        layer_conditions=layer_conditions)

    generator = ModelGenerator(db_manager,selector, 10)

    exp_path = "/data/fyp23-dlltest/Hong/fyp_dl_library_testing"
    model_info_json_path = "/data/fyp23-dlltest/Hong/fyp_dl_library_testing/models/dummy_model.json"
    generator.generate(model_info_json_path, exp_path, 229)




