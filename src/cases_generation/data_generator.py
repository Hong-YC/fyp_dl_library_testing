from typing import Tuple, Optional
from pathlib import Path
import numpy as np


class DataGenerator(object):

    def __init__(self, config: int):
        super().__init__()
        self.__instance_num = config['instance_num']
        self.__ele_val_range = config['element_val_range']

    def generate(self, input_shapes: dict, data_dir: str, output_shapes: Optional[dict] = None):
        # create save path
        save_dir = Path(data_dir) / 'dataset'

        # Missing parent are created if needed, if the directory exists, do not throw an exception
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate dataset
        data_inputs_path = save_dir / 'inputs.npz'
        data_inputs = {input_name: self.__generate(input_shape) for input_name, input_shape in input_shapes.items()}
        np.savez(data_inputs_path, **data_inputs)

        # Generate ground_truth
        if output_shapes:
            ground_truths_path = save_dir / 'ground_truths.npz'
            ground_truths = {output_name: self.__generate(output_shape) for output_name, output_shape in output_shapes.items()}
            np.savez(ground_truths_path, **ground_truths)

    def __generate(self, shape: Tuple[Optional[int]]):
        a, b = self.__ele_val_range
        return np.random.rand(*(self.__instance_num, *shape[1:])) * (b - a) + a


if __name__ == '__main__':
    config = {
        'instance_num': 5,
        'element_val_range': (-1000, 1000),
    }
    vig = DataGenerator(config)
    input_shape = (None, 3, 32, 32)
    vig.generate({"test_input": input_shape}, "../data")
    path = "../data/dataset/inputs.npz"
    data = np.load(path)
    print(data["test_input"].shape)

