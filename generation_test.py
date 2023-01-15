from src.cases_generation.model_generator import ModelGenerator



if __name__ == '__main__':

    generator = ModelGenerator(10)
    exp_path = "/data/fyp23-dlltest/Hong/fyp_dl_library_testing"
    model_info_json_path = "/data/fyp23-dlltest/Hong/fyp_dl_library_testing/models/dummy_model.json"
    generator.generate(model_info_json_path, exp_path)




