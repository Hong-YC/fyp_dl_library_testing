

def get_layer_func(layer_type):
    import torch
    return getattr(torch.nn, layer_type)


seq_layer_types = [
    'Linear',
    'Conv2d',


]

layer_types = seq_layer_types



if __name__ == '__main__':
    c = get_layer_func("Conv1d")
    print(c(10, 20, 4))
