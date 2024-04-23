# -*- coding: utf-8 -*-

"""Model config in json format"""

CFG = {
    "data": {
        "path_COMPAS_data": "/fred/oz016/Chayan/COMPAS_populations_project/COMPAS_lnl_data_Z_all.hdf",
        "path_mock_data": "/fred/oz016/Chayan/COMPAS_populations_project/mock_data.hdf",
        
    },
    "train": {
        "num_training_samples": 10000,
        "test_samples_frac": 0.2,
        "batch_size": 1024,
        "epoches": 100,
        "train_from_checkpoint": False,
        "checkpoint_path": '/fred/oz016/Chayan/COMPAS_populations_project/NN-Surrogate/checkpoints/Saved_checkpoint/', # if train_from_checkpoint == True
        "optimizer": {
            "type": "adam"
        },
    },
    "model": {
#        "input": [516,10],
        "timesteps": 15,

# For original model        
        
        "layers": {
            "CNN_layer_1": 64,
            "CNN_layer_2": 32,
            "LSTM_layer_1": 32,
            "LSTM_layer_2": 32,
            "LSTM_layer_3": 32,
            "Output_layer": 1,
            "kernel_size": 3,
            "pool_size": 2,
            "learning_rate": 1e-3
        },
        
    }
}
