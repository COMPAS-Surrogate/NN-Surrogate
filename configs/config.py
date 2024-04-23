# -*- coding: utf-8 -*-

"""Model config in json format"""

CFG = {
    "data": {
        "path_COMPAS_data": "/fred/oz016/Chayan/COMPAS_populations_project/COMPAS_lnl_data_Z_all.hdf",
        "path_mock_data": "/fred/oz016/Chayan/COMPAS_populations_project/mock_data.hdf",
        
    },
    "train": {
        "num_training_samples": 10000,
        "dataset": 'COMPAS',
        "test_samples_frac": 0.2,
        "validation_samples_frac": 0.25,
        "batch_size": 32,
        "epoches": 100,
        "train_from_checkpoint": False,
        "checkpoint_path": '/fred/oz016/Chayan/COMPAS_populations_project/NN-Surrogate/checkpoints/Saved_checkpoint/',
        "model_path": '/fred/oz016/Chayan/COMPAS_populations_project/NN-Surrogate/checkpoints/',
        "optimizer": {
            "type": "adam"
        },
    },
    "model": {
        "learning_rate": 1e-3,
        
    }
}
