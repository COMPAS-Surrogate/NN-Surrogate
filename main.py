# -*- coding: utf-8 -*-
""" main.py """

from configs.config import CFG
from model.model import COMPAS_NN


def run():
    """Builds model, loads data, trains and evaluates"""
    model = COMPAS_NN(CFG)
    model.load_and_preprocess_data()
    model.build()
    model.train()
    model.evaluate()
    model.plot_results()


if __name__ == '__main__':
    run()