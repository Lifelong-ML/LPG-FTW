import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import csv
from mjrl.utils.logger import DataLog

def make_train_plots(log = None,
                     log_path = None,
                     keys = None,
                     save_loc = None):
    if log is None and log_path is None:
        print("Need to provide either the log or path to a log file")
    if log is None:
        logger = DataLog()
        logger.read_log(log_path)
        log = logger.log
    # make plots for specified keys
    for key in keys:
        if key in log.keys():
            plt.figure(figsize=(10,6))
            plt.plot(log[key])
            plt.title(key)
            plt.savefig(save_loc+'/'+key+'.pdf')#, dpi=100)
            plt.close()

def make_multitask_train_plots(loggers,
                                keys = None,
                                save_loc = None):
    for key in keys:
        sum_key = 0
        for logger in loggers.values():
            if key in logger.log.keys():
                sum_key += np.array(logger.log[key])
        plt.figure(figsize=(10,6))
        plt.plot(sum_key / len(loggers))
        plt.title(key)
        plt.savefig(save_loc+'/'+key+'.pdf')#, dpi=100)
        plt.close()

def make_multitask_test_plots(mean_test_perf, save_loc):
    plt.figure(figsize=(10,6))
    plt.plot(list(mean_test_perf.values()))
    plt.title('Test performance')
    plt.savefig(save_loc+'/test_perf.pdf')
