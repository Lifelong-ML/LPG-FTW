import numpy as np
import ast
import itertools
import pandas as pd 

def append_result(job_name, results_prefix, results_list, num_tasks):
    try:
        with open(job_name + results_prefix + 'results.txt', 'r') as f:
            results_str = f.read()
        results_dict = ast.literal_eval(results_str)
    except:
        results_dict = {t: np.nan for t in range(num_tasks)}
    results_list.append(np.mean(list(results_dict.values())))

all_envs = ['halfcheetah','hopper', 'walker']
all_variations = ['gravity','bodyparts']

for env, variation in itertools.product(*[all_envs, all_variations]):
    num_seeds = 5
    if env == 'walker':
        num_tasks = 50
    else:
        num_tasks = 20

    jumpstart_results_ours = []
    jumpstart_results_stl = []
    jumpstart_results_ewc_shared_notscaled = []
    jumpstart_results_pgella = []

    finetuning_results_ours = []
    finetuning_results_stl = []
    finetuning_results_ewc_shared_notscaled = []
    finetuning_results_pgella = []

    forward_transfer_results_ours = []
    forward_transfer_results_stl = []
    forward_transfer_results_ewc_shared_notscaled = []
    forward_transfer_results_pgella = []

    results_ours = []
    results_stl = []
    results_ewc_shared_notscaled = []
    results_pgella = []

    for i in range(num_seeds):
        job_name_ours_seed = 'results/' + env +'_lpgftw_' + variation + '_exp/seed_{}/'.format(i)
        job_name_stl_seed = 'results/' + env + '_stl_' + variation + '_exp/seed_{}/'.format(i)
        job_name_ewc_shared_notscaled_seed = 'results/ewc_std_shared_notscaled/' + env + '_ewc_' + variation + '_exp/seed_{}/'.format(i)
        job_name_pgella_seed = 'results/' + env + '_pgella_' + variation + '_exp/seed_{}/'.format(i)


        append_result(job_name_ours_seed, 'start_', jumpstart_results_ours, num_tasks)
        append_result(job_name_stl_seed, 'start_', jumpstart_results_stl, num_tasks)
        append_result(job_name_ewc_shared_notscaled_seed, 'start_', jumpstart_results_ewc_shared_notscaled, num_tasks)

        append_result(job_name_ours_seed, 'finetune_', finetuning_results_ours, num_tasks)
        append_result(job_name_stl_seed, 'finetune_', finetuning_results_stl, num_tasks)
        append_result(job_name_ewc_shared_notscaled_seed, 'finetune_', finetuning_results_ewc_shared_notscaled, num_tasks)

        append_result(job_name_ours_seed, 'forward_transfer_', forward_transfer_results_ours, num_tasks)
        append_result(job_name_pgella_seed, 'forward_transfer_', forward_transfer_results_pgella, num_tasks)

        append_result(job_name_ours_seed, '', results_ours, num_tasks)
        append_result(job_name_ewc_shared_notscaled_seed, '', results_ewc_shared_notscaled, num_tasks)
        append_result(job_name_pgella_seed, '', results_pgella, num_tasks)

    results_df = pd.DataFrame(
        {'Start': ['{:.0f}\u00B1{:.0f}'.format(np.mean(jumpstart_results_ours), np.std(jumpstart_results_ours)/ np.sqrt(num_seeds)),
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(jumpstart_results_stl), np.std(jumpstart_results_stl) / np.sqrt(num_seeds)),
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(jumpstart_results_ewc_shared_notscaled), np.std(jumpstart_results_ewc_shared_notscaled) / np.sqrt(num_seeds)), 
                    '\u2014'],
        'Tune': ['{:.0f}\u00B1{:.0f}'.format(np.mean(finetuning_results_ours), np.std(finetuning_results_ours) / np.sqrt(num_seeds)), 
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(finetuning_results_stl), np.std(finetuning_results_stl) / np.sqrt(num_seeds)), 
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(finetuning_results_ewc_shared_notscaled), np.std(finetuning_results_ewc_shared_notscaled) / np.sqrt(num_seeds)), 
                    '\u2014'],
        'Update': ['{:.0f}\u00B1{:.0f}'.format(np.mean(forward_transfer_results_ours), np.std(forward_transfer_results_ours) / np.sqrt(num_seeds)), 
                   '\u2014', 
                   '\u2014', 
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(forward_transfer_results_pgella), np.std(forward_transfer_results_pgella) / np.sqrt(num_seeds))],

        'Final': ['{:.0f}\u00B1{:.0f}'.format(np.mean(results_ours), np.std(results_ours) / np.sqrt(num_seeds)),
                    '\u2014',
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(results_ewc_shared_notscaled), np.std(results_ewc_shared_notscaled) / np.sqrt(num_seeds)),
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(results_pgella), np.std(results_pgella) / np.sqrt(num_seeds))
                    ]
        }, index=['LPG-FTW', 'STL', 'EWC', 'PG-ELLA'])
    print(results_df.to_markdown() + '\n')