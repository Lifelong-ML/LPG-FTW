import numpy as np
import ast
import pandas as pd 


def append_result(job_name, results_prefix, results_list, num_tasks, suffix=''):
    try:
        with open(job_name + results_prefix + 'results' + suffix + '.txt', 'r') as f:
            results_str = f.read()
        results_dict = ast.literal_eval(results_str)
    except:
        results_dict = {t: 0 for t in range(num_tasks)}
        raise
    results_list.append(np.mean(list(results_dict.values())))

num_seeds = 5


environments = ['metaworld', 'metaworldMT50']

for env in environments:
    if env == 'metaworld':
        num_tasks = 10
    else:
        num_tasks = 48
    jumpstart_results_ours = []
    jumpstart_results_stl = []
    jumpstart_results_ewc_taskspecific_notscaled = []
    jumpstart_results_pgella = jumpstart_results_stl
    jumpstart_results_er = []

    finetuning_results_ours = []
    finetuning_results_stl = []
    finetuning_results_ewc_taskspecific_notscaled = []
    finetuning_results_pgella = finetuning_results_stl
    finetuning_results_er = []

    forward_transfer_results_ours = []
    forward_transfer_results_stl = finetuning_results_stl
    forward_transfer_results_ewc_taskspecific_notscaled = finetuning_results_ewc_taskspecific_notscaled
    forward_transfer_results_pgella = []
    forward_transfer_results_er = finetuning_results_er

    results_ours = []
    results_stl = finetuning_results_stl
    results_ewc_taskspecific_notscaled = []
    results_pgella = []
    results_er = []
    for i in range(num_seeds):
        job_name_ours_seed = 'results/' + env + '_lpgftw_exp/seed_{}/'.format(i)
        job_name_stl_seed = 'results/' + env + '_stl_exp/seed_{}/'.format(i)
        job_name_ewc_taskspecific_notscaled_seed = 'results/' + env + '_ewc_exp/seed_{}/'.format(i)
        job_name_pgella_seed = 'results/' + env + '_pgella_exp/seed_{}/'.format(i)
        job_name_er_seed = 'results/' + env + '_er_exp/seed_{}/'.format(i)


        append_result(job_name_ours_seed, 'start_', jumpstart_results_ours, num_tasks)
        append_result(job_name_stl_seed, 'start_', jumpstart_results_stl, num_tasks)
        append_result(job_name_ewc_taskspecific_notscaled_seed, 'start_', jumpstart_results_ewc_taskspecific_notscaled, num_tasks)
        append_result(job_name_er_seed, 'start_', jumpstart_results_er, num_tasks)

        append_result(job_name_ours_seed, 'finetune_', finetuning_results_ours, num_tasks)
        append_result(job_name_stl_seed, 'finetune_', finetuning_results_stl, num_tasks)
        append_result(job_name_ewc_taskspecific_notscaled_seed, 'finetune_', finetuning_results_ewc_taskspecific_notscaled, num_tasks)
        append_result(job_name_er_seed, 'finetune_', finetuning_results_er, num_tasks)

        append_result(job_name_ours_seed, 'forward_transfer_', forward_transfer_results_ours, num_tasks)
        append_result(job_name_pgella_seed, 'forward_transfer_', forward_transfer_results_pgella, num_tasks)

        append_result(job_name_ours_seed, '', results_ours, num_tasks, suffix='')
        append_result(job_name_ewc_taskspecific_notscaled_seed, '', results_ewc_taskspecific_notscaled, num_tasks)
        append_result(job_name_pgella_seed, '', results_pgella, num_tasks)
        append_result(job_name_er_seed, '', results_er, num_tasks, suffix='')

    print(env)
    results_df = pd.DataFrame(
        {'Start': ['{:.0f}\u00B1{:.0f}'.format(np.mean(jumpstart_results_ours), np.std(jumpstart_results_ours)/ np.sqrt(num_seeds)),
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(jumpstart_results_stl), np.std(jumpstart_results_stl) / np.sqrt(num_seeds)),
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(jumpstart_results_ewc_taskspecific_notscaled), np.std(jumpstart_results_ewc_taskspecific_notscaled) / np.sqrt(num_seeds)), 
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(jumpstart_results_er), np.std(jumpstart_results_er) / np.sqrt(num_seeds)), 
                    '\u2014'],
        'Tune': ['{:.0f}\u00B1{:.0f}'.format(np.mean(finetuning_results_ours), np.std(finetuning_results_ours) / np.sqrt(num_seeds)), 
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(finetuning_results_stl), np.std(finetuning_results_stl) / np.sqrt(num_seeds)), 
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(finetuning_results_ewc_taskspecific_notscaled), np.std(finetuning_results_ewc_taskspecific_notscaled) / np.sqrt(num_seeds)), 
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(finetuning_results_er), np.std(finetuning_results_er) / np.sqrt(num_seeds)), 
                    '\u2014'],
        'Update': ['{:.0f}\u00B1{:.0f}'.format(np.mean(forward_transfer_results_ours), np.std(forward_transfer_results_ours) / np.sqrt(num_seeds)), 
                   '\u2014', 
                   '\u2014', 
                   '\u2014', 
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(forward_transfer_results_pgella), np.std(forward_transfer_results_pgella) / np.sqrt(num_seeds))],

        'Final': ['{:.0f}\u00B1{:.0f}'.format(np.mean(results_ours), np.std(results_ours) / np.sqrt(num_seeds)),
                    '\u2014',
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(results_ewc_taskspecific_notscaled), np.std(results_ewc_taskspecific_notscaled) / np.sqrt(num_seeds)),
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(results_er), np.std(results_er) / np.sqrt(num_seeds)),
                    '{:.0f}\u00B1{:.0f}'.format(np.mean(results_pgella), np.std(results_pgella) / np.sqrt(num_seeds))
                    ]
        }, index=['LPG-FTW', 'STL', 'EWC', 'ER', 'PG-ELLA'])
    print(results_df.to_markdown() + '\n')




        
