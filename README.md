# LPG-FTW

This is the source code used for [Lifelong Policy Gradient Learning of Factored Policies for Faster Training Without Forgetting (Mendez et al., 2020)](https://arxiv.org/abs/2007.07011).

This package contains the implementations of two variants of LPG-FTW, with REINFORCE and NPG as the base learners. Code used for running experiments is provided in the `experiments/` directory. Before running any experiment for the MuJoCo tasks, you must create the tasks by running the relevant script from the `experiments/mtl_*_tasks/create_tasks` directory. The main files containing the code for LPG-FTW are 

* `mjrl/utils/algorithms/`
    * `batch_reinforce_ftw.py`
    * `npg_cg_ftw.py`
* `mjrl/utils/policies/`
    * `gaussian_linear_lpg_ftw.py` 
    * `gaussian_mlp_lpg_ftw.py`

The main dependencies are `python=3.7`, `gym`, `mujoco-py=1.50`, `gym-extensions-mod`, and `pytorch=1.3`. The first three dependencies are included in this package for ease of installation. The fourth dependency is a modified version of the `gym-extensions` package, as described in the paper. The final dependency is `metaworld`, also included in this package.


## Installation instructions (Linux)

- Download MuJoCo binaries from the official [website](http://www.mujoco.org/) and also obtain the license key.
- Unzip the downloaded mjpro150 directory into `~/.mujoco/mjpro150`, and place your license key (mjkey.txt) at `~/.mujoco/mjkey.txt`
- Install osmesa related dependencies:
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev build-essential libglfw3
```
- Update `bashrc` by adding the following lines and source it
```
export LD_LIBRARY_PATH="<path/to/.mujoco>/mjpro150/bin:$LD_LIBRARY_PATH"
export MUJOCO_PY_FORCE_CPU=True
alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so'
```
- Install this package using

```
$ conda update conda
$ cd path/to/lpg_ftw
$ conda env create -f env.yml
$ conda activate lpg-ftw
$ cd gym/mujoco-py
$ pip install -e .
$ cd ../
$ pip install -e .
$ cd ../gym-extensions-mod
$ pip install -e .
$ cd ../metaworld
$ pip install -e .
$ cd ..
$ pip install -e .
```



## Reproducing results

Tables 1 and 2 contain the main results in the paper. 

To reproduce the results, you must first train policies for each of the algorithms as below:

```
$ python experiments/metaworld_tasks/metaworld_[algorithm].py
$ python experiments/metaworld_tasks/metaworldMT50_[algorithm].py
```

Note that PG-ELLA must be trained _after_ STL, since it uses the STL pre-trained policies. Then, to evaluate the policy at different stages of training (start, tune, and update), execute:

```
$ python experiments/[stage]/metaworld_[algorithm].py
$ python experiments/[stage]/metaworldMT50_[algorithm].py
```

This will save all the files needed for creating the tables. Then simply execute the following command to generate the tables:

```$ python mjrl/utils/make_results_tables_metaworld.py```

To generate the tables. Instructions for recreating results on the OpenAI domains are almost identical, replacing the `metaworld_tasks/` directory for `mtl_bodypart_tasks/` or `mtl_gravity_tasks/` where appropriate, and using `make_results_tables_openai.py` for creating the tables.



Table 1: Results of LPG-FTW on MT10 (equivalent to Figure 3.a (bottom))
|         | Start    | Tune         | Update      | Final       |
|:--------|:--------:|:------------:|:-----------:|:-----------:|
| LPG-FTW | 4523±508 | 160553±2847  | 161142±3415 | 154873±5415 |
| STL     | 5217±100 | 135328±8534  | —           | —           |
| EWC     | 5009±407 | 145060±12859 | —           | 22202±5065  |
| ER      | 3679±650 | 48495±7141   | —           | 5083±1710   |
| PG-ELLA | —        | —            | 44796±4606  | 12546±5448  |


Table 2: Results of LPG-FTW on MT50 (equivalent to Figure 3.b (bottom))
|         | Start    | Tune        | Update      | Final       |
|:--------|:--------:|:-----------:|:-----------:|:-----------:|
| LPG-FTW | 2505±166 | 161047±3497 | 161060±3892 | 160739±3933 |
| STL     | 4308±128 | 136837±2888 | —           | —           |
| EWC     | 1001±150 | 71168±16915 | —           | 567±120     |
| ER      | 3373±310 | 33323±2229  | —           | 25389±1910  |
| PG-ELLA | —        | —           | 10292±1113  | 125±130     |

## Video Results

Please navigate to the `videos/` directory to visualize the training process of LPG-FTW on the MuJoCo and Meta-World domains.

## Citing LPG-FTW

If you use LPG-FTW, please cite our paper

```
@article{mendez2020lifelong,
  title={Lifelong Policy Gradient Learning of Factored Policies for Faster Training Without Forgetting},
  author={Mendez, Jorge A. and Wang, Boyu and Eaton, Eric},
  journal={arXiv preprint arXiv:2007.07011},
  year={2020}
}
```

---
This package was built on the [mjrl](https://github.com/aravindr93/mjrl) package. The README for the original package can be found [here](README_mjrl.md)
