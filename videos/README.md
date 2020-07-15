# LPG-FTW Videos

Here we include training videos for [Lifelong Policy Gradient Learning of Factored Policies for Faster Training Without Forgetting (Mendez et al., 2020)](https://arxiv.org/abs/2007.07011).

## OpenAI MuJoCo Domains

Videos on MuJoCo domains, in the `mujoco/` directory, show the learning progress of LPG-FTW versus STL on six different domains. To create these videos, we considered the last task learned by each algorithm on the first random seed, and, for every 5-th learning iteration, we sampled one trajectory using the policy learned up to that point. This ensures that the trajectories shown here are not cherry-picked, and so are representative of the true learning process. In fact, one of the six videos (labeled `_negative_result` in the filename) shows STL performing better than LPG-FTW. 

## Meta-World Domains

Meta-World videos, in the `metaworld/` directory, show the learning progress of LPG-FTW compared to STL on 10 of the 48 Meta-World tasks. In this case, to ensure the visualized tasks are both 1) diverse and complex enough to showcase the power of LPG-FTW and 2) learned in a sequence that represents the full learning sequence of LPG-FTW, we created a semi-random training curriculum. In particular, we created a random permutation of the task order which ensured that the following task sequence was maintained: 

```
0 -> Reach
5 -> Coffee Button
10 -> Hammer
15 -> Soccer
20 -> Door Unlock
25 -> Faucet Open
30 -> Basketball
35 -> Box Close
40 -> Window Open
47 -> Coffee Push
```

Note that this is different from the purely random task order used in the experiments in our paper, and is only used for visualization purposes.