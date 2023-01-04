# Explainable Reinforcement Learning via Causal World Models
by Zhongwei Yu

This repository implements Explainable Causal Reinforcement Learning with attention.

## 1. Requirement

The requirement file of the python environment is provided in `requirement.txt`.

To create the required conda environment, use
```
conda create --name <env> --file requirement.txt
```

Our experiments involve StarCraftII Learning Enviroment. The StarCraftII program and maps can be downloaded from https://github.com/Blizzard/s2client-proto. Make sure you have installed StartCraftII and pysc2 correctly!

## 2. Usages
The main usages of this code is provided by `run.py`. It executes experiment commands:
```
python run.py <command> <arguments>
```
The supported commands are:
* `model-based`: train a policy using environment models.
* `model-free`: train a policy using a model-free algorithm (PPO).
* `fitting`: fit a model for a policy (deprecated)
* `train-explain`: train explainatory models for a policy.
* `test-explain`: present explanation examples.

Type this in your console to see the arguments of the commands:
```
python run.py <command> -h
```

For example, to train a policy for the Build-Marine environment using models:
```
python run.py model-based buildmarine --seed=1 --run-id=run-1
```
The results and log files are saved in the `experiments\` directory.

### 2.1. Supported Environments

We support 4 environments:
- `lunarlander` for Lunarlander environment with discrete action space
- `lunarlander` with argument `--continuous` for Lunarlander environment with continuous action space
- `buildmarine` for Build-Mrine environment
- `cartpole` for Cartpole environment

### 2.2. Hyper-parameters

Most hyper-parameters are managed by the `Config` object defined in `learning\config.py`. Default config is specified in `alg\_env_setting.py`.

we may also use specified config file for each experiment simply using the argument `--config=xxx.json`. the config files used for our main experiments are in the `configs` directory.

## 3. Work Flow

1. Execute the `model-based` or `model-free` command with `run.py` to learn a policy on a given environment. This will create an experiment directory in `experiments\`. Mostlikely, it will be `experiments\<env_id>\<model-based|model-free>\run-xxx`
2. If you used the `model-based` command:
    1. Go to the experiment directory, you should find the saved `actor.nn`, `env-model-x.nn` and `causal-graph.json`.
    2. Rename any environment model to `explain-env-model.nn` and the causal graph to `explain-causal-graph.json`.
3. Otherwise:
    1. go to the experiment directory, you should find the saved `actor.nn`.
    2. execute the following command using `run.py`:
        ```
        train-explain <your experiment direcotry> [--n-sample] [--n-step]
        ```
        to train a post-hoc model. When completed, you shall find `explain-env-model.nn` and `explain-causal-graph.json` in the experiment directory.
4. Execute `test-explain <your experiment directory>` using `run.py` to see examples of causal chains. This command starts an interaction cycle.