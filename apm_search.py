# Windows 10 Anconda python3.8
# coding=utf-8
'''
Author       : LiAo
Date         : 2022-07-25 10:55:32
LastEditTime : 2022-07-25 19:03:42
LastAuthor   : LiAo
Description  : Please add file description
'''
from nni.experiment import Experiment

# Step 1: Prepare the model - Done(apm_run.py)
# Step 2: Define search space

search_space = {
    'backbone': {'_type': 'choice', '_value': ['tf_efficientnetv2_b2', 'tf_efficientnetv2_b3', 'tf_efficientnetv2_s', 'tf_efficientnet_b2', 'tf_efficientnet_b3', 'tf_efficientnet_b4']},
    'pool': {'_type': 'choice', '_value': [True]},
    'pool_size': {'_type': 'choice', '_value': [(300, 300), (320, 320), (340, 340), (360, 360)]},
    'pool_type': {'_type': 'choice', '_value': ['avg', 'max', 'bilinear']},
    'lr': {'_type': 'choice', '_value': [1e-2, 5e-3, 1e-3, 8e-4]},
    'drop_rate': {'_type': 'choice', '_value': [0.5, 0.6, 0.7, 0.8]},
    'loss': {'_type': 'choice', '_value': ['focal', 'cross']}
}

# Step 3: Configure the experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python apm_run.py'
experiment.config.trial_code_directory = '.'

experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
# 最大实验次数
experiment.config.max_trial_number = 100
# 只有单卡, 所以只能设置为1
experiment.config.trial_concurrency = 8
# experiment.config.training_service.use_active_gpu = True
# experiment.config.maxTrialNumberPerGpu = 2
# experiment.config.gpuIndices = [0, 1, 2, 3]

# Step 4: Run the experiment
experiment.run(8122)

experiment.stop()
