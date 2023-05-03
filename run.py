from alg import Experiment


Experiment.run()

# For measuring AIM accuracy.
# Experiment.run(["fitting", "aimtest", 
#                 "--seed=1", "--train-size=20000", "--aim=0.1", "--n-batch=50000",
#                 "--test-size=5000", "--explore=1", "--n-step=1", '--ablation=dense'])
