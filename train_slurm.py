import os
from numpy.random import randint
from models.mlp import DeepOrt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_obj = DeepOrt()
if 'SLURM_JOB_ID' in os.environ:
    model_obj.job_id = os.environ['SLURM_JOB_ID']
else:
    model_obj.job_id = int(''.join(["{}".format(randint(0, 9)) for num in range(0, 8)]))
model_obj.start_model()
