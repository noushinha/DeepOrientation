this repository is the implementation of [DeepOrientation: Deep Orientation Estimation of Macromolecules in Cryo-electron tomography]().

DeepOrt is a learning-based network for orientation estimation based on six degrees of freedom of the object (6DoF).
The network architecture includes a multi-layer perceptron.
The requirements.txt file shows the necessary packages to use the repository.

<h3>Data availability</h3>
Our data is available at: https://www.zib.de/ext-data/PolNet_Medium_Size_Dataset_4v4r_and_3j9i.zip

<h3>Training DeepOrt</h3>
In case you are running on a single GPU/CPU workstation, please simply run:

```python train_slurm.py```

In case you would like to run a slurm job, set the following paths in the ```submit_tf.sh``` file:
```# job error file
#SBATCH --error=<path/to/result/foldr/deeport_errors_%j.err>
# job output file
#SBATCH --output=<path/to/result/folder/deeport_output_%j.out>
```
Then using the following command, starts the training on the cluster.
``` sbatch submit_tf.sh```


In order to set training parameters, please use the ```config.py``` file in the main root.

The data folder should have the following structure:

```
data
 |____ <dataset name>
       |_____ npy
       |_____ res
```
The ```npy``` folder holds the npy files (euler, quaternions, 6dof), train and test data.
While the ```res``` folder is where the results will be dumped.

citation:

```
```

