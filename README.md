# Common Mechanism 

Sourse code for the paper  [Answering Private Queries Adaptively using the Common Mechanism](https://arxiv.org/abs/2212.00135).

## Description

Given two linear Gaussian Mechanisms, the Common Mechanism captures the common information shared by these two mechanism. With the help of the Common Mechanism, analyst can make statistical inference without spending extra privacy loss budget in some problems.

The algorithms are implemented in the following files. 

-  **brazil/CM_rand.py**:  Calculate the Common Mechanism of two linear Gaussian Mechanisms
-  **agegender/class_finegrain.py**: Calculate the Common Mechanism of three or more linear Gaussian Mechanisms



## Usage

The following code gets the Common Mechanism of two linear Gaussian Mechanisms. You need to specify `B1, S1` as the query matrix and covariance matrix for mechanism 1, `B2, S2` as the query matrix and covariance matrix for mechanism 2. This method uses **analytical solution** to get the common mechanism.

```python
# import the CM_rand.py in the brazil directory 
from CM_rand import config, CM

args = config()
# Mechanism 1: (B1, S1)
# Mechanism 2: (B2, S2)
com_mech = CM(args, B1, S1, B2, S2)
```

The following code gets the Common Mechanism of three or more linear Gaussian Mechanisms. You need to specify `B1, S1` as the query matrix and covariance matrix for mechanism 1, `B2, S2` as the query matrix and covariance matrix for mechanism 2. This method uses **numerical solver** to get the common mechanism.

```python
# import the class_finegrain.py in the agegender directory 
from class_finegrain import config, Mechanism, CommonMechanism

args = config()
# Mechanism 1: (B1, S1)
# Mechanism 2: (B2, S2)
# Mechanism 3: (B3, S3)
mech1 = Mechanism(args, B1, S1)
mech2 = Mechanism(args, B2, S2)
mech3 = Mechanism(args, B3, S3)
com_mech = CommonMechanism(args, [mech1, mech2, mech3])
```


## Datasets
The datasets are provided as zip files, you need to unzip them and put them in a proper directory.
- **AgeGender.zip**: It contains a .npy file that combine data blocks in each state.
- **HispRace.zip**: Each .npy file contains data blocks in one state. 
- **brazil.zip**: It groups the original Brazil Census data by (State, Occupation), the .npy contains data blocks after the grouping. (The brazil.npy should be placed inside the brazil folder, otherwise you need to specify the path to it)


## Run Experiment

The following files contain codes for experiments in the paper. Note that due to randomness and different machine performance, the result may not be exactly the same as the numbers in the paper. 
- **hisprace** Marginals on HispRace (section 7.3)
  - Because the hisprace dataset is large (~3GB), we use [dask](https://www.dask.org/) to run the code in parallel.
- **brazil** Marginals on the Brazil Dataset (section 7.4).
  - We recommend running this experiment instead of the hisprace experiment, because the dataset is smaller and the code runs faster.
- **agegender** Census Application: Age/Gender Histograms (section 7.5).
  - It gives a general strategy to choose among 4 mechanisms.

To compare with the optimal Gaussian Mechanism mentioned in section 7.2, you need to use the solver [SM-II](https://github.com/cmla-psu/matrixqueries), **SM-II/7_common.py** shows how to use the solver. You need to save the query matrix $B_{com}$ and the covariance matrix $S_{com}$ of the common mechanism to a .npy file.
```python
track = {}
track['mat'] = com_mech.B_com
track['cov'] = com_mech.S_com
np.save("cm-brazil.npy", track, allow_pickle=True)
```


## Citing this work

You are encouraged to cite the paper [Answering Private Queries Adaptively using the Common Mechanism](https://arxiv.org/abs/2212.00135) if you use this tool for academic research:

```bibtex
@misc{xiao2022answering,
      title={Answering Private Linear Queries Adaptively using the Common Mechanism}, 
      author={Yingtai Xiao and Guanhong Wang and Danfeng Zhang and Daniel Kifer},
      year={2022},
      eprint={2212.00135},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```



## License

[MIT](https://github.com/cmla-psu/commonmech/blob/main/LICENSE).
