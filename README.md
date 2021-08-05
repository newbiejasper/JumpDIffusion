# Learning jump diffusion behavior from aggregate gene expression data

## Motivation

Dynamic models of gene expression are urgently required. Different from trajectory inference and RNA velocity, a more direct method can be applied to reconstruct long-time developmental trajectories for individual cells.

## Data 

Assuming that we have observed totally $J$ time points during the whole time interval $[0,T]$, where $0=T_0, T_1, \cdots, T_{J-1} = T$ and the time partition may not be equal. At these time points, there are $N_i(i=0,1,\cdots,J-1)$ i.i.d. samples $\{X_{T_i}^j\}_{j=1}^{N_i}(i=0,1,\cdots,J-1)$ that we term aggregate observations. The individuals $\{X_{T_i}^j\}_{j=1}^{N_i}$ observed at time $T_i$ are often not identical to those $\{X_{T_{i-1}}^j\}_{j=1}^{N_{i-1}}$ observed at the previous time $T_{i-1}$. 

## Implement

### Workflow
File `plotting workflow for our algorithm.ipynb` is used to give an overview of our algorithm.


### Synthetic dataset
In the `Synthetic` folder.
`OU.py`, `SDEnoJump.py`, `SDEexpJump.py`, and `SDEgauJump.py` are four methods fitting the synthetic dataset. 

The outputs include:

Wasserstein loss at $3\delta$: `Sou1`,`Snj1`,`Swj1`,`Swgj1`.

Wasserstein loss at $6\delta$: `Sou2`,`Snj2`,`Swj2`,`Swgj2`.

Generator $G$: `netGSou.pt`,`netGSnj.pt`,`netGSwj.pt`,`netGSwgj.pt`.

The file `four methods on synthetic dataset.ipynb` compares the overall performance of the four methods. 

### Real dataset

#### Cell differentiation dataset

##### Prediction task
In the `Real/PredictionTask` folder.

Predicting D4 based on D0, D2, D7: `OU_Task1.ipynb`,`SDEnoJump_Task1.ipynb`,`SDEgauJump_Task1.ipynb`.

Outputs: `OU_Task1.pt`,`SDEnoJump_Task1.pt`,`SDEgauJump_Task1.pt`.
Overall comparison: `Task1_comparison.ipynb`.

Predicting D7 based on D0, D2, D4: `OU_Task2.ipynb`,`SDEnoJump_Task2.ipynb`,`SDEgauJump_Task2.ipynb`.

Outputs: `OU_Task2.pt`,`SDEnoJump_Task2.pt`,`SDEgauJump_Task2.pt`.
Overall comparison: `Task2_comparison.ipynb`.

##### Trajectory, Velocity, Cluster
In the `Real/AllTime` folder.

Training with all the four days: `AllTimeTraining.py`.
Outputs: `AllTimeTrainingError`, `AllTimeTrainingnetG.pt`.
Overall computation: `TrajectoryVelocityCluster.ipynb`.

Training with another 14 genes: `markergene.ipynb`.

#####  Stability
In the `Real/Stability` folder.

###### Permutations of data

Different noise levels: `epsilon0.1.ipynb`,`epsilon0.01.ipynb`,`epsilon0.001.ipynb`.
Outputs: `epsilon0.1.pt`,`epsilon0.01.pt`,`epsilon0.001.pt`.

Analysis of the noise effects: `PerturbationOfData.ipynb`.

###### Hyperparameters

`AllTime.py` tries different hyperparameter combinations, and outputs `log.out`. Those parameter combinations that achieve lower training errors are used to train different models, and the results are placed in the folder `hyperparameter40`. The analysis of hyperparameters is at `Hyperparameter.ipynb`.

##### Monocle
In the `Real/monocle` folder.

`monocle.ipynb` outputs the monocle results.

#### Cell cycle dataset
In the `Real/Cell_cycle` folder.

`cycle-phate.ipynb` trains the cell cycle data `layout.txt`.

### DM v.s. DTM
In the folder `DM-DTM`.

`ConvergenceRate.ipynb` compares the  performance of DM and DTM on different models and 1-Lipschitz continuous test functions.
