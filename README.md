# Overview

The code  repository for the paper "A data-driven method to learn a jump diffusion process from aggregate biological gene expression data"

# Code 
## Reproduction of the paper results

1. The **Synthetic** folder reproduces the results in Section 3.1.
2. The **Real** folder reproduces the results in Section 3.2, Section 3.3, and the supplementary material.
3. The **WorkflowPlot** folder is used to reproduce Figure 1, and thus gives the overview of our algorithm.

**Notice**: To speed up the reproduction process,  we provide the models that have been well trained in the **TrainedModel** folder. 

## Perform our algorithm on new data
We provide the file **TrainingFunction.ipynb**  for practitioners to use our model.  The main function is **Train**, and an example is given to show the training process.

```python
Train(AggregateData,n_steps,bd,intensity,num_hidden,dim_hidden,step_size=0.05,n_epochs=30000,n_critic=4,lr=0.0001,Seed=80)
```

The comments on these parameters are also given in the **TrainingFunction.ipynb** file.

# Dataset
To use on your own machine, the gene expression data  at four days (four csv files) should be placed in the root directory.

Additionally, the embryonic stem cell differentiation dataset is from  [Klein et al. (2015)]( https://pubmed.ncbi.nlm.nih.gov/26000487/)  and is available at [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525). 

The four files are linked at:

D0: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1599494](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1599494).

D2:[https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1599497](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1599497).

D4:[https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1599498](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1599498).

D7:[https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1599499](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM1599499).

