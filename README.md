# How much do we really know about Structure Learning from Observational i.i.d. Data? Interpretable, multi-dimensional Evaluation Framework for Causal Disocvery

* This is the official github repository for the paper benchmarking seven different families of causal discovery techniques from the continuous and combinatorial branch on nonlinear i.i.d. data simulated with increasing nonidedntifiability. [[paper](https://arxiv.org/abs/2409.19377)]

## **Contributions**
In addition to our contribution in benchmarking context, we design a multi-dimensional, interpretable performance evaluation framework consisting of two components, i.e., DOS and EBM, based on requirements in causal disocvery w.r.t. holistic evaluation and interpretability:  <br>
![Alt text](Results_Visualization/Images/DOS_Int.PNG)


## **Key Results**
The linear combinatorial approach R2-SortnRegress and the pre-trained nonlinear continuous optimization method AVICI achieve the best results in terms of the six-dimensional performance indicator DOS. Additionaly, the analysis of the EBM importance and interaction score shows that the performance of these two methods is more robust to variations in the experimental experimental factors included in our simulation framework than the performance of the remaining models.

## **Reproducibility**
1. Install requirements. ```pip install -r requirements.txt```

2. Simulation of data, application of the 14 causal discovery techniques included in our study and evaluation of inferred graphs: You can use the notebook ```Causal_Graph_Inference_and_Evaluation.ipynb``` in the folder  ```./Src/```. The notebook shows how to define the values for the experimental factors of the synthetic data generating process, and how to run and evaluate the models on a demo-version of the simulation (smaller proportion of the simulated datasets).

3. Generation of visualizations included in the results section of the study:
You can use the notebook ```Large_Scale_Sensitivity_Analysis.ipynb``` in the folder  ```Src```. The class for generating the visualizations is in the folder  ```./Results/Results_Visualization/```.


## **Acknowledgements**

We appreciate the following github repositories very much for the valuable code base and datasets:

gCastle: https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle

DAGMA: https://github.com/kevinsbello/dagma/blob/main/README.md

DAS (DoDiscover): https://github.com/py-why/dodiscover

NoTears: https://github.com/xunzheng/notears

NoCurl: https://github.com/fishmoon1234/DAG-NoCurl

R2-SortnRegress (CausalDisco): https://github.com/CausalDisco/CausalDisco


## **Contact**

If you have any questions or concerns, please contact us: velegeor@hu-berlin.de, stefan.lessmann@hu-berlin.de. 

## **Citation**

If you find this repo useful in your research, please consider citing our paper as follows:

```
@article{VelevLessmann-2024-CausalDiscovery,
  title     = {How much do we really know about Structure Learning from Observational i.i.d. Data? Interpretable, multi-dimensional EValuation Framework for Causal Disocvery},
  author    = Velev, Georg and
              Lessmann, Stefan},
  journal = {arxiv preprint arXiv:2409.19377},
  year      = {2024}
}
```

