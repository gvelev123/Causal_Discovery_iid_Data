# Interpretable, multi-dimensional Evaluation Framework for Causal Discovery from observational i.i.d. Data

* This is the official github repository for the paper benchmarking seven different families of causal discovery techniques from the continuous and combinatorial branch on nonlinear i.i.d. data simulated with increasing nonidedntifiability. [[paper]([[https://arxiv.org/abs/2409.19377](https://pdf.sciencedirectassets.com/271625/1-s2.0-S0020025525X00256/1-s2.0-S0020025525007741/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEsaCXVzLWVhc3QtMSJGMEQCIHkZ42OpoCL2ZOrYMtcxrn2UFjt8xg%2Bd%2F07rWNks2hznAiA9uHiwow9eKfiaWuMcaZGNCBz%2FUmUJ%2BwD4h6NRnPL8FyqyBQgUEAUaDDA1OTAwMzU0Njg2NSIMhmdWYIDjdmQ1pAmhKo8FpDMmlols1hDy30j8RmqyARNuQNNIKORMRlB0nlA7s%2B0Qq9dt3x8B5JhcZWOwFDnpSV2gG2BI0hsZrF%2BLauGwreUpFQ5JXIF%2BxP2277sCSGoIs9QpJkeEVWet4UMm8GqDFkZ9WZdisaZ0BzCtfg1EruTAWiEMKRvR9GkFDSknb33ckC5M3%2F6ENjd6DMuyozPV%2Bu1Za4tzSQIIqGQL1YOUcJfFPZMMgmmklvupX5fpF8q32gVMeP7o7YT5iSt7UF6YbMRJcWu%2FmLGHRPghT1V3ZnnIEzWQDY2jNEf83kAcHGJU7mN1f2GOfI3pfweg84PsLr9VT3NfDrf5sliaC5oQ6WEGRjXe1zWbujZYA7lobkyNHLBv8Vhs8bLG3vvb2jWQQx15ujCKDbLnjXnC6Y7USJwdk4BN0NXSQZ1hA0HQJTBi4bndNn6QeB8xAKSWAj9Swx8%2F%2F7m9qMIR9Pi90epE2ABNkR%2FtF6LWnxgWwM4qQuVMFXbVE07JH9tRHT0kfmD0gZV81ehRaEIOOpscUDxrMurAtPhhXmdxy%2BRO2q49oH5cDu7RWMauSv%2ByyhAtemBCfHkL%2BD2FJlgSRgpUMsTd2mus1DHmUpVP%2FApsZTrSsK7ksNq29ojRFUe3uyZs3f3BfxXJaR8M8GKku%2BysCLDqH7ZgcYSljiYhrD1g6JGQ5BZyHGI6NkXaeTbA44WCuDrG3QXbdNlleK03pv4zKoPAqn43r1ZPF5NYdUfZ70hSVnZIZavKeUyCC8y4fFCLIBDMXOnxixapJ4f9XwH3H79Sl537foOsCdZWJSWiwHCYCE9UkGll%2BLLB421lSDevPKs3LeRAIxX3uYZj3CgWSaOtuLpbyoAHXaoiBfFhewW%2BOjCdhLvJBjqyAbHfwJegOnbfx34UIHHn24tncQBfdJpcrjA3whaxcWFm4LEJ8i9mACGdAmAb7nRoU19TQ1KUPep7O1VGLcRnepsvHyYW%2FVGjFXuq79VBy1AC%2FDVBBtY7UhSubPmCziq4aG0r48i5AB0mr%2BpLl0H3saaDOwzvxAsjCXytPxFW3cAAa48DcVpHfpI86fc0cL3Fvrq1G2LYJWIWPfKdoXF6GPd6hb2A2ltgUojvMD7VctEjATI%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251202T120918Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=ASIAQ3PHCVTYSPHEH5RE%2F20251202%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=5cdad57e3217b80eacb0a1035dcc0124bb5b0969a9d8ca42208fa6fc5899f860&hash=fdc8fc40d2a669480a6962ba0f9d44855a429195efdb5392b44836ea61c7aad1&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0020025525007741&tid=spdf-24baa7ee-ea2a-40b6-a1fd-d09885408d84&sid=b70308427ec897436629977639305e4592e8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1e035f05550202500257&rr=9a7ab6f23811236c&cc=de)](https://www.sciencedirect.com/science/article/pii/S0020025525007741))]

## **Contributions**
In addition to our contribution in benchmarking context, we design a multi-dimensional, interpretable performance evaluation framework consisting of two components, i.e., DOS and EBM, based on requirements in causal disocvery w.r.t. holistic evaluation and interpretability:  <br>
![Alt text](Results_Visualization/Images/DOS_Int.PNG)


## **Key Results**
The linear combinatorial approach R2-SortnRegress and the pre-trained nonlinear continuous optimization method AVICI achieve the best results in terms of the six-dimensional performance indicator DOS. The figure below from the results section of our study provides an overview of the performance of structure learning models in terms of DOS: <br>
<img src="https://github.com/gvelev123/Causal_Discovery_iid_Data/blob/main/Results_Visualization/Images/Results_Summary.PNG" alt="Alt-Text" width="700" height="800"><br>
Additionaly, the analysis of the EBM importance and interaction scores showed that the performance of the two best performing methods (R2-SortnRegress and AVICI) is more robust to variations in the experimental experimental factors included in our simulation framework than the performance of the remaining models.

## **Reproducibility**
1. Install requirements. ```pip install -r requirements.txt```

2. Simulation of data, application of the 14 causal discovery techniques included in our study and evaluation of inferred graphs: you can use the notebook ```Causal_Graph_Inference_and_Evaluation.ipynb``` in the folder  ```./Src/```. The notebook shows how to define the values for the experimental factors of the synthetic data generating process, and how to run and evaluate the models on a demo-version of the simulation (smaller proportion of the simulated datasets). Since the pickle-files containing the simulated datasets, the evaluations and the pandas dataframe containing all results exceed the allowed 25 MB size on github, we provide you with a gdrive link, which you can use in order to download all results files: https://drive.google.com/drive/folders/1UOlMKeqokCwFhF1AkyeMdAKLtE46JfXa?usp=drive_link. So that you are able to reproduce the evaluation of the results, and the visualizations (3. step below), you should save the downloaded files in the folder ```./Performance_Evaluation_Framework/Results/```.

3. Generation of visualizations included in the results section of the study:
you can use the notebook ```Large_Scale_Sensitivity_Analysis.ipynb``` in the folder  ```./Src/```. The class for generating the visualizations is in the folder  ```./Performance_Evaluation_Framework/Results/Results_Visualization/```.


## **Acknowledgements**

We appreciate the following github repositories very much for the valuable code base:

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
@article{VelevLessmann-2026-CausalDiscovery,
  title     = {Interpretable, multi-dimensional Evaluation Framework for Causal Discovery from observational i.i.d. Data},
  author    = Velev, Georg and
              Lessmann, Stefan},
  journal   = {Information Sciences},
  volume    = {723},
  year      = {2026}
}
```

