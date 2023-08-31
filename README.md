# HyAI

### Introduction
This repository consists of source files for the implementation of a hybrid AI (HyAI) framework. We have not publicly available the HyAI Knowledge Graph (KG) due to sensitive information about chronic hepatitis B virus (HBV) infected patients. HyAI is conceptualized following the design principles described by Bekkhum, V, et al. [**Design Patterns**](https://link.springer.com/article/10.1007/s10489-021-02394-3); following basic vocabulary for representing the components actor, input and output, process, and models (as depicted in the below Figure). HyAI framework consists of four design patterns: (i) Ontology and KG, (ii) KG Embedding, (iii) Pattern Detection, and (iv) Pattern Analysis and Explanation.


![C-KAP2023-Page-3](https://github.com/SDM-TIB/HyAI/assets/25593410/42cf771d-b82d-4097-b0f7-0ceb3581b171)


### Understanding HBV Patients (use case) using HyAI
We used HyAI in the use case of uncovering parameters of clinical, demographic, and immune phenotyping data that characterize chronic HBV patients with functional cure. HyAI has been implemented using state-of-the-art tools and techniques (as depicted in the below Figure). The heterogeneous datasets consists of 87 chronic HBV patients, including age, sex, 18 clinical observational parameters, 45 immune phenotyping parameters, and HBV treatment. The Ontology and KG system received a data integration system (DIS), that is composed of a unified schema (classes and properties), data sources, and RML mapping assertions. The KG embedding models (TransE, TransH, RESCAL, ERMLP) have been used to transform holistic profiles of 87 chronic HBV patients into low-dimensional vector representations. The Pattern Detection system used community detection algorithms (KMean, SemEP, METIS) to identify groups of HBV patients who shares similar features.


![CKAP2023_figure_pipeline_edit](https://github.com/SDM-TIB/HyAI/assets/25593410/88d680c9-f4cc-41a1-9726-deee61879739)

### Experiment Results

### Requirements
- GNU Compiler Collection (GCC) or Clang
- GNU make
- pykeen
- pandas
-	numpy
-	scipy
-	seaborn
-	sklearn
-	yellowbrick

Citing:
If you find HyAI helpful in your work please cite the papers:
```
Shahi Dost, Ariam Rivas, Hanan Begali, Annett Ziegler, Elmira Aliabadi, Markus Cornberg, Anke RM Kraft  and Maria-Esther Vidal. 2023.
Unraveling the Hepatitis B Cure: A Hybrid AI Approach for Capturing Knowledge about the Immune System’s Impact (KCAP’2023)[https://www.k-cap.org/2023/]
```

### License
HyAI codes and source files are licensed under GNU General Public [License v3.0](https://github.com/SDM-TIB/HyAI/blob/main/LICENSE).

### Authors
