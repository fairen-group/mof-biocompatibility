<p align="center">
  <img src="utils/logo.png" style="padding:10px;" width="700"/>
</p>  

# mof-biocompatibility
### Guiding the rational design of biocompatible metal-organic frameworks for drug delivery using machine learning.
The python code in this repository is capable of executing the pipeline as illustrated below.  
<p align="center">
  <img src="utils/Schematic 1.png" style="padding:10px;" width="700"/>
</p>  

Here, we provide an overview of the code. 
* data_featurize.py: Featurization of the data to produce 197 descriptors capturing the information of the molecule at various lengthscales.
  You need to have 'rdkit' installed which you can install with 'pip install rdkit' or 'conda install -c conda-forge rdkit'.
* data_sampling.py: Sampling of the data to ensure balanced classes. Majority class is undersampled (random sampling) and the minority class is oversampled (ADASYN algorithm).
  You need to have the imbalanced-learn package installed, which you can install with 'pip install imblearn'.
* feature_selection.py: Selecting the KBest features (outlined in the manuscript) as implemented using scikit-learn, which can be installed using 'pip install scikit-learn'.
* gbt_coarsegrid.py: The Gradient Boosting Machine (GBM) trained on a coarse-grid of hyperparameters as outlined in the manuscript and the schematic above. The model has been implemented using scikit-learn.
* generate_features.py: This code is used to pass a dataset of molecules through data_featurize to generate a CSV file of featurized data. 

## 💪 Getting Started
