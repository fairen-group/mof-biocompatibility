# bioMOFx: A High-Throughput Approach to Screen Large MOF Libraries for Biocompatibility.  

This is the official implementation for our upcoming paper (under revision):
### Guiding the rational design of biocompatible metal-organic frameworks for drug delivery using machine learning.
[Dhruv Menon](https://scholar.google.com/citations?user=NMOjZLQAAAAJ&hl=en&oi=ao)\,
[David Fairen-Jimenez](https://scholar.google.com/citations?user=F3UKbZsAAAAJ&hl=en&oi=ao)\*

For correspondence please contact: dm958[at]cam[dot]ac[dot]uk or df334[at]cam[dot]ac[dot]uk.

The python code in this repository is capable of executing the pipeline as illustrated below.  
<p align="center">
  <img src="utils/Schematic 1.png" style="padding:10px;" width="700"/>
</p>  

Here, we provide a walk-through of the code. It would be best to execute these in a dedicated environment.   
```
conda create -y -n biomofx python=3.10
conda activate biomofx
```
* data_featurize.py: Featurization of the data to produce 197 descriptors capturing the information of the molecule at various lengthscales.
  You need to have 'rdkit' installed which you can install with:
  ```
  pip install rdkit
  ```
  or
  ```
  conda install -c conda-forge rdkit
  ```
  For a list of SMILES, generate features by adding the following lines of code to the script:
  ```python
  smiles_list = [] # add here the SMILES string
  features = feature_gen(smiles_list)
  ```
* data_sampling.py: Sampling of the data to ensure balanced classes. Majority class is undersampled (random sampling) and the minority class is oversampled (ADASYN algorithm).
  You need to have the imbalanced-learn package installed, which you can install with 'pip install imblearn'.
* feature_selection.py: Selecting the KBest features (outlined in the manuscript) as implemented using scikit-learn, which can be installed using 'pip install scikit-learn'.
* gbt_coarsegrid.py: The Gradient Boosting Machine (GBM) trained on a coarse-grid of hyperparameters as outlined in the manuscript and the schematic above. The model has been implemented using scikit-learn.
* generate_features.py: This code is used to pass a dataset of molecules through data_featurize to generate a CSV file of featurized data.
* prediction_HTS.py: Predict the toxicity of MOF linker molecules in a high-throughput manner.
* rfc.py: Training the best performing model - the random forest (RF) using the optimum hyperparameters as deduced from the coarsegrid and finegrid hyperparameter optimization processes. The best performing model is saved after this code is executed.
* rfc_coarsegrid.py: The RF trained on a coarse-grid of hyperparameters as outlined in the manuscript and the schematic above. The model has been implemented using scikit-learn.
* rfc_finegrid.py: The RF trained on a fine grid of hyperparameters as outlined in the manuscript and the schematic above. The model has been implemented using scikit-learn.
* svc_coarsegrid.py: The Support Vector Machine (SVM) trained on a coarse-grid of hyperparameters as outlined in the manuscript and the schematic above. The model has been implemented using scikit-learn.

The code for the fragmentation of a MOF into its building blocks (moffragmentor) has been developed by Jablonka et al. If you are using this code, please cite: Jablonka, K.M., Rosen, A.S., Krishnapriyan, A.S. and Smit, B., 2023. An ecosystem for digital reticular chemistry.

Here we provide an overview of the part of the code we developed to leverage moffragmentor in a high-throughput manner.
* mof_parser.py: The crystallographic information file (CIF) of the MOFs as extracted from the CSD need to be re-prased due to some occupancy issues. This code is capable of doing so in a high-throughput manner.
* fragmentor.py: Carries out MOF fragmentation in a high-throughput manner. Ideally saves the CSV file with the name of the MOF (CSD refcode), metallic node and linker. However, in case of an issue, it also prints a text file with all the details.

In addition to this, we provide a list of CIF files of Zr-centered MOFs (as a test case) which have been parsed, and the list of these MOFs that have been identified to be safe. 

## Getting Started
The code provided is readily implementable - the only change required is putting in the correct path to the directory where the data is stored. If you want to train the models yourself, follow these steps:
1. Featurize the data: For this, use generate_features.py.
2. Sample the data: For this, use data_sampling.py
3. Select the 'K'-best features: For this, use feature_selection.py
4. Model training (coarse-grid): Train the model on the features selected on a coarse-grid of hyperparameters. While you can use rf_coarsegrid.py, gbt_coarsegrid.py or svc_coarsegrid.py, we recommend you use either the rf_coarsegrid.py or gbt_coarsegrid.py.
5. Model training (fine-grid): Train the model on a finer grid (based on the outcome of the coarse-grid optimization). In case you are using a GBM, make appropriate changes to rfc_finegrid.py, else leave it as is. This should give you the hyperparameters with the best performance.
6. Final Model training and saving: For this, use rfc.py - with the hyperparameters as outputted in the fine-grid optimization. This code will save a model which can be used for future predictions.
7. MOF fragmentation: First, place all the CIF files in a directory and run mof_parser.py. This will save parsed CIF files into a new directory. Then run fragmentor.py. This will save a CSV file with the list of nodes and linkers.
8. High-throughput screening: For this, use prediction_HTS.py.

Alternatively, if you have less expertise in code, and simply want to use the model to quickly identify the potential toxicity of linkers you are interested in - you can download and use the Jupyter Notebook provided. The ZIP file also contains the dataset on which the best performing model was trained on. 

## Contributing
Contributions, whether filing an issue, making a pull request, or forking, are appreciated. 

## License
This code package is licensed under the MIT License. 

## Development and Funding
This work was carried out at the Adsorption and Advanced Materials (A2ML) Laboratory. Supported by the Engineering and Physical Sciences Research Council (EPSRC) and the Trinity Henry-Barlow (Honorary) Scholarship. 
<p align="center">
  <img src="utils/a2ml_logo.png" style="padding:10px;" width="700"/>
</p>  

