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
  smiles_list = [] # add here the SMILES string from the data
  features = feature_generate(smiles_list) # returns the feature dataframe that can be used for model training
  ```
* generate_features.py: This code is used to pass a dataset of molecules through data_featurize to generate a CSV file of featurized data.  
* data_sampling.py: Sampling of the data to ensure balanced classes. Majority class is undersampled (random sampling) and the minority class is oversampled (ADASYN algorithm).
  You need to have the imbalanced-learn package installed, which you can install with:
  ```
  pip install imbalanced-learn
  ```
  Here operating inside an environment helps, in order to avoid the 'dependency hell'. However, just in case, you may check the dependencies here: [https://imbalanced-learn.org/stable/install.html]
  For undersampling, we recommend using random undersampling. Here, be sure to define the sampling strategy as follows:
  ```python
  strategy = {-1: 0, 0: 5000, 1: 0}
  '''
  0 being the majority class is being undersampled to 5000 points, while -1 and 1 being the minority classes are not undersampled.
  '''
  ```
  For undersampling/oversampling, simply call the respective functions:
  ```python
  X_sampled, y_sampled = undersample_random(X, y) # for undersampling
  X_sampled, y_sampled = oversample_ADASYN(X, y) # for oversampling
  ```
* feature_selection.py: Selecting the KBest features (outlined in the manuscript) as implemented using scikit-learn, which can be installed using:
  ```
  pip install scikit-learn
  ```
  If you are directly running this script, please provide the path to the featurized data and the number of features to be selected as:
  ```python
  path = 'data.csv' # add your path here
  K = 110 # number of features to be selected (best performance reported for 110 features using the i.p. data - please refer to manuscript)
  ```
  Run as follows (will save the data as a csv file):
  ```python
  _ = run(path, K)
  ```
* gbt_coarsegrid.py: The Gradient Boosting Machine (GBM) trained on a coarse-grid of hyperparameters as outlined in the manuscript and the schematic above. The model has been implemented using scikit-learn.
  Now, depending on the computational resources at your disposal, you may consider making the parameter space larger or smaller.
  ```python
  param_grid = {'learning_rate : [0.1, 0.01, 0.001],
                'max_iter' : [100, 200, 300, 400, 500],
                'max_leaf_nodes' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                'min_samples_leaf' : [20, 30, 40, 50, 60, 70, 80, 90, 100],
                'l2_regularization' : [0, 0.5, 0.75, 1, 10]}
  ```
  Depending on the feature-set, with the parameter space outlined above, this could take anywhere between a couple of hours, to over 12 hours to execute. To run the code, simply execute:
  ```python
  path = 'data.csv' # path to the data
  scores = run_model(path) # returns the model performance
  ```
* rfc_coarsegrid.py: The Random Forest (RF) trained on a coarse-grid of hyperparameters as outlined in the manuscript and the schematic above. The model has been implemented using scikit-learn.
  The parameter space we have defined is as follows:
  ```python
  param_grid = {'n_estimators' : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                'min_samples_split' : [2, 5, 10]}
  ```
  You can expect similar timescales - based on the size of the dataset you are training the models on.
* svc_coarsegrid.py: The Support Vector Machine (SVM) trained on a coarse-grid of hyperparameters as outlined in the manuscript and the schematic above. The model has been implemented using scikit-learn.
  The parameter space we have defined is as follows:
  ```python
  param_grid = {'C' : [0.1, 1, 5, 10],
                'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
                'decision_function_shape' : ['ovo', 'ovr']}
  ```
* rfc_finegrid.py: The RF trained on a fine grid of hyperparameters as outlined in the manuscript and the schematic above. The model has been implemented using scikit-learn.
* rfc.py: Training the best performing model - the RF using the optimum hyperparameters as deduced from the coarsegrid and finegrid hyperparameter optimization processes. The best performing model is saved after this code is executed.
  To run the model:
  ```python
  path = 'data.csv' # path to where the data is stored
  _, _, model, _, _, _, _, _, _, _, _ = run_model(path) 
  ```
  This will save the trained model for future use as finalized_model.sav (~200 Mb)
* prediction_HTS.py: Predict the toxicity of MOF linker molecules in a high-throughput manner.
  You would need to have a saved model to execute this code. To run, you need a csv file with the list of linkers to be tested.
  Add the path to where the list is saved as:
  ```python
  data_path = 'data.csv'
  ```
  Add the path to where the model is saved as:
  ```python
  model_path = 'finalized_model.sav'
  ```
In order to fragment MOF structures into their building MOFs, we recommend using *moffragmentor*, developed by Jablonka et al., accessible here: [https://github.com/kjappelbaum/moffragmentor/tree/main]. The library can be installed using:
```
pip install moffragmentor
```
Please note: *moffragmentor* is dependent on openbabel, which may be installed on your anaconda environment using:
```
conda install openbabel -c conda-forge
```
From personal experience, openbabel is a bit tricky to set up, so please be careful. While we provide some useful helper scripts for using the library (check utils) - we would refer you to the source repository for a more comprehensive discussion. If you are using this code, please cite: Jablonka, K.M., Rosen, A.S., Krishnapriyan, A.S. and Smit, B., 2023. An ecosystem for digital reticular chemistry. [https://doi.org/10.1021/acscentsci.2c01177]

Alternatively, you may use *Poremake* for MOF deconstruction, developed by Lee et al., accessible here: [https://github.com/Sangwon91/PORMAKE]. The library can be installed using:
```
pip install poremake
```
If you are using this code, please cite: Lee, S., Kim, B., Cho, H., Lee, H., Lee, S.Y., Cho, E.S. and Kim, J., 2021. Computational screening of trillions of metal–organic frameworks for high-performance methane storage. ACS Applied Materials & Interfaces, 13(20), pp.23647-23654. [https://doi.org/10.1021/acsami.1c02471]

In case you want to clean structures, you can remove bound and unbound solvents using scripts developed previously by us. These scripts are deposited in the utils directory. If you are using this code, please cite: Moghadam, P.Z., Li, A., Wiggin, S.B., Tao, A., Maloney, A.G., Wood, P.A., Ward, S.C. and Fairen-Jimenez, D., 2017. Development of a Cambridge Structural Database subset: a collection of metal–organic frameworks for past, present, and future. Chemistry of Materials, 29(7), pp.2618-2625. [https://doi.org/10.1021/acs.chemmater.7b00441]

In case you want to re-parse some CIF files to correct occupancy issues, you may use the script mofparser.py. Simply, provide the directory with the structures at:
```python
directory = 'directory_path/' # add the path to your directory
```
## Getting Started
The code provided is readily implementable - the only change required in most cases is putting in the correct path to the directory where the data is stored. **Note:** *problems may arise due to conflicts between dependencies.* 

Alternatively, if you have less expertise in code, and simply want to use the model to quickly identify the potential toxicity of linkers you are interested in - you can download and use the Jupyter Notebook provided. The ZIP file also contains the dataset on which the best performing model was trained on. **Note:** *This is for the i.p. route of administration.*

## Data
In the directory *training_data*, we provide CSV files of clean, raw data for toxicity profiles through the oral and i.p. routes of administration respectively. While the original dataset has reported values of LD50, we have categorized these values (please, see our discussion in the manuscript). If you are using this data, please cite the original source as follows: Wu, L., Yan, B., Han, J., Li, R., Xiao, J., He, S. and Bo, X., 2023. TOXRIC: a comprehensive database of toxicological data and benchmarks. Nucleic Acids Research, 51(D1), pp.D1432-D1445. [https://doi.org/10.1093/nar/gkac1074]

In the directory *metal_data*, we provide our curated data for the reported toxicity profiles of metal centres.

In the directory *safe-MOFs*, we provide CSD refcodes and pore properties of MOFs which were reported to be safe.
## Contributing
Contributions, whether filing an issue, making a pull request, or forking, are appreciated. 

## License
This code package is licensed under the MIT License. 

## Development and Funding
This work was carried out at the Adsorption and Advanced Materials (A2ML) Laboratory. Supported by the Engineering and Physical Sciences Research Council (EPSRC) and the Trinity Henry-Barlow (Honorary) Scholarship. 
<p align="center">
  <img src="utils/a2ml_logo.png" style="padding:10px;" width="700"/>
</p>  

