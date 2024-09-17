This is the official implementation for our upcoming paper (under revision):
### Guiding the rational design of biocompatible metal-organic frameworks for drug delivery.
[Dhruv Menon](https://scholar.google.com/citations?user=NMOjZLQAAAAJ&hl=en&oi=ao)\,
[David Fairen-Jimenez](https://scholar.google.com/citations?user=F3UKbZsAAAAJ&hl=en&oi=ao)\*

For correspondence please contact: dm958[at]cam[dot]ac[dot]uk or df334[at]cam[dot]ac[dot]uk.

The python code in this repository is capable of executing the pipeline as illustrated below.  
<p align="center">
  <img src="img/Schematic 1.png" style="padding:10px;" width="700"/>
</p>  

## Code Walkthrough
Here, we provide a walk-through of the code. It would be best to execute these in a dedicated environment. To avoid dependency hell, I suggest installing chemprop which has all the required packages needed here, with the exception of Shap and imbalanced learn.    
```
conda create -y -n biomof
conda activate biomof
pip install chemprop
pip install shap
pip install imbalanced-learn
```
The code as such is well-documented and largely self-explanatory. A suggested workflow is as follows:
1. Featurise the raw data. The data provided has been cleaned beforehand for duplicate entries and null values.
2. Following featurisation, we will carry out feature selection.
3. Next, we split the data into a train and independent test set. We do the splitting before any sampling to ensure that there is no information leakage. This allows us to gauge the true performance of the model on real-world data - which may often be heavily imbalanced.
4. We perform data sampling on the train set to obtain equal datapoints for each class. Classifiers trained on imbalanced data, may be biased.
5. We perform a gridsearch of hyperparameters to get a rough estimate of hyperparameters that are able to learn our features well. This has been implemented for all three models - Random Forest (RF), Gradient Boosting Machine (GBM) and Support Vector Machine (SVM). Following this, the best performing model(s) are further optimised using a finer grid search. The grid search results will be saved as a .csv file.
6. Once we have a good set of hyperparameters, we train the production model. This will save (i) the model weights and (ii) ROC-AUC scores and plots. 

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
The code provided is readily implementable - the only change required in most cases is putting in the correct path to the directory where the data is stored. **Note:** *problems may arise due to conflicts between dependencies. Chemprop works with any issues.* 

Alternatively, if you have less expertise in code, and simply want to use the model to quickly identify the potential toxicity of linkers you are interested in - you can download and use the Jupyter Notebook provided. The ZIP file also contains the dataset on which the best performing model was trained on. **Note:** *This is for the i.p. route of administration.*

## Data
In the directory *training_data*, we provide CSV files of clean, raw data for toxicity profiles through the oral and i.p. routes of administration respectively. While the original dataset has reported values of LD50, we have categorized these values (please, see our discussion in the manuscript). If you are using this data, please cite the original source as follows: Wu, L., Yan, B., Han, J., Li, R., Xiao, J., He, S. and Bo, X., 2023. TOXRIC: a comprehensive database of toxicological data and benchmarks. Nucleic Acids Research, 51(D1), pp.D1432-D1445. [https://doi.org/10.1093/nar/gkac1074]

In the directory *metal_data*, we provide our curated data for the reported toxicity profiles of metal centres.

In the directory *safe-MOFs*, we provide CSD refcodes and pore properties of MOFs which were reported to be safe and 'less' toxic respectively.

## Citing
If you use the tools developed in this study - please consider citing our work. **This will be updated as soon as the paper is published (underway)**

Menon, D. and Fairen-Jimenez, D., 2024. Guiding the rational design of biocompatible MOFs for drug delivery. *Under revision*

## Contributing
Contributions, whether filing an issue, making a pull request, or forking, are appreciated. Please note that the repository may not be regularly maintained.

## License
This code package is licensed under the MIT License. 

## Development and Funding
This work was carried out at the Adsorption and Advanced Materials (A2ML) Laboratory. Supported by the Engineering and Physical Sciences Research Council (EPSRC) and the Trinity Henry-Barlow (Honorary) Scholarship. 


