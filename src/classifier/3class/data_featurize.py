###############################################################
#
# This script is for featurization of the dataset. Pass the raw dataset here and this script will generate the featurized dataset.
# There will be some internal checks and cleaning - sometimes the features are not generated correctly, are NaN values - these will be removed.
# The benchmark timings (on CPU) are as follows:
# ip_clean_dataset = 877 seconds (ca. 14.5 mins)
# oral_clean_dataset = 505 seconds (ca. 8.5 mins)
#
###############################################################


import pandas as pd
import generate_features # for calculating descriptors.
import time
import numpy as np

def feature_generate(smiles_list):
    features = pd.DataFrame(columns=['EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 
        'EState_VSA7', 'EState_VSA8', 'EState_VSA9','EState_VSA10', 'EState_VSA11', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3',
        'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'VSA_EState10', 'QED',
        'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 
        'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'MinAbsPartialCharge', 'NumRadicalElectrons', 'FpDensityMorgan2',
        'FpDensityMorgan3', 'FpDensityMorgan1', 'HeavyAtomMolWt', 'MaxAbsPartialCharge', 'MinPartialCharge', 'ExactMolWt',
        'MolWt', 'NumValenceElectrons', 'MaxPartialCharge', 'MolLogP', 'MolMR', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA2', 
        'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 
        'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 
        'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 
        'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'TPSA',
        'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinEStateIndex', 'MinAbsEStateIndex', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert',
        'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S',
        'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole',
        'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
        'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
        'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine',
        'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam',
        'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
        'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid',
        'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
        'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_unbrch_alkane',
        'fr_urea', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 
        'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHDonors', 'NumHeteroatoms',
        'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount'])
    
    for smiles in smiles_list:
        EState = generate_features.EState_Desc(smiles)
        QED = generate_features.QED(smiles)
        GeoDesc = generate_features.GeoDesc(smiles)
        GenDesc = generate_features.GenDesc(smiles)
        Crippen = generate_features.Crippen(smiles)
        MolSurf = generate_features.MolSurf(smiles)
        EBasic = generate_features.EState_Basic(smiles)
        FragDesc = generate_features.Frag_Desc(smiles)
        Lipinski = generate_features.LipinskiDesc(smiles)

        features_combine = EState + QED + GeoDesc + GenDesc + Crippen + MolSurf + EBasic + FragDesc + Lipinski
        features.loc[len(features)] = features_combine
        
    return features


def data_processing(raw_data = None):
    '''
    Standard data pre-processing.
    Input raw data - with list of SMILES and corresponding toxicity.
    Data has been cleaned before hand - but after featurisation there may be some missing values. So for sanity, we clean again post-featurisation. 
    '''

    smiles = raw_data['Canonical SMILES'] # Check that the column name is correct.
    toxicity = raw_data['Category'] # Check that the column name is correct.

    '''
    First, we featurise the SMILES passed. 
    '''
    start = time.time()
    print('Beginning featurisation')
    features = feature_generate(smiles)
    print('Completed featurisation')
    end = time.time()
    duration = end - start
    print('Featurisation took %.2f seconds' % duration)
    
    features_smiles = features.assign(SMILES = smiles) # adding the SMILES column.
    dataset = features_smiles.assign(Category = toxicity) # adding the label.

    '''
    Might need to remove NaNs - remove all rows that contain NaNs in any column. This could arise during featurization.
    First, replace infinte values with NaNs, and then remove NaNs.
    '''
    rows_init = len(dataset.index)
    print('Rows before final cleanup: ' + str(rows_init))
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True) 
    dataset_clean = dataset.dropna(axis = 0, how = 'any') 
    rows_final = len(dataset_clean.index) 
    print('Rows after final cleanup: ' + str(rows_final))

    print('Dataset generated')
    return dataset_clean

data = pd.read_csv('./data/oral_data_cleaned.csv') # change the name of the CSV file accordingly.
dataset = data_processing(data)
dataset.to_csv('./data/oral_data_featurised.csv') # change the name of the CSV file accordingly. 