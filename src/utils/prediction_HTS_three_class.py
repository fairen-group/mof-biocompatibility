###############################################################
#
# Bringing it all together, this piece of code can predict the toxicity of a MOF in a high-throughput manner.
# The fragmentation needs to be done before this piece of code can be executed (see utils).  
# The input should be a CSV file containing the list of linkers.
# The output in return is a CSV file of predicted toxicity values.
# Note: Not benchmarked. Can only be executed once a trained model (.sav) is ready.
#
###############################################################

import pandas as pd
import pickle # load the model.

from rdkit import Chem # since feature selection was performed - we only need those features. So we featurize here directly.
from rdkit.Chem import Descriptors  

'''
The function below simply loads the CSV file with the list of linkers. 
Provide the filepath as the path to this CSV file.
'''

def load_data(filepath):
    mofs = pd.read_csv(filepath)
    return mofs


'''
Provide the filepath here.

'''
data_path = 'predictions_updated.csv'

'''
This is the path to the saved ML model (the optimized, best performing one).
'''

model_path = './models/ip_finalized_model_three_class.sav'

'''
The function below simply loads the model. In case you are changing the filepath, update the path above accordingly.
'''
def load_model(filepath):
    model = pickle.load(open(filepath, 'rb'))
    return model


'''
The list of linkers obviously need to be featurized. The function written below does the same.
The function is written again so that this piece of code can be run independently.
'''


def featurize(linker):
    try:
        m = Chem.MolFromSmiles(linker)
        EState_VSA1 = Descriptors.EState_VSA1(m) 
        EState_VSA2 = Descriptors.EState_VSA2(m)
        EState_VSA4 = Descriptors.EState_VSA4(m)
        EState_VSA5 = Descriptors.EState_VSA5(m)
        EState_VSA6 = Descriptors.EState_VSA6(m)
        EState_VSA7 = Descriptors.EState_VSA7(m)
        EState_VSA8 = Descriptors.EState_VSA8(m)
        EState_VSA9 = Descriptors.EState_VSA9(m)
        EState_VSA10 = Descriptors.EState_VSA10(m)
        VSA_EState1 = Descriptors.VSA_EState1(m)
        VSA_EState2 = Descriptors.VSA_EState2(m)
        VSA_EState3 = Descriptors.VSA_EState3(m)
        VSA_EState5 = Descriptors.VSA_EState5(m)
        VSA_EState6 = Descriptors.VSA_EState6(m)
        VSA_EState8 = Descriptors.VSA_EState8(m)
        VSA_EState9 = Descriptors.VSA_EState9(m)
        QED = Descriptors.qed(m)
        Chi0 = Descriptors.Chi0(m)
        Chi0n = Descriptors.Chi0n(m)
        Chi0v = Descriptors.Chi0v(m)
        Chi1 = Descriptors.Chi1(m)
        Chi1n = Descriptors.Chi1n(m)
        Chi1v = Descriptors.Chi1v(m)
        Chi2n = Descriptors.Chi2n(m)
        Chi3n = Descriptors.Chi3n(m)
        Chi3v = Descriptors.Chi3v(m)
        Chi4n = Descriptors.Chi4n(m)
        Chi4v = Descriptors.Chi4v(m)
        HallKierAlpha = Descriptors.HallKierAlpha(m)
        Kappa1 = Descriptors.Kappa1(m)
        Kappa2 = Descriptors.Kappa2(m)
        MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(m)
        FpDensityMorgan2 = Descriptors.FpDensityMorgan2(m)
        FpDensityMorgan3 = Descriptors.FpDensityMorgan3(m)
        FpDensityMorgan1 = Descriptors.FpDensityMorgan1(m)
        HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(m)
        ExactMolWt = Descriptors.ExactMolWt(m)
        MolWt = Descriptors.MolWt(m)
        NumValenceElectrons = Descriptors.NumValenceElectrons(m)
        MaxPartialCharge = Descriptors.MaxPartialCharge(m)
        MolMR = Descriptors.MolMR(m)
        LabuteASA = Descriptors.LabuteASA(m)
        PEOE_VSA1 = Descriptors.PEOE_VSA1(m)
        PEOE_VSA2 = Descriptors.PEOE_VSA2(m)
        PEOE_VSA3 = Descriptors.PEOE_VSA3(m)
        PEOE_VSA9 = Descriptors.PEOE_VSA9(m)
        PEOE_VSA10 = Descriptors.PEOE_VSA10(m)
        PEOE_VSA11 = Descriptors.PEOE_VSA11(m)
        PEOE_VSA13 = Descriptors.PEOE_VSA13(m)
        PEOE_VSA14 = Descriptors.PEOE_VSA14(m)
        SMR_VSA1 = Descriptors.SMR_VSA1(m)
        SMR_VSA2 = Descriptors.SMR_VSA2(m)
        SMR_VSA3 = Descriptors.SMR_VSA3(m)
        SMR_VSA4 = Descriptors.SMR_VSA4(m)
        SMR_VSA5 = Descriptors.SMR_VSA5(m)
        SMR_VSA6 = Descriptors.SMR_VSA6(m)
        SMR_VSA7 = Descriptors.SMR_VSA7(m)
        SMR_VSA10 = Descriptors.SMR_VSA10(m)
        SlogP_VSA2 = Descriptors.SlogP_VSA2(m)
        SlogP_VSA3 = Descriptors.SlogP_VSA3(m)
        SlogP_VSA4 = Descriptors.SlogP_VSA4(m)
        SlogP_VSA5 = Descriptors.SlogP_VSA5(m)
        SlogP_VSA6 = Descriptors.SlogP_VSA6(m)
        SlogP_VSA8 = Descriptors.SlogP_VSA8(m)
        TPSA = Descriptors.TPSA(m)
        MaxAbsEStateIndex = Descriptors.MaxAbsEStateIndex(m)
        MaxEStateIndex = Descriptors.MaxEStateIndex(m)
        MinEStateIndex = Descriptors.MinEStateIndex(m)
        fr_Al_OH = Descriptors.fr_Al_OH(m)
        fr_Al_OH_noTert = Descriptors.fr_Al_OH_noTert(m)
        fr_ArN = Descriptors.fr_ArN(m)
        fr_Ar_N = Descriptors.fr_Ar_N(m)
        fr_C_O = Descriptors.fr_C_O(m)
        fr_C_O_noCOO = Descriptors.fr_C_O_noCOO(m)
        fr_NH0 = Descriptors.fr_NH0(m)
        fr_Ndealkylation1 = Descriptors.fr_Ndealkylation1(m)
        fr_alkyl_halide = Descriptors.fr_alkyl_halide(m)
        fr_allylic_oxid = Descriptors.fr_allylic_oxid(m)
        fr_amide = Descriptors.fr_amide(m)
        fr_aniline = Descriptors.fr_aniline(m)
        fr_benzene = Descriptors.fr_benzene(m)
        fr_bicyclic = Descriptors.fr_bicyclic(m)
        fr_epoxide = Descriptors.fr_epoxide(m)
        fr_ester = Descriptors.fr_ester(m)
        fr_ether = Descriptors.fr_ether(m)
        fr_ketone = Descriptors.fr_ketone(m)
        fr_ketone_Topliss = Descriptors.fr_ketone_Topliss(m)
        fr_lactam = Descriptors.fr_lactam(m)
        fr_lactone = Descriptors.fr_lactone(m)
        fr_para_hydroxylation = Descriptors.fr_para_hydroxylation(m)
        fr_phos_acid = Descriptors.fr_phos_acid(m)	
        fr_phos_ester = Descriptors.fr_phos_ester(m)
        fr_piperzine = Descriptors.fr_piperzine(m)	
        fr_quatN = Descriptors.fr_quatN(m)
        FractionCSP3 = Descriptors.FractionCSP3(m)
        HeavyAtomCount = Descriptors.HeavyAtomCount(m)
        NHOHCount = Descriptors.NHOHCount(m)
        NOCount = Descriptors.NOCount(m)
        NumAliphaticCarbocycles = Descriptors.NumAliphaticCarbocycles(m)
        NumAliphaticHeterocycles = Descriptors.NumAliphaticHeterocycles(m)
        NumAliphaticRings = Descriptors.NumAliphaticRings(m)
        NumAromaticCarbocycles = Descriptors.NumAromaticCarbocycles(m)
        NumAromaticHeterocycles = Descriptors.NumAromaticHeterocycles(m)
        NumAromaticRings = Descriptors.NumAromaticRings(m) 
        NumHDonors = Descriptors.NumHDonors(m)
        NumHeteroatoms = Descriptors.NumHeteroatoms(m)
        NumRotatableBonds = Descriptors.NumRotatableBonds(m)
        NumSaturatedCarbocycles = Descriptors.NumSaturatedCarbocycles(m)	
        NumSaturatedHeterocycles = Descriptors.NumSaturatedHeterocycles(m)	
        NumSaturatedRings = Descriptors.NumSaturatedRings(m)
													
        features = {'EState_VSA1' : EState_VSA1,
                    'EState_VSA2' : EState_VSA2,
                    'EState_VSA4' : EState_VSA4,
                    'EState_VSA5' : EState_VSA5,
                    'EState_VSA6' : EState_VSA6,
                    'EState_VSA7' : EState_VSA7,
                    'EState_VSA8' : EState_VSA8,
                    'EState_VSA9' : EState_VSA9,
                    'EState_VSA10' : EState_VSA10,
                    'VSA_EState1' : VSA_EState1,
                    'VSA_EState2' : VSA_EState2,
                    'VSA_EState3' : VSA_EState3,
                    'VSA_EState5' : VSA_EState5,
                    'VSA_EState6' : VSA_EState6,
                    'VSA_EState8' : VSA_EState8,
                    'VSA_EState9' : VSA_EState9,
                    'QED' : QED,
                    'Chi0' : Chi0,
                    'Chi0n' : Chi0n,
                    'Chi0v' : Chi0v,
                    'Chi1' : Chi1,
                    'Chi1n' : Chi1n,
                    'Chi1v' : Chi1v,
                    'Chi2n' : Chi2n,
                    'Chi3n' : Chi3n,
                    'Chi3v' : Chi3v,
                    'Chi4n' : Chi4n,
                    'Chi4v' : Chi4v,
                    'HallKierAlpha' : HallKierAlpha,
                    'Kappa1' : Kappa1,
                    'Kappa2' : Kappa2,
                    'MinAbsPartialCharge' : MinAbsPartialCharge,
                    'FpDensityMorgan2' : FpDensityMorgan2,
                    'FpDensityMorgan3' : FpDensityMorgan3,
                    'FpDensityMorgan1' : FpDensityMorgan1,
                    'HeavyAtomMolWt' : HeavyAtomMolWt,
                    'ExactMolWt': ExactMolWt,
                    'MolWt' : MolWt,
                    'NumValenceElectrons' : NumValenceElectrons,
                    'MaxPartialCharge' : MaxPartialCharge,
                    'MolMR' : MolMR,
                    'LabuteASA' : LabuteASA,
                    'PEOE_VSA1' : PEOE_VSA1,
                    'PEOE_VSA2' : PEOE_VSA2,
                    'PEOE_VSA3' : PEOE_VSA3,
                    'PEOE_VSA9' : PEOE_VSA9,
                    'PEOE_VSA10' : PEOE_VSA10,
                    'PEOE_VSA11' : PEOE_VSA11,
                    'PEOE_VSA13' : PEOE_VSA13,
                    'PEOE_VSA14' : PEOE_VSA14,
                    'SMR_VSA1' : SMR_VSA1,
                    'SMR_VSA2' : SMR_VSA2,
                    'SMR_VSA3' : SMR_VSA3,
                    'SMR_VSA4' : SMR_VSA4,
                    'SMR_VSA5' : SMR_VSA5,
                    'SMR_VSA6' : SMR_VSA6,
                    'SMR_VSA7' : SMR_VSA7,
                    'SMR_VSA10' : SMR_VSA10,
                    'SlogP_VSA2' : SlogP_VSA2,
                    'SlogP_VSA3' : SlogP_VSA3,
                    'SlogP_VSA4' : SlogP_VSA4,
                    'SlogP_VSA5' : SlogP_VSA5,
                    'SlogP_VSA6' : SlogP_VSA6,
                    'SlogP_VSA8' : SlogP_VSA8, 
                    'TPSA' : TPSA,
                    'MaxAbsEStateIndex' : MaxAbsEStateIndex,
                    'MaxEStateIndex' : MaxEStateIndex,
                    'MinEStateIndex' : MinEStateIndex,
                    'fr_Al_OH' : fr_Al_OH,
                    'fr_Al_OH_noTert' : fr_Al_OH_noTert,
                    'fr_ArN' : fr_ArN,
                    'fr_Ar_N' : fr_Ar_N,
                    'fr_C_O' : fr_C_O,
                    'fr_C_O_noCOO' : fr_C_O_noCOO,
                    'fr_NH0' : fr_NH0,
                    'fr_Ndealkylation1' : fr_Ndealkylation1,
                    'fr_alkyl_halide' : fr_alkyl_halide,
                    'fr_allylic_oxid' : fr_allylic_oxid,
                    'fr_amide' : fr_amide,
                    'fr_aniline' : fr_aniline,
                    'fr_benzene' : fr_benzene,
                    'fr_bicyclic' : fr_bicyclic,
                    'fr_epoxide' : fr_epoxide,
                    'fr_ester' : fr_ester,
                    'fr_ether' : fr_ether,
                    'fr_ketone' : fr_ketone,
                    'fr_ketone_Topliss' : fr_ketone_Topliss,
                    'fr_lactam' : fr_lactam, 
                    'fr_lactone' : fr_lactone,# fr_lactone
                    'fr_para_hydroxylation' : fr_para_hydroxylation,
                    'fr_phos_acid' : fr_phos_acid,
                    'fr_phos_ester' : fr_phos_ester,
                    'fr_piperzine' : fr_piperzine,
                    'fr_quatN' : fr_quatN,
                    'FractionCSP3' : FractionCSP3,
                    'HeavyAtomCount' : HeavyAtomCount,
                    'NHOHCount' : NHOHCount,
                    'NOCount' : NOCount,
                    'NumAliphaticCarbocycles' : NumAliphaticCarbocycles,
                    'NumAliphaticHeterocycles' : NumAliphaticHeterocycles,
                    'NumAliphaticRings' : NumAliphaticRings,
                    'NumAromaticCarbocycles' : NumAromaticCarbocycles,
                    'NumAromaticHeterocycles' : NumAromaticHeterocycles,
                    'NumAromaticRings' : NumAromaticRings,
                    'NumHDonors' : NumHDonors,
                    'NumHeteroatoms' : NumHeteroatoms,
                    'NumRotatableBonds' : NumRotatableBonds,
                    'NumSaturatedCarbocycles' : NumSaturatedCarbocycles,
                    'NumSaturatedHeterocycles' : NumSaturatedHeterocycles,
                    'NumSaturatedRings' : NumSaturatedRings}
                
    except:
        print('Error with featurization') # happens sometimes for very complex molecules.
    
    return features

'''
Bringing it all together, we will now perform the high-throughput screen.
'''
def predict():
    
    '''
    The output will be a dataframe with the following columns.
    Name: CSD Refcode for the MOF.
    Metal: The metal center (from moffragmentor).
    Linker: The linker (from moffragmentor).
    Toxicity: The prediction of the model (-1, 0 or 1).
    Category: The corresponding category of toxicity.
    '''

    name = []
    metal = []
    linker = []
    toxicity = []
    category = []
    features = []

    data = load_data(data_path)
    model = load_model(model_path)
    linkers = data['Linker']
    success_pred = 0

    for i in range(len(linkers)):
        print('Predicting toxicity for '+str(linkers[i]).strip())
        '''
        In some cases, the linkers are fragmented in such a way that the featurization is not possible. To catch this, we will run a try-except block.
        '''

        try: 
            input = featurize(str(linkers[i]))
            X = pd.DataFrame([input], columns=input.keys())
            y_pred = model.predict(X)
            toxicity.append(y_pred)
            if y_pred == -1:
                print('Linker '+str(linkers[i])+' is fatal')
                category.append('Fatal')
            elif y_pred == 0:
                print('Linker '+str(linkers[i])+' is toxic')
                category.append('Toxic')
            else:
                print('Linker '+str(linkers[i])+' is safe')
                category.append('Safe')
            success_pred += 1

            name.append(str(data['Name'][i]).strip())
            metal.append(str(data['Metal Centre'][i]).strip())
            linker.append(data['Linker'][i])
            features.append(input.values())
            feature_name = input.keys()

        except:
            print('Could not functionalise linker')
            success_pred += 0
    '''
    Let us store the features as a dataframe.
    '''
         
    predictions = pd.DataFrame(list(zip(name, metal, linker, toxicity, category)), columns=['Name', 'Metal Centre', 'Linker', 'Toxicity', 'Category'])
    predictions_path = './data/predictions_updated.csv'
    predictions.to_csv(predictions_path)

    features_gen = pd.DataFrame(features, columns = feature_name)
    features_gen['Name'] = name
    features_gen['Metal'] = metal
    features_gen['Linker'] = linker
    features_gen['Toxicity'] = toxicity
    features_gen['Category'] = category

    path_features = './predictions_updated_features.csv'
    features_gen.to_csv(path_features)

    '''
    Below add the path to the CSV file where the results are to be saved.
    '''
    return predictions

print('Starting the screen.')
predictions = predict() 
print('Finished the high-throughput screen.')
