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
data_path = './data/predictions_toxic.csv'

'''
This is the path to the saved ML model (the optimized, best performing one).
'''

model_path = './models/ip_finalized_model_two_class.sav'

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
        EState_VSA3 = Descriptors.EState_VSA3(m)
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
        VSA_EState7 = Descriptors.VSA_EState7(m)
        VSA_EState8 = Descriptors.VSA_EState8(m)
        VSA_EState9 = Descriptors.VSA_EState9(m)
        VSA_EState10 = Descriptors.VSA_EState10(m)
        QED = Descriptors.qed(m)
        BertzCT = Descriptors.BertzCT(m)
        Chi0 = Descriptors.Chi0(m)
        Chi1 = Descriptors.Chi1(m)
        Chi3n = Descriptors.Chi3n(m)
        Chi4n = Descriptors.Chi4n(m)
        Chi4v = Descriptors.Chi4v(m)
        HallKierAlpha = Descriptors.HallKierAlpha(m)
        MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(m)
        HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(m)
        MaxAbsPartialCharge = Descriptors.MaxAbsPartialCharge(m)
        MinPartialCharge = Descriptors.MinPartialCharge(m)
        ExactMolWt = Descriptors.ExactMolWt(m)
        MolWt = Descriptors.MolWt(m)
        MaxPartialCharge = Descriptors.MaxPartialCharge(m)
        LabuteASA = Descriptors.LabuteASA(m)
        PEOE_VSA2 = Descriptors.PEOE_VSA2(m)
        PEOE_VSA3 = Descriptors.PEOE_VSA3(m)
        PEOE_VSA6 = Descriptors.PEOE_VSA3(m)
        PEOE_VSA7 = Descriptors.PEOE_VSA3(m)
        PEOE_VSA9 = Descriptors.PEOE_VSA9(m)
        PEOE_VSA10 = Descriptors.PEOE_VSA10(m)
        PEOE_VSA12 = Descriptors.PEOE_VSA12(m)
        PEOE_VSA13 = Descriptors.PEOE_VSA13(m)
        PEOE_VSA14 = Descriptors.PEOE_VSA14(m)
        SMR_VSA1 = Descriptors.SMR_VSA1(m)
        SMR_VSA2 = Descriptors.SMR_VSA2(m)
        SMR_VSA3 = Descriptors.SMR_VSA3(m)
        SMR_VSA5 = Descriptors.SMR_VSA5(m)
        SMR_VSA6 = Descriptors.SMR_VSA6(m)
        SMR_VSA7 = Descriptors.SMR_VSA7(m)
        SMR_VSA9 = Descriptors.SMR_VSA9(m)
        SMR_VSA10 = Descriptors.SMR_VSA10(m)
        SlogP_VSA1 = Descriptors.SlogP_VSA1(m)
        SlogP_VSA2 = Descriptors.SlogP_VSA2(m)
        SlogP_VSA3 = Descriptors.SlogP_VSA3(m)
        SlogP_VSA5 = Descriptors.SlogP_VSA5(m)
        SlogP_VSA6 = Descriptors.SlogP_VSA6(m)
        SlogP_VSA8 = Descriptors.SlogP_VSA8(m)
        SlogP_VSA10 = Descriptors.SlogP_VSA10(m)
        SlogP_VSA12 = Descriptors.SlogP_VSA12(m)
        TPSA = Descriptors.TPSA(m)
        MaxAbsEStateIndex = Descriptors.MaxAbsEStateIndex(m)
        MaxEStateIndex = Descriptors.MaxEStateIndex(m)
        MinEStateIndex = Descriptors.MinEStateIndex(m)
        MinAbsEStateIndex = Descriptors.MinAbsEStateIndex(m)
        fr_Al_COO = Descriptors.fr_Al_COO(m)
        fr_ArN = Descriptors.fr_ArN(m)
        fr_Ar_N = Descriptors.fr_Ar_N(m)
        fr_Ar_NH = Descriptors.fr_Ar_NH(m)
        fr_COO = Descriptors.fr_COO(m)
        fr_COO2 = Descriptors.fr_COO2(m)
        fr_C_O = Descriptors.fr_C_O(m)
        fr_C_O_noCOO = Descriptors.fr_C_O_noCOO(m)
        fr_NH0 = Descriptors.fr_NH0(m)
        fr_NH1 = Descriptors.fr_NH1(m)
        fr_Ndealkylation1 = Descriptors.fr_Ndealkylation1(m)
        fr_Ndealkylation2 = Descriptors.fr_Ndealkylation2(m)
        fr_Nhpyrrole = Descriptors.fr_Nhpyrrole(m)
        fr_amide = Descriptors.fr_amide(m)
        fr_aniline = Descriptors.fr_aniline(m)
        fr_benzene = Descriptors.fr_benzene(m)
        fr_ether = Descriptors.fr_ether(m)
        fr_halogen = Descriptors.fr_halogen(m)
        fr_hdrzone = Descriptors.fr_hdrzone(m)
        fr_imide = Descriptors.fr_imide(m)
        fr_morpholine = Descriptors.fr_morpholine(m)
        fr_nitro = Descriptors.fr_nitro(m)
        fr_nitro_arom = Descriptors.fr_nitro_arom(m)
        fr_nitro_arom_nonortho = Descriptors.fr_nitro_arom_nonortho(m)
        fr_para_hydroxylation = Descriptors.fr_para_hydroxylation(m)
        fr_piperdine = Descriptors.fr_piperdine(m)	
        fr_priamide = Descriptors.fr_priamide(m)
        fr_quatN = Descriptors.fr_quatN(m)
        fr_sulfonamd = Descriptors.fr_sulfonamd(m)
        fr_urea = Descriptors.fr_urea(m)
        FractionCSP3 = Descriptors.FractionCSP3(m)
        HeavyAtomCount = Descriptors.HeavyAtomCount(m)
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
        RingCount = Descriptors.RingCount(m)
													
        features = {'EState_VSA1' : EState_VSA1,
                    'EState_VSA2' : EState_VSA2,
                    'EState_VSA3' : EState_VSA3,
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
                    'VSA_EState7' : VSA_EState7,
                    'VSA_EState8' : VSA_EState8,
                    'VSA_EState9' : VSA_EState9,
                    'VSA_EState10' : VSA_EState10,
                    'QED' : QED,
                    'BertzCT' : BertzCT,
                    'Chi0' : Chi0,
                    'Chi1' : Chi1,
                    'Chi3n' : Chi3n,
                    'Chi4n' : Chi4n,
                    'Chi4v' : Chi4v,
                    'HallKierAlpha' : HallKierAlpha,
                    'MinAbsPartialCharge' : MinAbsPartialCharge,
                    'HeavyAtomMolWt' : HeavyAtomMolWt,
                    'MaxAbsPartialCharge' : MaxAbsPartialCharge,
                    'MinPartialCharge' : MinPartialCharge,
                    'ExactMolWt': ExactMolWt,
                    'MolWt' : MolWt,
                    'MaxPartialCharge' : MaxPartialCharge,
                    'LabuteASA' : LabuteASA,
                    'PEOE_VSA2' : PEOE_VSA2,
                    'PEOE_VSA3' : PEOE_VSA3,
                    'PEOE_VSA6' : PEOE_VSA6,
                    'PEOE_VSA7' : PEOE_VSA7,
                    'PEOE_VSA9' : PEOE_VSA9,
                    'PEOE_VSA10' : PEOE_VSA10,
                    'PEOE_VSA12' : PEOE_VSA12,
                    'PEOE_VSA13' : PEOE_VSA13,
                    'PEOE_VSA14' : PEOE_VSA14,
                    'SMR_VSA1' : SMR_VSA1,
                    'SMR_VSA2' : SMR_VSA2,
                    'SMR_VSA3' : SMR_VSA3,
                    'SMR_VSA5' : SMR_VSA5,
                    'SMR_VSA6' : SMR_VSA6,
                    'SMR_VSA7' : SMR_VSA7,
                    'SMR_VSA9' : SMR_VSA9,
                    'SMR_VSA10' : SMR_VSA10,
                    'SlogP_VSA1' : SlogP_VSA1,
                    'SlogP_VSA2' : SlogP_VSA2,
                    'SlogP_VSA3' : SlogP_VSA3,
                    'SlogP_VSA5' : SlogP_VSA5,
                    'SlogP_VSA6' : SlogP_VSA6,
                    'SlogP_VSA8' : SlogP_VSA8, 
                    'SlogP_VSA10' : SlogP_VSA10,
                    'SlogP_VSA12' : SlogP_VSA12,
                    'TPSA' : TPSA,
                    'MaxAbsEStateIndex' : MaxAbsEStateIndex,
                    'MaxEStateIndex' : MaxEStateIndex,
                    'MinEStateIndex' : MinEStateIndex,
                    'MinAbsEStateIndex' : MinAbsEStateIndex,
                    'fr_Al_COO' : fr_Al_COO,
                    'fr_ArN' : fr_ArN,
                    'fr_Ar_N' : fr_Ar_N,
                    'fr_Ar_NH' : fr_Ar_NH,
                    'fr_COO' : fr_COO,
                    'fr_COO2' : fr_COO2,
                    'fr_C_O' : fr_C_O,
                    'fr_C_O_noCOO' : fr_C_O_noCOO,
                    'fr_NH0' : fr_NH0,
                    'fr_NH1' : fr_NH1,
                    'fr_Ndealkylation1' : fr_Ndealkylation1,
                    'fr_Ndealkylation2' : fr_Ndealkylation2,
                    'fr_Nhpyrrole' : fr_Nhpyrrole,
                    'fr_amide' : fr_amide,
                    'fr_aniline' : fr_aniline,
                    'fr_benzene' : fr_benzene,
                    'fr_ether' : fr_ether,
                    'fr_halogen' : fr_halogen,
                    'fr_hdrzone' : fr_hdrzone,
                    'fr_imide' : fr_imide,
                    'fr_morpholine' : fr_morpholine,
                    'fr_nitro' : fr_nitro,
                    'fr_nitro_arom' : fr_nitro_arom,
                    'fr_nitro_arom_nonortho' : fr_nitro_arom_nonortho,
                    'fr_para_hydroxylation' : fr_para_hydroxylation,
                    'fr_piperdine' : fr_piperdine,
                    'fr_priamide' : fr_priamide,
                    'fr_quatN' : fr_quatN,
                    'fr_sulfonamd' : fr_sulfonamd,
                    'fr_urea' : fr_urea,
                    'FractionCSP3' : FractionCSP3,
                    'HeavyAtomCount' : HeavyAtomCount,
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
                    'NumSaturatedRings' : NumSaturatedRings,
                    'RingCount' : RingCount}
                
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
    Toxicity: The prediction of the model (-1 or 1).
    Category: The corresponding category of toxicity.
    '''

    name = []
    metal = []
    linker = []
    toxicity = []
    category = []

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
                print('Linker '+str(linkers[i])+' is more toxic.')
                category.append('More Toxic')
            else:
                print('Linker '+str(linkers[i])+' is less toxic.')
                category.append('Less Toxic')

            success_pred += 1

            name.append(str(data['Name'][i]).strip())
            metal.append(str(data['Metal'][i]).strip())
            linker.append(data['Linker'][i])

        except:
            print('Could not functionalise linker')
            success_pred += 0
    predictions = pd.DataFrame(list(zip(name, metal, linker, toxicity, category)), columns=['Name', 'Metal Centre', 'Linker', 'Toxicity', 'Category'])

    '''
    Below add the path to the CSV file where the results are to be saved.
    '''

    final_path = './data/predictions_toxic_twoclass.csv'
    predictions.to_csv(final_path)
    return predictions

print('Starting the screen.')
predictions = predict() 
print('Finished the high-throughput screen.')
