'''
Here, we will featurize individual molecules when passed through the SMILES representation. 
In total, we calculate 198 descriptors - divided into different types based on the information of the molecule they capture.
'''
from rdkit import Chem 
from rdkit.Chem import Descriptors

# from rdkit.Chem import rdMolDescriptors
# from rdkit.Chem import Descriptors3D 
# While 3D descriptors are not particularly applicable in this case, they will prove to be useful in certain instances.

'''
In order to understand the physical meaning of these descriptors, please refer to the supporint information.
The first set of descriptors we will calculate are EState (Electro-topological State) Descriptors (from rdkit)
'''

def EState_Desc(smiles):
    m = Chem.MolFromSmiles(smiles) # extract a molecule from a passed SMILES string.
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
    EState_VSA11 = Descriptors.EState_VSA11(m)

    VSA_EState1 = Descriptors.VSA_EState1(m)
    VSA_EState2 = Descriptors.VSA_EState2(m)
    VSA_EState3 = Descriptors.VSA_EState3(m)
    VSA_EState4 = Descriptors.VSA_EState4(m)
    VSA_EState5 = Descriptors.VSA_EState5(m)
    VSA_EState6 = Descriptors.VSA_EState6(m)
    VSA_EState7 = Descriptors.VSA_EState7(m)
    VSA_EState8 = Descriptors.VSA_EState8(m)
    VSA_EState9 = Descriptors.VSA_EState9(m)
    VSA_EState10 = Descriptors.VSA_EState10(m)

    EState = [EState_VSA1, EState_VSA2, EState_VSA3, EState_VSA4, EState_VSA5, EState_VSA6, EState_VSA7, EState_VSA8, EState_VSA9,
        EState_VSA10, EState_VSA11, VSA_EState1, VSA_EState2, VSA_EState3, VSA_EState4, VSA_EState5, VSA_EState6, VSA_EState7, VSA_EState8,
        VSA_EState9, VSA_EState10] # we will return the EState descriptors as a list.
    
    return EState 

'''
Next let us calculate the QED - Quantitative Estimate of Drug-likeliness. 
We discuss this descriptor in detail in both the main manuscript, and the supporting information.

'''
def QED(smiles):
    m = Chem.MolFromSmiles(smiles)
    qed = Descriptors.qed(m)
    QED = [qed]
    return QED

'''
Next, we calculate topological/topochemical descriptors - we call these geometric descriptors. 
Just for some clarity, we provide a small description - also covered in the supporting information.

'''

def GeoDesc(smiles):
    m = Chem.MolFromSmiles(smiles)
    BalabanJ = Descriptors.BalabanJ(m) # Balaban's J value
    BertzCT	 = Descriptors.BertzCT(m) # A topological index meant to quantify "complexity"
    Chi0 = Descriptors.Chi0(m) # From equations (1),(9) and (10) of Rev. Comp. Chem. vol 2, 367-422, (1991)
    Chi0n = Descriptors.Chi0n(m) # Similar to Hall Kier Chi0v, but uses nVal instead of valence. 
    Chi0v = Descriptors.Chi0v(m) # From equations (5),(9) and (10) of Rev. Comp. Chem. vol 2, 367-422, (1991)
    Chi1 = Descriptors.Chi1(m) # From equations (1),(11) and (12) of Rev. Comp. Chem. vol 2, 367-422, (1991)
    Chi1n = Descriptors.Chi1n(m) # Similar to Hall Kier Chi1v, but uses nVal instead of valence
    Chi1v = Descriptors.Chi1v(m) # From equations (5),(11) and (12) of Rev. Comp. Chem. vol 2, 367-422, (1991)
    Chi2n = Descriptors.Chi2n(m) # Similar to Hall Kier Chi2v, but uses nVal instead of valence. 
    Chi3n = Descriptors.Chi3n(m) # Similar to Hall Kier Chi3v, but uses nVal instead of valence. 
    Chi3v = Descriptors.Chi3v(m) # From equations (5),(15) and (16) of Rev. Comp. Chem. vol 2, 367-422, (1991)
    Chi4n = Descriptors.Chi4n(m) # Similar to Hall Kier Chi4v, but uses nVal instead of valence. 
    Chi4v = Descriptors.Chi4v(m) # From equations (5),(15) and (16) of Rev. Comp. Chem. vol 2, 367-422, (1991).
    HallKierAlpha = Descriptors.HallKierAlpha(m) # The Hall-Kier alpha value for a molecule
    Ipc	= Descriptors.Ipc(m) # The information content of the coefficients of the characteristic polynomial of the adjacency matrix of a hydrogen-suppressed graph of a molecule
    Kappa1 = Descriptors.Kappa1(m) # Hall-Kier Kappa1 value
    Kappa2 = Descriptors.Kappa2(m) # Hall-Kier Kappa2 value
    Kappa3 = Descriptors.Kappa3(m) # Hall-Kier Kappa3 value

    GeoDesc = [BalabanJ, BertzCT, Chi0, Chi0n, Chi0v, Chi1, Chi1n, Chi1v, Chi2n, Chi3n, Chi3v, Chi4n, Chi4v, HallKierAlpha, Ipc, 
    Kappa1, Kappa2, Kappa3]

    return GeoDesc

'''
Next, we calculate general descriptors - capturing general molecular properties such as molecular weight. 
Just for some clarity, we provide a small description - also covered in the supporting information.

'''
def GenDesc(smiles):
    m = Chem.MolFromSmiles(smiles)
    MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(m) #Minimal absolute partial charge
    NumRadicalElectrons	= Descriptors.NumRadicalElectrons(m) #The number of radical electrons the molecule has (says nothing about spin state)
    FpDensityMorgan2 = Descriptors.FpDensityMorgan2(m) #Morgan fingerprint, radius 2
    FpDensityMorgan3 = Descriptors.FpDensityMorgan3(m) #Morgan fingerprint, radius 3
    FpDensityMorgan1 = Descriptors.FpDensityMorgan1(m) #Morgan fingerprint, radius 1
    HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(m) #The average molecular weight of the molecule ignoring hydrogens
    MaxAbsPartialCharge = Descriptors.MaxAbsPartialCharge(m) #Maximum absolute partial charge
    MinPartialCharge = Descriptors.MinPartialCharge(m) #Minimal partial charge
    ExactMolWt = Descriptors.ExactMolWt(m) #The exact molecular weight of the molecule
    MolWt = Descriptors.MolWt(m) #The average molecular weight of the molecule
    NumValenceElectrons	= Descriptors.NumValenceElectrons(m) #The number of valence electrons the molecule has
    MaxPartialCharge = Descriptors.MaxPartialCharge(m) #Maximum partial charge

    GenDesc = [MinAbsPartialCharge, NumRadicalElectrons, FpDensityMorgan2, FpDensityMorgan3, FpDensityMorgan1, HeavyAtomMolWt, 
    MaxAbsPartialCharge, MinPartialCharge, ExactMolWt, MolWt, NumValenceElectrons, MaxPartialCharge]

    return GenDesc

'''
We now calculate descriptors relating to solubility and molar refractivity, when measured using Crippen's Approach.
'''

def Crippen(smiles):
    m = Chem.MolFromSmiles(smiles)
    MolLogP = Descriptors.MolLogP(m)
    MolMR = Descriptors.MolMR(m)

    Crippen = [MolLogP, MolMR]
    return Crippen

'''
Next, we calculate some molecular surface area descriptors.
'''

def MolSurf(smiles):
    m = Chem.MolFromSmiles(smiles)
    LabuteASA = Descriptors.LabuteASA(m) #Labute's Approximate Surface Area (ASA from MOE)
    PEOE_VSA1 = Descriptors.PEOE_VSA1(m) #MOE Charge VSA Descriptor 1 (-inf < x < -0.30)
    PEOE_VSA10 = Descriptors.PEOE_VSA10(m) #MOE Charge VSA Descriptor 10 ( 0.10 <= x < 0.15)
    PEOE_VSA11 = Descriptors.PEOE_VSA11(m) #MOE Charge VSA Descriptor 11 ( 0.15 <= x < 0.20)
    PEOE_VSA12 = Descriptors.PEOE_VSA12(m) #MOE Charge VSA Descriptor 12 ( 0.20 <= x < 0.25)
    PEOE_VSA13 = Descriptors.PEOE_VSA13(m) #MOE Charge VSA Descriptor 13 ( 0.25 <= x < 0.30)
    PEOE_VSA14 = Descriptors.PEOE_VSA14(m) #MOE Charge VSA Descriptor 14 ( 0.30 <= x < inf)
    PEOE_VSA2 = Descriptors.PEOE_VSA2(m) #MOE Charge VSA Descriptor 2 (-0.30 <= x < -0.25)
    PEOE_VSA3 = Descriptors.PEOE_VSA3(m) #MOE Charge VSA Descriptor 3 (-0.25 <= x < -0.20)
    PEOE_VSA4 = Descriptors.PEOE_VSA4(m) #MOE Charge VSA Descriptor 4 (-0.20 <= x < -0.15)
    PEOE_VSA5 = Descriptors.PEOE_VSA5(m) #MOE Charge VSA Descriptor 5 (-0.15 <= x < -0.10)
    PEOE_VSA6 = Descriptors.PEOE_VSA6(m) #MOE Charge VSA Descriptor 6 (-0.10 <= x < -0.05)
    PEOE_VSA7 = Descriptors.PEOE_VSA7(m) #MOE Charge VSA Descriptor 7 (-0.05 <= x < 0.00)
    PEOE_VSA8 = Descriptors.PEOE_VSA8(m) #MOE Charge VSA Descriptor 8 ( 0.00 <= x < 0.05)
    PEOE_VSA9 = Descriptors.PEOE_VSA9(m) #MOE Charge VSA Descriptor 9 ( 0.05 <= x < 0.10)
    SMR_VSA1 = Descriptors.SMR_VSA1(m) #MOE MR VSA Descriptor 1 (-inf < x < 1.29)
    SMR_VSA10 = Descriptors.SMR_VSA10(m) #MOE MR VSA Descriptor 10 ( 4.00 <= x < inf)
    SMR_VSA2 = Descriptors.SMR_VSA2(m) #MOE MR VSA Descriptor 2 ( 1.29 <= x < 1.82)
    SMR_VSA3 = Descriptors.SMR_VSA3(m) #MOE MR VSA Descriptor 3 ( 1.82 <= x < 2.24)
    SMR_VSA4 = Descriptors.SMR_VSA4(m) #MOE MR VSA Descriptor 4 ( 2.24 <= x < 2.45)
    SMR_VSA5 = Descriptors.SMR_VSA5(m) #MOE MR VSA Descriptor 5 ( 2.45 <= x < 2.75)
    SMR_VSA6 = Descriptors.SMR_VSA6(m) #MOE MR VSA Descriptor 6 ( 2.75 <= x < 3.05)
    SMR_VSA7 = Descriptors.SMR_VSA7(m) #MOE MR VSA Descriptor 7 ( 3.05 <= x < 3.63)
    SMR_VSA8 = Descriptors.SMR_VSA8(m) #MOE MR VSA Descriptor 8 ( 3.63 <= x < 3.80)
    SMR_VSA9 = Descriptors.SMR_VSA9(m) #MOE MR VSA Descriptor 9 ( 3.80 <= x < 4.00)
    SlogP_VSA1 = Descriptors.SlogP_VSA1(m) #MOE logP VSA Descriptor 1 (-inf < x < -0.40)
    SlogP_VSA10	= Descriptors.SlogP_VSA10(m) #MOE logP VSA Descriptor 10 ( 0.40 <= x < 0.50)
    SlogP_VSA11	= Descriptors.SlogP_VSA11(m) #MOE logP VSA Descriptor 11 ( 0.50 <= x < 0.60)
    SlogP_VSA12	= Descriptors.SlogP_VSA12(m) #MOE logP VSA Descriptor 12 ( 0.60 <= x < inf)
    SlogP_VSA2 = Descriptors.SlogP_VSA2(m) #MOE logP VSA Descriptor 2 (-0.40 <= x < -0.20)
    SlogP_VSA3 = Descriptors.SlogP_VSA3(m) #MOE logP VSA Descriptor 3 (-0.20 <= x < 0.00)
    SlogP_VSA4 = Descriptors.SlogP_VSA4(m) #MOE logP VSA Descriptor 4 ( 0.00 <= x < 0.10)
    SlogP_VSA5 = Descriptors.SlogP_VSA5(m) #MOE logP VSA Descriptor 5 ( 0.10 <= x < 0.15)
    SlogP_VSA6 = Descriptors.SlogP_VSA6(m) #MOE logP VSA Descriptor 6 ( 0.15 <= x < 0.20)
    SlogP_VSA7 = Descriptors.SlogP_VSA7(m) #MOE logP VSA Descriptor 7 ( 0.20 <= x < 0.25)
    SlogP_VSA8 = Descriptors.SlogP_VSA8(m) #MOE logP VSA Descriptor 8 ( 0.25 <= x < 0.30)
    SlogP_VSA9 = Descriptors.SlogP_VSA9(m) #MOE logP VSA Descriptor 9 ( 0.30 <= x < 0.40)
    TPSA = Descriptors.TPSA(m) #The polar surface area of a molecule based upon fragments.

    MolSurf = [LabuteASA, PEOE_VSA1, PEOE_VSA2, PEOE_VSA3, PEOE_VSA4, PEOE_VSA5, PEOE_VSA6, PEOE_VSA7, PEOE_VSA8, PEOE_VSA9, PEOE_VSA10, 
    PEOE_VSA11, PEOE_VSA12, PEOE_VSA13, PEOE_VSA14, SMR_VSA1, SMR_VSA2, SMR_VSA3, SMR_VSA4, SMR_VSA5, SMR_VSA6, SMR_VSA7, SMR_VSA8,
    SMR_VSA9, SMR_VSA10, SlogP_VSA1, SlogP_VSA2, SlogP_VSA3, SlogP_VSA4, SlogP_VSA5, SlogP_VSA6, SlogP_VSA7, SlogP_VSA8, SlogP_VSA9,
    SlogP_VSA10, SlogP_VSA11, SlogP_VSA12, TPSA]

    return MolSurf

# descriptors based on a molecule's 3D structure
# def Desc3d(smiles):
#    m = Chem.MolFromSmiles(smiles)
#    PMI1 = Descriptors3D.PMI1(m)	
#    #First (smallest) principal moment of inertia
#    PMI2 = Descriptors3D.PMI2(m)
#    #Second principal moment of inertia
#    PMI3 = Descriptors3D.PMI3(m)
#    #Third (largest) principal moment of inertia
#    NPR1 = Descriptors3D.NPR1(m)
#    #Normalized principal moments ratio 1 (=I1/I3)
#    NPR2 = Descriptors3D.NPR2(m)	
#    #Normalized principal moments ratio 2 (=I2/I3)
#    RadiusOfGyration = Descriptors3D.RadiusOfGyration(m)	
#    #Radius of gyration
#    InertialShapeFactor = Descriptors3D.InertialShapeFactor(m)	
#    #Inertial shape factor
#    Eccentricity = Descriptors3D.Eccentricity(m)
#    #Molecular eccentricity
#    Asphericity = Descriptors3D.Asphericity(m)
#    #Molecular asphericity
#    SpherocityIndex	= Descriptors3D.SpherocityIndex(m)
#    #Molecular spherocity index

#    Desc3D = [PMI1, PMI2, PMI3, NPR1, NPR2, RadiusOfGyration, InertialShapeFactor, Eccentricity, Asphericity, SpherocityIndex]
#    return Desc3D 

'''
We calculate descriptors based on basic EState

'''
def EState_Basic(smiles):
    m = Chem.MolFromSmiles(smiles)
    MaxAbsEStateIndex = Descriptors.MaxAbsEStateIndex(m) #Maximum absolute EState index
    MaxEStateIndex = Descriptors.MaxEStateIndex(m) #Maximum EState index
    MinEStateIndex = Descriptors.MinEStateIndex(m) #Minimum EState index
    MinAbsEStateIndex = Descriptors.MinAbsEStateIndex(m) #Minimum absolute EState index

    EState_Basic = [MaxAbsEStateIndex, MaxEStateIndex, MinEStateIndex, MinAbsEStateIndex]
    return EState_Basic

'''
We calculate fragment-based descriptors

'''
def Frag_Desc(smiles):
    m = Chem.MolFromSmiles(smiles)
    fr_Al_COO = Descriptors.fr_Al_COO(m) #Number of aliphatic carboxylic acids
    fr_Al_OH = Descriptors.fr_Al_OH(m) # 	Number of aliphatic hydroxyl groups
    fr_Al_OH_noTert = Descriptors.fr_Al_OH_noTert(m) #	Number of aliphatic hydroxyl groups excluding tert-OH
    fr_ArN = Descriptors.fr_Ar_N(m) #	Number of N functional groups attached to aromatics
    fr_Ar_COO = Descriptors.fr_Ar_COO(m) #	Number of Aromatic carboxylic acide
    fr_Ar_N = Descriptors.fr_Ar_N(m) #	Number of aromatic nitrogens
    fr_Ar_NH = Descriptors.fr_Ar_NH(m) #	Number of aromatic amines
    fr_Ar_OH = Descriptors.fr_Ar_OH(m) #	Number of aromatic hydroxyl groups
    fr_COO = Descriptors.fr_COO(m) #	Number of carboxylic acids
    fr_COO2 = Descriptors.fr_COO2(m) #	Number of carboxylic acids
    fr_C_O = Descriptors.fr_C_O(m) #	Number of carbonyl O
    fr_C_O_noCOO = Descriptors.fr_C_O_noCOO(m) #	Number of carbonyl O, excluding COOH
    fr_C_S = Descriptors.fr_C_S(m) #	Number of thiocarbonyl
    fr_HOCCN = Descriptors.fr_HOCCN(m) #	Number of C(OH)CCN-Ctert-alkyl or C(OH)CCNcyclic
    fr_Imine = Descriptors.fr_Imine(m) #	Number of Imines
    fr_NH0 = Descriptors.fr_NH0(m) #	Number of Tertiary amines
    fr_NH1 = Descriptors.fr_NH1(m) #	Number of Secondary amines
    fr_NH2 = Descriptors.fr_NH2(m) #	Number of Primary amines
    fr_N_O = Descriptors.fr_N_O(m) #	Number of hydroxylamine groups
    fr_Ndealkylation1 = Descriptors.fr_Ndealkylation1(m) #	Number of XCCNR groups
    fr_Ndealkylation2 = Descriptors.fr_Ndealkylation2(m) #	Number of tert-alicyclic amines (no heteroatoms, not quinine-like bridged N)
    fr_Nhpyrrole = Descriptors.fr_Nhpyrrole(m) #	Number of H-pyrrole nitrogens
    fr_SH = Descriptors.fr_SH(m) #	Number of thiol groups
    fr_aldehyde = Descriptors.fr_aldehyde(m) #	Number of aldehydes
    fr_alkyl_carbamate = Descriptors.fr_alkyl_carbamate(m) #	Number of alkyl carbamates (subject to hydrolysis)
    fr_alkyl_halide	= Descriptors.fr_alkyl_halide(m) #Number of alkyl halides
    fr_allylic_oxid	= Descriptors.fr_allylic_oxid(m) #Number of allylic oxidation sites excluding steroid dienone
    fr_amide = Descriptors.fr_amide(m) #Number of amides
    fr_amidine = Descriptors.fr_amidine(m) #Number of amidine groups
    fr_aniline = Descriptors.fr_aniline(m) #	Number of anilines
    fr_aryl_methyl = Descriptors.fr_aryl_methyl(m) #	Number of aryl methyl sites for hydroxylation
    fr_azide = Descriptors.fr_azide(m) #	Number of azide groups
    fr_azo = Descriptors.fr_azo(m) #	Number of azo groups
    fr_barbitur = Descriptors.fr_barbitur(m) #	Number of barbiturate groups
    fr_benzene = Descriptors.fr_benzene(m) #	Number of benzene rings
    fr_benzodiazepine = Descriptors.fr_benzodiazepine(m) #	Number of benzodiazepines with no additional fused rings
    fr_bicyclic = Descriptors.fr_bicyclic(m) #	Bicyclic
    fr_diazo = Descriptors.fr_diazo(m) #	Number of diazo groups
    fr_dihydropyridine = Descriptors.fr_dihydropyridine(m) #	Number of dihydropyridines
    fr_epoxide = Descriptors.fr_epoxide(m) #	Number of epoxide rings
    fr_ester = Descriptors.fr_ester(m) #	Number of esters
    fr_ether = Descriptors.fr_ether(m) #	Number of ether oxygens (including phenoxy)
    fr_furan = Descriptors.fr_furan(m) #	Number of furan rings
    fr_guanido = Descriptors.fr_guanido(m) #	Number of guanidine groups
    fr_halogen = Descriptors.fr_halogen(m) #	Number of halogens
    fr_hdrzine = Descriptors.fr_hdrzine(m) #	Number of hydrazine groups
    fr_hdrzone = Descriptors.fr_hdrzone(m) #	Number of hydrazone groups
    fr_imidazole = Descriptors.fr_imidazole(m) #	Number of imidazole rings
    fr_imide = Descriptors.fr_imide(m) #	Number of imide groups
    fr_isocyan = Descriptors.fr_isocyan(m) #	Number of isocyanates
    fr_isothiocyan = Descriptors.fr_isothiocyan(m) #	Number of isothiocyanates
    fr_ketone = Descriptors.fr_ketone(m) #	Number of ketones
    fr_ketone_Topliss = Descriptors.fr_ketone_Topliss(m) #	Number of ketones excluding diaryl, a,b-unsat
    fr_lactam = Descriptors.fr_lactam(m) #	Number of beta lactams
    fr_lactone = Descriptors.fr_lactone(m) #	Number of cyclic esters (lactones)
    fr_methoxy = Descriptors.fr_methoxy(m) #	Number of methoxy groups -OCH3
    fr_morpholine = Descriptors.fr_morpholine(m) #	Number of morpholine rings
    fr_nitrile = Descriptors.fr_nitrile(m) #	Number of nitriles
    fr_nitro = Descriptors.fr_nitro(m) #	Number of nitro groups
    fr_nitro_arom = Descriptors.fr_nitro_arom(m) #	Number of nitro benzene ring substituent
    fr_nitro_arom_nonortho = Descriptors.fr_nitro_arom_nonortho(m) #	Number of non-ortho nitro benzene ring substituents
    fr_nitroso = Descriptors.fr_nitroso(m) #	Number of nitroso groups, excluding NO2
    fr_oxazole = Descriptors.fr_oxazole(m) #	Number of oxazole rings
    fr_oxime = Descriptors.fr_oxime(m) #	Number of oxime groups
    fr_para_hydroxylation = Descriptors.fr_para_hydroxylation(m) #	Number of para-hydroxylation sites
    fr_phenol = Descriptors.fr_phenol(m) #	Number of phenols
    fr_phenol_noOrthoHbond = Descriptors.fr_phenol_noOrthoHbond(m) #	Number of phenolic OH excluding ortho intramolecular Hbond substituents
    fr_phos_acid = Descriptors.fr_phos_acid(m) #	Number of phosphoric acid groups
    fr_phos_ester = Descriptors.fr_phos_ester(m) #	Number of phosphoric ester groups
    fr_piperdine = Descriptors.fr_piperdine(m) #	Number of piperdine rings
    fr_piperzine = Descriptors.fr_piperzine(m) #	Number of piperzine rings
    fr_priamide = Descriptors.fr_priamide(m) #	Number of primary amides
    fr_prisulfonamd = Descriptors.fr_prisulfonamd(m) #	Number of primary sulfonamides
    fr_pyridine = Descriptors.fr_pyridine(m) #	Number of pyridine rings
    fr_quatN = Descriptors.fr_quatN(m) #	Number of quarternary nitrogens
    fr_sulfide = Descriptors.fr_sulfide(m) #	Number of thioether
    fr_sulfonamd = Descriptors.fr_sulfonamd(m) #	Number of sulfonamides
    fr_sulfone = Descriptors.fr_sulfone(m) #	Number of sulfone groups
    fr_term_acetylene = Descriptors.fr_term_acetylene(m) #	Number of terminal acetylenes
    fr_tetrazole = Descriptors.fr_tetrazole(m) #	Number of tetrazole rings
    fr_thiazole = Descriptors.fr_thiazole(m) #	Number of tetrazole rings
    fr_thiocyan = Descriptors.fr_thiocyan(m) #	Number of thiocyanates
    fr_unbrch_alkane = Descriptors.fr_unbrch_alkane(m) #	Number of unbranched alkanes of at least 4 members (excludes halogenated alkanes)
    fr_urea = Descriptors.fr_urea(m) #Number of urea groups

    fragDesc = [fr_Al_COO, fr_Al_OH, fr_Al_OH_noTert, fr_ArN ,fr_Ar_COO, fr_Ar_N, fr_Ar_NH, fr_Ar_OH, fr_COO,
    fr_COO2, fr_C_O, fr_C_O_noCOO, fr_C_S, fr_HOCCN, fr_Imine, fr_NH0, fr_NH1, fr_NH2, fr_N_O, fr_Ndealkylation1,
    fr_Ndealkylation2, fr_Nhpyrrole, fr_SH, fr_aldehyde, fr_alkyl_carbamate, fr_alkyl_halide, fr_allylic_oxid, fr_amide, 
    fr_amidine, fr_aniline, fr_aryl_methyl, fr_azide, fr_azo, fr_barbitur, fr_benzene, fr_benzodiazepine, fr_bicyclic, 
    fr_diazo, fr_dihydropyridine, fr_epoxide, fr_ester, fr_ether, fr_furan, fr_guanido, fr_halogen, fr_hdrzine, fr_hdrzone,
    fr_imidazole, fr_imide, fr_isocyan, fr_isothiocyan, fr_ketone, fr_ketone_Topliss, fr_lactam, fr_lactone, fr_methoxy, 
    fr_morpholine, fr_nitrile, fr_nitro, fr_nitro_arom, fr_nitro_arom_nonortho, fr_nitroso, fr_oxazole, fr_oxime, fr_para_hydroxylation,
    fr_phenol, fr_phenol_noOrthoHbond, fr_phos_acid, fr_phos_ester, fr_piperdine, fr_piperzine, fr_priamide, fr_prisulfonamd,
    fr_pyridine, fr_quatN, fr_sulfide, fr_sulfonamd, fr_sulfone, fr_term_acetylene, fr_tetrazole, fr_thiazole, fr_thiocyan, 
    fr_unbrch_alkane, fr_urea]

    return fragDesc

'''
At last, we calculate fragments based on Lipinski's parameters 
'''

def LipinskiDesc(smiles):
    m = Chem.MolFromSmiles(smiles)
    FractionCSP3 = Descriptors.FractionCSP3(m) #The fraction of C atoms that are SP3 hybridized
    HeavyAtomCount = Descriptors.HeavyAtomCount(m) #The number of heavy atoms a molecule
    NHOHCount = Descriptors.NHOHCount(m) #The number of NHs or OHs
    NOCount = Descriptors.NOCount(m) #The number of Nitrogens and Oxygens
    NumAliphaticCarbocycles = Descriptors.NumAliphaticCarbocycles(m) #The number of aliphatic (containing at least one non-aromatic bond) carbocycles for a molecule
    NumAliphaticHeterocycles = Descriptors.NumAliphaticCarbocycles(m) #The number of aliphatic (containing at least one non-aromatic bond) heterocycles for a molecule
    NumAliphaticRings = Descriptors.NumAliphaticRings(m) #The number of aliphatic (containing at least one non-aromatic bond) rings for a molecule
    NumAromaticCarbocycles = Descriptors.NumAromaticCarbocycles(m) #The number of aromatic carbocycles for a molecule
    NumAromaticHeterocycles = Descriptors.NumAromaticHeterocycles(m) #The number of aromatic heterocycles for a molecule
    NumAromaticRings = Descriptors.NumAromaticRings(m) #The number of aromatic rings for a molecule
    NumHAcceptors = Descriptors.NumHAcceptors(m) #The number of Hydrogen Bond Acceptors
    NumHDonors = Descriptors.NumHDonors(m) #The number of Hydrogen Bond Donors
    NumHeteroatoms = Descriptors.NumHeteroatoms(m) #The number of Heteroatoms
    NumRotatableBonds = Descriptors.NumRotatableBonds(m) #The number of Rotatable Bonds
    NumSaturatedCarbocycles = Descriptors.NumSaturatedCarbocycles(m) #The number of saturated carbocycles for a molecule
    NumSaturatedHeterocycles = Descriptors.NumSaturatedHeterocycles(m) #The number of saturated heterocycles for a molecule
    NumSaturatedRings = Descriptors.NumSaturatedRings(m) #The number of saturated rings for a molecule
    RingCount = Descriptors.RingCount(m) #Ring count

    LipinskiDesc = [FractionCSP3, HeavyAtomCount, NHOHCount, NOCount, NumAliphaticCarbocycles, NumAliphaticHeterocycles, 
    NumAliphaticRings, NumAromaticCarbocycles, NumAromaticHeterocycles, NumAromaticRings, NumHDonors, NumHeteroatoms, NumRotatableBonds,
    NumSaturatedCarbocycles, NumSaturatedHeterocycles, NumSaturatedRings, RingCount]

    return LipinskiDesc
