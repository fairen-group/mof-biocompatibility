from moffragmentor import MOF

hkust = MOF.from_cif('/Users/dhruvmenon/Documents/PhD/ml_biocompatibility/source_code/HKUST1.cif')
print('MOF imported succesfully')
print('Starting fragmentation')
fragments = hkust.fragment()
print('Finished fragmentation')
smiles = fragments.linkers[0].smiles
print(smiles)