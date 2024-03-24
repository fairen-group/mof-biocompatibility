'''
In case you chose to use moffragmentor for deconstructing the MOF structure - 
This script utilises moffragmentor developed by Jablonka et al. (2023) to fragment MOFs in a high-throughput manner.
Here, you need to save cleaned CIF files (those passed through the parser code) into a directory, and add the path of 
the directory to the relevant part of the code.

This could take a while depending on the number of structures to be fragmented. Parallelisation is recommended in such cases.

'''

from moffragmentor import MOF
import os
import pandas as pd
import sys

f = open("output", "w") #1 - in case this is being run on a cluster, the user can also set path to relevant output directory, 
sys.stdout = f #2

'''
Supply the name of the directory below. 

'''
directory = '' 
files = os.listdir(directory)
index = 0

'''
In the original code, we followed the naming as per the conventions of the CSD. 

'''
name = [] # storing the name of the MOF - reference: Cambridge Structural Database (CSD)
metal = [] # storing the metallic center.
linker = [] # storing the linker post-fragmentation - in the SMILES format.

'''
We will essentially loop over the entire directory, going file by file.
'''

while index < len(files):
    filename = files[index]
    with open(os.path.join(directory, filename)) as f:
        print('Fragmenting MOF '+ str(index)) # keeping track of which MOF is being fragmented - in order to troubleshoot in case of errors.
        print('Opening '+ str(filename)) # opening the cif file
        mof_name = str(filename) # extracting the name of the MOF - important for tracking while handling large libraries.
        '''
        If extracted from the CSD, in some cases, you may need to remove common suffixes from filenames.  In such a case, replace the code to this:
        mof_name = str(filename).removesuffix('.XYZ.cif') - replace accordingly. 
        '''
        name.append(mof_name) # storing the name of the MOF to list
        print('Loading structure') 
        path = directory + str(filename)
        print('Loaded structure')
        try:
            '''
            If the CIF file is not correct, this will throw an error.
            '''
            mof = MOF.from_cif(path) # converting cif file to MOF structure 
        except:
            print('Error handling .cif file, please make sure that adequate format used.')
            pass # in case of an error in high-throughput, we will move forward regardless.
        print('Starting fragmentation')
        try: # if fragmentation fails, we will just ignore the MOF structure all-together
            fragments = mof.fragment() # fragmenting MOF structure
            print('Finished fragmentation')
            node = fragments.nodes[0] # extracting node
            metal.append(node) # appending node to list
            print('Identified node:' +str(node))
            smiles = fragments.linkers[0].smiles # extracting linker in the SMILES representation.
            linker.append(smiles) # appending linker to list
            print('Identified linker: '+str(smiles))
            print('Saving linker')
        except: 
            print('Could not fragment this MOF, please check .cif file')
            pass
        index += 1

fragmented = pd.DataFrame(
        {'Name' : name,
         'Metal Centre' : metal,
         'Linker' : linker
        })
fragmented.to_csv('fragmented.csv') # add path accordingly.
print('Finished')

f.close() #3
