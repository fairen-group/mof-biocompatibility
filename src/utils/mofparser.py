'''
In this python script, we need to parse the MOF cif file in a way that is appropriate to be passed to moffragmentor. We do so because in some cases, there are issues encountered
in structures obtained from the CSD

Thus, in such cases, first the structures should be passed through this script, and then through the script for fragmentation. 
Again, the code is written to be compatible with a high-throughput screen.

A new directory of CIF files is saved post-parsing.

'''

from pymatgen.io.cif import CifParser
import os

'''
Below, add the path to the directory with the saved structures.
'''
directory = '/' # add the path to the directory
files = os.listdir(directory)
index = 0
while index < len(files):
    filename = files[index]
    try:
        with open(os.path.join(directory, filename)) as f:
            print('Opening '+ str(filename))
            path = directory + str(filename)
            '''
            Increasing the occupancy tolerance seems to do the trick.

            '''
            s = CifParser(path, occupancy_tolerance=100).get_structures()[0]
            '''
            Below, again specify the directory path where you want the parsed files to be saved.
            '''
            s.to('parsed/' + str(filename))
            print('Finished parsing ' + str(filename))
            index += 1
    except:
        print('Invalid cif file with no structures!')
        index += 1
    
