'''

A code that can automatically convert a non-p1 CIF file to a p1 symmetry in a high-throughput manner.
The only dependency is openbabel - it is recommended that this code is run in an environment with openbabel setup.

'''

import os

'''

directory_init - add a filepath to the directory with all non-p1 CIF files.
directory_final - this is where the p1 CIF files will be saved - add accordingly.

'''

directory_init = ''
directory_final = ''

files = os.listdir(directory_init)
index = 0
while index < len(files):
    filename = files[index]
    filepath = directory_init + filename
    print('Opening: ' + filename)
    non_P1 = filename[:-4]
    P1 = non_P1 + '_P1.cif'
    P1_path = directory_final + P1
    '''
    
    Below is essentially the query that is run on the terminal to call openbabel and do the symmetry conversion. 
    This is a fairly quick job - can be fairly high-throughput.
    
    '''
    query = 'obabel -icif ' + filepath + ' -ocif -O ' + P1_path + ' --fillUC strict'
    os.system(query) 
    print('P1 generated.')
    index += 1