import os
import pandas as pd

def read_sto_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the index of the 'endheader' line
    header_end = next(i for i, line in enumerate(lines) if line.strip() == 'endheader')
    
    # The next line after 'endheader' contains the column names
    column_names = lines[header_end + 1].strip().split('\t')
    
    # Read the data, starting from the line after the column names
    df = pd.read_csv(file_path, 
                     delimiter='\t', 
                     skiprows=header_end + 2,  # Skip header + column names row
                     names=column_names)  # Use the extracted column names
    
    return df

def readMotionFile(filename):
    """ Reads OpenSim .sto files.
    Parameters
    ----------
    filename: absolute path to the .sto file
    Returns
    -------
    header: the header of the .sto
    labels: the labels of the columns
    data: an array of the data
    """

    if not os.path.exists(filename):
        print('file do not exists')

    file_id = open(filename, 'r')

    # read header
    next_line = file_id.readline()
    print(file_id)
    exit()
    header = [next_line]
    nc = 0
    nr = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()

    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data