import pandas as pd
import os
import time
import numpy as np

relevant_path = "/home/pi/Crypto/logs/archive/"
cleanup = True
included_extensions = ['csv']
file_names = [fn for fn in os.listdir(relevant_path)
              if any(fn.endswith(ext) for ext in included_extensions)]

if len(file_names) > 1:
    print('Found {} files'.format(len(file_names)))
    all_data = np.zeros((0,0))

    for f in file_names:
        new_data = pd.read_csv('{}/{}'.format(relevant_path,f),header=None).values
        if all_data.shape[0]>0:
            all_data = np.concatenate((all_data,new_data),axis=0)
        else:
            all_data = new_data


    try:
        toWrite = pd.DataFrame(data = all_data)
        toWrite = toWrite.sort_values(axis=0,by=0)
        fn = '{}/rebalance_log.csv'.format(relevant_path,time.strftime('%m_%d_%y_%H_%M_%S'))
        toWrite.to_csv(fn,header = False,index = False)
        print('Wrote logs to {}'.format(fn))
        if cleanup:
            num_files = 0
            for f in file_names:
                if f != 'rebalance_log.csv':
                    os.remove('{}/{}'.format(relevant_path,f))
                    num_files += 1
            print("Cleaned up {} files".format(num_files))
    except:
        print('Error writing to log file!')
    
else:
    print('No files to archive')



    
