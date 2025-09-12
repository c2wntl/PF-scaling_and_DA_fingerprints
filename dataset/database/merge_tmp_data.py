import pandas as pd
import os 

if __name__ == "__main__":

    # high_density_host = ['BN','MoS2','WSe2','GaSe','InSe','P']
    high_density_host = ['BN','GaSe']
    low_density_host = ['MoS2','WSe2']

    tmp_high_df = pd.DataFrame()
    for host in high_density_host:
        filename = f'tmp_data/tmp-2dmd-cfid-{host}_high_density_defects.csv'
        df = pd.read_csv(filename,index_col=0)
        tmp_high_df = pd.concat([tmp_high_df,df])
        
    

    print(tmp_high_df)
    