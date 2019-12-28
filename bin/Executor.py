import matplotlib 
matplotlib.use('Agg')

import configparser
import pandas as pd
import os, ast, time

from SL0_GeneralFunc import GetBackSomeDirectoryAndGetAbsPath, TimeCataloging, CreateKey, LevBasedPrint, AddRecommendation
from SL1_ImportData import ImportData
from SL2_DataManagerOfHoldoutDB import GenerateHoldoutDB, AddObsFromHoldoutDB


def execute_clust():
    
    
    CycleType = 'TrainTest'
    
    
    
    
    t0 = int(time.time())
    print('Execution Start ' + str(t0))
    
    ## Waiting till GlTest is occuring
    if os.path.exists('FlagRaised_GlTestOccurring_DontRunTrainTest'):
        print('Flag: "FlagRaised_GlTestOccurring_DontRunTrainTest" is raised therfore waiting till it completes.')
        while(os.path.exists('FlagRaised_GlTestOccurring_DontRunTrainTest')):
            time.sleep(1)
    ## Raising Flag So That Predict Doesn't Occur While This Is Running
    pd.DataFrame().to_csv('FlagRaised_TrainingOccurring_DontRunGlTest')
    
    
    ConfigFilePath = '../config/Config.ini'
    _, absModConfPath = GetBackSomeDirectoryAndGetAbsPath(ConfigFilePath)

    config = configparser.ConfigParser()
    config.read(absModConfPath)
    config['IterationAim']['CycleType'] = CycleType

    
    ## Import Data
    
    InputRawDF = ImportData(config)
    InputRawDF.head()

    
    ## Developing HoldoutDB
    '''
    Though We are importing HoldoutDB, it's not used further -- in later/other version it could be made to use 
    domain based filter here and passing these to get those observation from HoldoutDB.
    '''
    CriticalClassSer = InputRawDF['isBotHits'] > 0
    HoldoutDB = GenerateHoldoutDB(InputRawDF, CriticalClassSer, config)

    ## Accessing Observations from HoldoutDB
    InputRawBalancedDF = AddObsFromHoldoutDB (InputRawDF, config)
    display(InputRawBalancedDF.head())
    
    
    
    
    
    
    
    
    
    
    ## Removing Flag, So That Predict can Run
    os.unlink('FlagRaised_TrainingOccurring_DontRunGlTest')
    
    print('completed main '+str(int(time.time())))
    
def main():
    execute_clust()

"""run every hour""";
if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(ex)

        