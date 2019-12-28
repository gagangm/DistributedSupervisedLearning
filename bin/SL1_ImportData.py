
# Importing Data
'''
Description: 
    This file provide some function that are toe used for importing data .
Function this file Contains:
    - ImportData: Used to import data either from BQ or from Storage.
'''

# ----------------------------------------------- Loading Libraries ----------------------------------------------- #
import pandas as pd
import glob, os, ast, time
from datetime import datetime, date, timedelta
from SL0_GeneralFunc import LevBasedPrint, AddRecommendation


# ------------------------------------------ GrabAnySizeDatafromGoogleBQ ------------------------------------------ #
def Exec_BQ(query, projectid):
    LevBasedPrint('Inside "'+Exec_BQ.__name__+'" function.',3,1)
    LevBasedPrint('',3,1)
    return pd.io.gbq.read_gbq(query, project_id=projectid, index_col=None, col_order=None, reauth=False, private_key=None) #, verbose=True deprecated


def GenerateTableNames(config):
    '''
    Make use of Domain based parameters to get the data.
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>----------- #
    DatasetName = config['BigQueryConfig']['DatasetName']
    SIDs = ast.literal_eval(config['DomainConfig']['SIDs'])
    DataGrabMethodology = config['DomainConfig']['UseStaticOrDynamicCurrentDay']
    LevBasedPrint('Inside "'+GenerateTableNames.__name__+'" function and configurations for this has been set.',3,1)
    LevBasedPrint('Data collection methodology that has been selected : ' + str(DataGrabMethodology),3)
    if DataGrabMethodology == 'static':
        Dates = ast.literal_eval(config['IfStatic']['Date']) 
        StaDataWindow = ast.literal_eval(config['IfStatic']['DataGrabWindow_Days'])
    elif DataGrabMethodology == 'dynamic':
        DynDataWindow = int(ast.literal_eval(config['IfDynamic']['DataGrabWindow_Hr']))
    else:
        txt = 'Exception: Wrong Configuration has been passed in "UseStaticOrDynamicCurrentDay".'
        AddRecommendation(txt, config)
        raise Exception(txt)
    
    # -----------------------------<<<  Generating Table Names  >>>------------------------------ #
    ## Generating Table Names
    if DataGrabMethodology == 'static':
        if StaDataWindow != '-':
            CustomDate = date(2000 + int(Dates[0][4:6]), int(Dates[0][2:4]), int(Dates[0][0:2])) 
            format = '%d%m%y'
            Dates = [ (CustomDate + timedelta(days=i)).strftime(format) for i in range(int(StaDataWindow)) ]
        TableToInclude = ''
        for i in range(len(SIDs)):
            for j in range(len(Dates)):
                TableToInclude += '\n\tTABLE_QUERY([{}.Citadel_Stream],\'table_id like "'.format(DatasetName) + SIDs[i] + '_' + Dates[j] + '_%"\'),'
    elif DataGrabMethodology == 'dynamic':
        CurrentTime = datetime(time.gmtime().tm_year, time.gmtime().tm_mon, time.gmtime().tm_mday, time.gmtime().tm_hour, time.gmtime().tm_min, time.gmtime().tm_sec) ## UTC        
        TableDateToTake = []
        while DynDataWindow >= -1:  ## -1 to even include the current hour table
            tempDate = CurrentTime - timedelta(days = 0, hours = DynDataWindow, minutes = 0)
            TableDateToTake.append(tempDate.strftime(format = '%d%m%y_%H'))
            DynDataWindow -= 1
        TableToInclude, TableCnt = '', 0
        for i in range(len(SIDs)):
            for j in range(len(TableDateToTake)):
                TableCnt += 0
                TableToInclude += '\n\tTABLE_QUERY([{}.Citadel_Stream],\'table_id like "'.format(DatasetName) + SIDs[i] + '_' + TableDateToTake[j] + '%"\'),'
        LevBasedPrint('Total number of tables accessed : '+str(TableCnt),3)
    # ---------------------------------------<<<  xyz  >>>--------------------------------------- #
    LevBasedPrint('',3,1)
    return TableToInclude
    # ------------------------------------------------------------------------------------------- #


def GrabAnySizeDatafromGoogleBQ(config):
    '''
    Incase if dataset size is too large then this function will enable the extraction of whole dataset by getting the data in chunks
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>----------- #
    ModuleSetting = config['Config']['ModuleSettingRuleName']
    BQ_Cred = config['BigQueryConfig']['ProjectID']
    if ModuleSetting == 'ICLSSTA': BinSizeBasedOnPeriod_Hr = int(config['Config']['ICLSSTA_BinSizeBasedOnPeriod_Hr'])
    BQ_QueryFile = config['InputPaths']['BQ_DataImportQuery']
    LimitToStartWith = config['BigQueryConfig']['BQ_LimitToStart']
    LimitDecreaseFactor = float(config['BigQueryConfig']['BQ_LimitDecreaseFactor'])
    LevBasedPrint('Inside "'+GrabAnySizeDatafromGoogleBQ.__name__+'" function and configurations for this has been set.',2,1)
    
    # -------------------------<<<  Generating Tables Name To Query  >>>------------------------- #
    TableToInclude = GenerateTableNames(config)
    #print(TableToInclude)
    
    # -------------------------<<<  Creating Bin Setting For ICLSSTA  >>>------------------------ #
    ## Getting the string that will be used to create bins for grouping based on a certain TimePeriod
    GroupsToInclude = ''
    if ModuleSetting == 'ICLSSTA':
        for i in range(1000): ##even if the bin size is as small as an hour, BQ has a limitation of accessing upto a max of 1000 Table, so this is the max possible limit 
            ll_insec = int(i*BinSizeBasedOnPeriod_Hr *3600)
            ul_insec = int((i+1)*BinSizeBasedOnPeriod_Hr *3600 - 1)
            GroupsToInclude += '\n\tWHEN (CurrentTimeStamp - CurrentHitTimeStamp) BETWEEN {low} AND {upp} THEN "Bin_{WhichBin}"'.format(low= ll_insec,upp= ul_insec, WhichBin= i)
    
    # ------------------------<<<  Reading Query From External File  >>>------------------------- #
    LevBasedPrint('Read from a locally saved Query File', 2)
    queryfile = open(BQ_QueryFile, 'r')
    query = queryfile.read()
    queryfile.close()
    
    # --------------------<<<  Importing Data in Max possible batch size  >>>-------------------- #
    ## looping over the limit and offset to grab the maximum possible bite in terms of observation that can be gathered
    ## GP
    start = int(LimitToStartWith)  # should be equal to the maximum number of observation that you want to extract
    ratio = 1/LimitDecreaseFactor
    limit = 1000  ## util which pt to try to gather the data ## Hardcoded
    length = 1000
    # query='''SELECT 1 limit {lim} offset {off}'''
    
    DF = pd.DataFrame()
    ##GP
    for i in [ int(start * ratio ** (n - 1)) for n in range(1, length + 1) if start * ratio ** (n - 1) > limit ]:
        if DF.shape == (0, 0):
            try:
                offcurr = 0
                while offcurr < start:
                    LevBasedPrint('Setting used in extracting data from BQ:\tNo. of obs. extracted per cycle (limit) = ' + str(i) + '\tOffset = ' + str(offcurr),2)
                    QueryToUse = query.format(BinToUse = GroupsToInclude, TableToInclude = TableToInclude, lim = str(i), off = str(offcurr))
                    tempDF = Exec_BQ(QueryToUse, BQ_Cred)
                    DF = DF.append(tempDF, ignore_index = True)
                    offcurr += i

            except Exception as error:
                txt = 'Exception: In importing data from BQ was thrown!\nLimit used: ' + str(i) + '\n' + str(error)
                LevBasedPrint(txt, 2)
                AddRecommendation(txt, config)
                # raise Exception(txt)
    
    # ---------------------------------------<<<  xyz  >>>--------------------------------------- #
    LevBasedPrint('',2,1)
    return DF
    # ------------------------------------------------------------------------------------------- #


# -------------------------------------------------- ImportData --------------------------------------------------- #
def ImportData(config):
    """
    Can be used to import data from either storage or BQ
    
    
    Extracts any size data from any SID of any number of days.
    
    Works in Two Configuration(config['IterationAim']['CycleType']), namely 'TrainTest' & 'GlTest'
    'TrainTest' is for models training purpose where This Dataset is split later too make dataset size adequate for training uing sampling
    'GlTest' is purely for prediction purpose, i.e. it will be used as testset only and will consume saved model to provide labels to observations
    """
    # -----------<<<  Setting constant values that are to be used inside function  >>>----------- #
    AccessDataFrom = config['DataCollection']['GetDataFrom']
    if AccessDataFrom == 'BQ':
        SettingToUse = config['IterationAim']['CycleType']
        if SettingToUse: GlTestDataSize = int(config['IterationAim']['GlTest_DataGrabWindow_Hr'])
        FileLocalSavingName = config['InputPaths']['BQ_RawDataStoringName'].format(SettingToUse)
        GetNewCopy = config['DomainConfig']['BQ_GetNewCopyOfData']
    elif AccessDataFrom ==  'Storage':
        FileName = config['InputPaths']['Storage_RawData']
    else:
        print('Wrong setting in "GetDataFrom", current value is {}'.format(AccessDataFrom))
        txt = 'Exception: Wrong Configuration has been passed in "GetDataFrom".'
        AddRecommendation(txt, config)
        raise Exception(txt)
    LevBasedPrint('Inside "'+ImportData.__name__+'" function and configurations for this has been set.',1,1)
    
    
    LevBasedPrint('Accessing data from {}'.format(AccessDataFrom), 1)
    # ----------------------------<<<  Accessing Data from BQ  >>>------------------------------- #
    if AccessDataFrom == 'BQ':
        
        # -----------------------<<<  Setting Configuration for GlTest  >>>-------------------------- #
        if(SettingToUse == 'GlTest'):
            config['IfStatic']['DataGrabWindow_Days'] = str(int(GlTestDataSize/24 + 1))
            config['IfDynamic']['DataGrabWindow_Hr'] = str(GlTestDataSize + 1)

        # --------------------------<<<  Get New Copy Of Data Or Reuse  >>>-------------------------- #
        if (os.path.exists(FileLocalSavingName) == False) | (GetNewCopy in ['True', 'true', 'T', 't', 'Yes', 'yes', 'Y', 'y']):
            DF = GrabAnySizeDatafromGoogleBQ(config)
            # if(SettingToUse == 'GlTest'):
            #     DF.drop(DF[DF.BinsBackFromCurrent != 'Bin_0'].index, inplace=True)
            #     DF.reset_index(drop=True, inplace=True)
            DF.to_csv(FileLocalSavingName, index=False)#, sep='|', encoding='utf-8')
            LevBasedPrint('Data extracted from BQ and saved locally to the File: '+ FileLocalSavingName, 1)
        else:
            DF = pd.read_csv(FileLocalSavingName)#, sep='|', encoding='utf-8')
            LevBasedPrint('Data Loaded From the File: '+ FileLocalSavingName, 1)
        LevBasedPrint('Data Shape: '+str(DF.shape), 1 )
    # --------------------------<<<  Accessing Data from Storage  >>>---------------------------- #
    elif AccessDataFrom == 'Storage':
        DF = pd.read_csv(FileName)#, sep='|', encoding='utf-8')
        LevBasedPrint('Data Loaded From the File: '+ FileName, 1)
    
    # ---------------------------------------<<<  xyz  >>>--------------------------------------- #
    LevBasedPrint('Data Import | Complete',1)
    LevBasedPrint('',1,1)
    return DF
    # ------------------------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------------------------------------------- #
## AP
# start = int(LimitToStartWith)  # should be equal to the maximum number of observation that you want to extract
# stop = -1
# step = -int(start/LimitDecreaseFactor)
# limit = int(start/LimitDecreaseFactor)  ## util which pt to try to gather the data
##AP
# for i in [i for i in range(start,stop, step) if i >= limit]: