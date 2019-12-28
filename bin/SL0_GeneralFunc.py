# General Functions
'''
Description: 
    This file provide some function that are for general use cases.
Function this file Contains:
    - GetBackSomeDirectoryAndGetAbsPath: Used to get absolute path based on the provided relative path.
    - TimeCataloging: To generate a REPORT file for the runtime/crashes of parts of the code.
    - CreateKey: Used to combine columns to return combined value which cann be used as KEY.
    - AddRecommendation: Used to add messages to the recommendation file. When all recommendation are 
                followed delete this file.
    - LevBasedPrint: It is used to print statement based on the level. i.e.  function level.
    - DataFrameScaling: It is used to Scale Features
    - DatasetPrimAnalysis: Used to understand Dataset as in Quantitative and Qualitative section and do a basic analysis of it.
'''

# ----------------------------------------------- Loading Libraries ----------------------------------------------- #
import time, os, sys, json, ast
import pandas as pd


# --------------------------------------- GetBackSomeDirectoryAndGetAbsPath --------------------------------------- #

def GetBackSomeDirectoryAndGetAbsPath(RelPath, msg = False):
    '''
    DirToMoveTo = ../A/B/
    CurrentAbsDir = /X/Y/Z  ##abs path
    
    returns /X/Y/Z, /X/Y/A/B/
    i.e. returns original and new path
    '''
    curr = str(os.getcwd())
    curr0 = curr
    path = RelPath
    DirToGoTo = path.split('/')
    for dirspli in DirToGoTo:
        if dirspli == '..':
            curr = '/'.join(curr.split('/')[0:-1])
        else:
            curr += '/' + dirspli
    if msg is True: print('Current directory where code is executed :', curr0)
    if msg is True: print('New directory path which was mentioned :', curr)
    return curr0, curr
    # ------------------------------------------------------------------------------------------- #



# ------------------------------------------------- TimeCataloging ------------------------------------------------ #

def TimeCataloging(config, Key, Value, First = 'Off'):
    '''
    To generate a REPORT file for the runtime/crashes of parts of the code.
    '''
    if First == 'On':
        ExecTime = time.strftime('%y_%m_%d_%Hhr_%Mmin(%Z)', time.gmtime())
        TimeConsumedReport = {
                'ExecutionTime': ExecTime,
                'ExecTimestamp': int(time.time()),
                'ImportInput': '-',
                'ImportBlKeys': '-',
                'ImportFeedbackData': '-',
                'CombineDataStrems': '-',
                'ComputeSizeOfThisIteration': '-',
                'AdaptiveKeySelection': '-',
                'BlacklistingKeys': '-',
                'UpdatingBlacklistLogs': '-',
                'WholeExecutionTime': '-'
            }
    ## Creating a DataFrame Containing Execution Time Results
    ExecTimePath = config['LogPaths']['ExecutionTimeTaken']
    col = ['ExecutionTime', 'ExecTimestamp', 'ImportInput', 'ImportBlKeys', 'ImportFeedbackData', 'CombineDataStrems', 'ComputeSizeOfThisIteration',
           'AdaptiveKeySelection', 'BlacklistingKeys', 'UpdatingBlacklistLogs', 'WholeExecutionTime']
    if(os.path.exists(ExecTimePath) is False):
        tempDF = pd.DataFrame(TimeConsumedReport, columns = col, index = [0]) #TimeConsumedReport.keys()
    else:
        tempDF = pd.read_csv(ExecTimePath)
        if First == 'On':
            tempDF = tempDF.append(TimeConsumedReport, ignore_index=True)
    ## Updating Entries
    try:
        tempDF.iloc[(len(tempDF)-1), tempDF.columns.get_loc(Key)] = Value
    except:
        print('Passed Key Doesn\'t Exist in Present Structure')
    ## Saving Locally
    tempDF.to_csv(ExecTimePath, index=False)
    if Key == 'WholeExecutionTime':
        return tempDF.iloc[len(tempDF)-1,:].to_dict()
    # ------------------------------------------------------------------------------------------- #



# --------------------------------------------------- CreateKey --------------------------------------------------- #

def CreateKey(DF, Key_ColToUse):
    '''
    Use to combine columns to generate a key which is seperated by '|'
    eg. Key_ColToUse = sid, bin & IP ==> return sid|Bin|IP 
    
    Other way: 
        Convert all columns to be sent to string first
        DF[['Col1','Col2','Col3']].apply(lambda x: ('|').join(x), axis=1)
    '''
    df = DF.copy()
    for col_ind in range(len(Key_ColToUse)):
        I1 = df.index.tolist()
        I2 = df[Key_ColToUse[col_ind]].astype('str').tolist()
        if col_ind == 0:
            df.index = I2
        else:
            df.index = [ "|".join([I1[ind], I2[ind]]) for ind in range(len(I1)) ] #, I3[ind]
    return df.index
    # ------------------------------------------------------------------------------------------- #



# ---------------------------------------------- AddRecommendation ------------------------------------------------ #

def AddRecommendation(msgToAdd, config):
    '''
    Used for adding recommendations inside a single recommendation File
    Delete this file after recommendationhas been followed.
    '''
    filePath = config['LogPaths']['RecommendationFile']
    _, absPathRecommFile = GetBackSomeDirectoryAndGetAbsPath(filePath)
    NewDf = pd.DataFrame({'Recommendation': msgToAdd}, columns=['Recommendation'], index=[0])
    
    if os.path.exists(absPathRecommFile):
        df = pd.read_csv(filePath)
        if msgToAdd not in list(df['Recommendation'].unique()):
            LevBasedPrint('New Recommendation has been added', 0)
            df = pd.concat([df,NewDf], ignore_index=True, sort=False)
        else:
            LevBasedPrint('This recommendation is already present, hence not adding.', 0)
    else:
        LevBasedPrint('First Recommendation has been added', 0)
        df = NewDf.copy()
    df.to_csv(absPathRecommFile, index = False)
    # ------------------------------------------------------------------------------------------- #



# ------------------------------------------------- LevBasedPrint ------------------------------------------------- #

def LevBasedPrint(txt, level=0, StartOrEnd=0):
    '''
    Use to print statement based on levels
    '''
    if StartOrEnd != 0:
        ## expecting '\t' len = 8 spaces
        print('', '+'+'-'*(112 - 8*level),sep= '\t'*level)
    if len(txt) != 0: print('', txt,sep= '\t'*level + '|'+' ')
    # ------------------------------------------------------------------------------------------- #



# ------------------------------------------------ DataFrameScaling ----------------------------------------------- #

def DataFrameScaling(DF, FeatScalerDict, config, FeatureScale_LocID = '-11', Explicit_Scaler = None, Explicit_Task = None):
    '''
    Since A custom variant of Data Scaling is used and this Model is to be preserved, to be used as for predict and also to be used as a way to point conceptual/data drift if it occurs.

    Each time a Scaling is to be done the basic information of the dataset at that pt will be saved with a unique location ID.
    The Dataset properties for Scaling are to be saved/added when Model is to be Trained and just to be read when the predict will be use.
    
    FeatScalerDict = {'Feat1ToBeScaled': 'Standard', 'Feat2ToBeScaled': 'Standard', 'Feat3ToBeScaled': 'Standard'}
    Scaling Settings: 'Normalized', 'Standard', 'Standard_Median', 'Nil'
    FeatureScale_LocID: is used to id the location where which data is to be used
    Explicit_Scaler: if mentioned Use the mentioned Scaler in place of the one mentioned in config file
    Explicit_Task : TrainTest // GlTest
    '''
    
    # -----------<<<  Setting constant values that are to be used inside function  >>>----------- #
    ScalingInformationFile = config['ModelPaths']['ScalingInfoFile'] 
    ConceptDriftFilePath = config['TempPaths']['ConceptualDriftFile']
    ToTransfDF = DF.loc[:, [ feat for feat in FeatScalerDict ] ]
    TransfDF =  DF.loc[:, [ col for col in DF.columns if col not in [ feat for feat in FeatScalerDict ] ] ]
    if Explicit_Scaler is not None:
        for feat in FeatScalerDict:
            FeatScalerDict[feat] = Explicit_Scaler
    Explicit_Task = config['IterationAim']['CycleType'] if Explicit_Task is None else Explicit_Task
    LevBasedPrint('Inside "'+DataFrameScaling.__name__+'" function and configurations for this has been set.',1,1)
    
    # ---------------<<<  Logging / Accessing Feature Stats from the Dataframe  >>>-------------- #
    ## Dataset Information is to be preserved
    if(Explicit_Task == 'TrainTest'):
        
        LevBasedPrint('>>> Saving features stats before scaling the feature. "TrainTest" is used.')
        ScalingFeatureOverallFile = {}
        ## Computing Measures that are used for Scaling
        tempFeatStatsDict = {}
        for col in ToTransfDF.columns:
            tempFeatStatsDict[col] = {'Min': ToTransfDF[col].min(),
                                      'Median': ToTransfDF[col].median(),
                                      'Max': ToTransfDF[col].max(),
                                      'Mean': ToTransfDF[col].mean(),
                                      'Std': ToTransfDF[col].std()
                                     }
        ScalingFeatureOverallFile[FeatureScale_LocID] = tempFeatStatsDict
        InfoForScaling = tempFeatStatsDict
    
        if os.path.exists(ScalingInformationFileName):
            file = open(ScalingInformationFileName, 'r')
            data = json.load(file)
            file.close()
            data[FeatureScale_LocID] = InfoForScaling
        else:
            data = ScalingFeatureOverallFile

        ## Preserving the Information Locally i.e saving model data 
        file = open(ScalingInformationFileName, 'w+')
        DictToWrite = json.dumps(data)
        file.write(DictToWrite)
        file.close()
    
    ## Dataset Information is to be extracted
    elif(Explicit_Task == 'GlTest'):
        
        LevBasedPrint('<<< Extracting features stats before scaling the feature. "GlTest" Setting is used.')
        if os.path.exists(ScalingInformationFileName):
            file = open(ScalingInformationFileName, 'r')
            data = json.load(file)
            file.close()
            if(FeatureScale_LocID in list(data.keys())):
                ## read information for that key
                InfoForScaling = data[FeatureScale_LocID]
            else:
                txt = 'Exception: FeastureScaleLocationID based key that should have been present in GlTest is not present. Try running TrainTest Cycle first.'
                LevBasedPrint(txt, 1)
                AddRecommendation(txt, config)
                raise Exception(txt)
        else:
            txt = 'Exception: "ScalingInfoFile" that should have existed is not present in the storage.'
            LevBasedPrint(txt, 1)
            AddRecommendation(txt, config)
            raise Exception(txt)
    
    ## Incase Task doesn't matches any setting "TrainTest" or "GlTest"
    else:
        txt = 'Exception: "config[\'IterationAim\'][\'CycleType\']" or "Explicit_Task" is containing some undefined value.'
        LevBasedPrint(txt, 1)
        AddRecommendation(txt, config)
        raise Exception(txt)
    
    
    # -------------------------------<<<  Scaling the Feature  >>>------------------------------- #
    BorderTxt = '+'+'-'*100
    LevBasedPrint('Scaling Features', 1)
    LevBasedPrint(str(BorderTxt), 1)
    ## Using the Information received from 'InfoForScaling', Scaling the Dataset
    for feat in featScalDict:
        LevBasedPrint('|\tScaling Feature "{}" using "{}" scaler'.foramt(feat, featScalDict[feat]), 1)
        li = list(ToTransfDF[feat])
        if(Scaler == 'Normalized'):
            TransfDF[feat] = [ (elem - InfoForScaling[feat]['Min']) / (InfoForScaling[feat]['Max'] - InfoForScaling[feat]['Min']) for elem in li ] 
        elif(Scaler == 'Standard'):
            TransfDF[feat] = [ (elem - InfoForScaling[feat]['Mean']) / InfoForScaling[feat]['Std'] for elem in li ] 
        elif(Scaler == 'Standard_Median'):
            TransfDF[feat] = [ (elem - InfoForScaling[feat]['Median']) / InfoForScaling[feat]['Std'] for elem in li ] 
        elif(Scaler == 'Nil'):
            TransfDF[feat] = li
        else:
            LevBasedPrint('This provided scaler "{}" is NOT defined'.format(featScalDict[feat]), 1)
            LevBasedPrint('Support for \'OneHotEncoding\' // \'DummyEncoding\' will be added ')
            continue
    LevBasedPrint(str(BorderTxt), 1)
    
    
    # --------------------------------<<<  Conceptual Drift  >>>--------------------------------- #
    '''
    Currently only with 'normalized'
    Additional Extension: Highlight observation whose values lies outside from that of TrainSet --- mark as outlier (Conceptual Drift)
    '''
    RemovedIndexFromDF, ConcpDftDF = [], None
    if Explicit_Task == 'GlTest':
        IndexOutsideRange = []
        LevBasedPrint('Scaling Features', 1)
        LevBasedPrint(str(BorderTxt), 1)
        for feat in featScalDict:
            if featScalDict[feat] == 'Normalized':
                ValOutsideRange = [ True if((obs < 0)|(obs > 1)) else False for obs in TransfDF[feat] ]
                if sum(ValOutsideRange) > 0: 
                    LevBasedPrint('|\tFeature "{}" contains "{}" Observation Outside the Range'.format(feat, sum(ValOutsideRange)), 1) 
                    IndOutRange = [ ind for ind in range(len(TransfDF[feat])) if((TransfDF[feat][ind] < 0)|(TransfDF[feat][ind] > 1)) ]
                    [ IndexOutsideRange.append(ind) for ind in IndOutRange ]
                    LevBasedPrint('|\tThese index are outside the defined range : '+ str(IndexOutsideRange), 1)
        LevBasedPrint(str(BorderTxt), 1)
     
    # ----------<<<  Seperating Conceptual Drift Observation and Scaled Observation  >>>--------- #
        RemovedIndexFromDF = list(pd.Series(IndexOutsideRange).unique())
        ConcpDftDF = TransfDF.iloc[RemovedIndexFromDF,:].reset_index(drop=True)
        TransfDF = TransfDF.iloc[[ i for i in range(len(TransfDF)) if i not in RemovedIndexFromDF ], :].reset_index(drop=True)
        
        if len(ConcpDftDF) > 0:
            if os.path.exists(ConceptDriftFilePath):
                tempDF = pd.read_csv(ConceptDriftFilePath)
                ConcpDftDF = ConcpDftDF.append(tempDF, ignore_index=True, sort = False)
            if(len(ConcpDftDF.columns) >= 3): ### Here createKey concept cann be used
                I1 = ConcpDftDF[ConcpDftDF.columns[0]].astype('str')
                I2 = ConcpDftDF[ConcpDftDF.columns[1]].astype('str')
                I3 = ConcpDftDF[ConcpDftDF.columns[2]].astype('str')
                ConcpDftDF.index = [ "||".join([I1[ind], I2[ind], I3[ind]]) for ind in range(len(I1)) ]
            elif(len(ConcpDftDF.columns) >= 2):
                I1 = ConcpDftDF[ConcpDftDF.columns[0]].astype('str')
                I2 = ConcpDftDF[ConcpDftDF.columns[1]].astype('str')
                ConcpDftDF.index = [ "||".join([I1[ind], I2[ind]]) for ind in range(len(I1)) ]
            else:
                ConcpDftDF.index = ConcpDftDF[ConcpDftDF.columns[0]].astype('str')
            ConcpDftDF.drop_duplicates(keep = 'first', inplace = True)
            ConcpDftDF.reset_index(drop=True, inplace=True)
            ConcpDftDF.to_csv(ConceptDriftFilePath,index=False)
            LevBasedPrint('Conceptual Drift based observations were present and they have been saved.', 1)
    
    
    # ---------------------------------------<<<  xyz  >>>--------------------------------------- #
    LevBasedPrint('', 1, 1)
    return new_df, ConcpDftDF 
    # ------------------------------------------------------------------------------------------- #



# ---------------------------------------------- DatasetPrimAnalysis ---------------------------------------------- #
def DatasetPrimAnalysis(DF):
    '''
    Function to understand the structure of the dataset
    
    np.isnan(yy).any(), np.isinf(xx).any(), np.isinf(yy).any()
    
    Arithmetic mean is present 
    GM and HM can can also be added 
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>----------- #
    df_explore = DF.copy()
    LevBasedPrint('Inside "'+DatasetPrimAnalysis.__name__+'" function and configurations for this has been set.',1,1)
    
    # -------------------------------<<<  Allowed To Execute  >>>-------------------------------- #
    LevBasedPrint('Overall dataset shape : {}'.format(df_explore.shape), 1)
    
    temp = pd.DataFrame(df_explore.isnull().sum(), columns = ['IsNullSum'])
    temp['dtypes'] = df_explore.dtypes.tolist()
    temp['IsNaSum'] = df_explore.isna().sum().tolist()
    
    temp_cat = temp.loc[temp['dtypes']=='O' ,:]
    if (len(temp_cat) > 0):
        df_cat = df_explore.loc[:,temp_cat.index].fillna('Missing-NA')
        LevBasedPrint('Dataset shape containing Qualitative feature : {}'.format(df_cat.shape), 1)
        temp_cat = temp_cat.join(df_cat.describe().T).fillna('')
        temp_cat['CategoriesName'] = [ list(df_cat[fea].unique()) for fea in temp_cat.index ]
        temp_cat['%Missing'] = [ round((temp_cat['IsNullSum'][i] / max(temp_cat['count']))*100,2) for i in range(len(temp_cat)) ]
        display(temp_cat)
#         print(temp_cat)

    temp_num = temp.loc[((temp['dtypes']=='int') | (temp['dtypes']=='float')),:]
    if (len(temp_num) > 0):
        df_num = df_explore.loc[:,temp_num.index]#.fillna('Missing-NA')
        LevBasedPrint('Dataset shape containing Quantitative feature : {}'.format(df_num.shape), 1)
        temp_num = temp_num.join(df_num.describe().T).fillna('')
        temp_num['%Missing'] = [ round((temp_num['IsNullSum'][i] / max(temp_num['count']))*100,2) for i in range(len(temp_num)) ]
        
        ## Converting float value to readable format
        colsFormatToChange = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        temp_num['count'] = [ int(ele) for ele  in temp_num['count'] ] 
        for col in colsFormatToChange:
            st_li = [ '{0:.10f}'.format(ele) for ele in temp_num[col] ] 
            temp_num[col] = [ st[:st.index('.')+4] for st in st_li ]
        display(temp_num)
#         print(temp_num)
    
    if len(temp)!=len(temp_cat)+len(temp_num):
        LevBasedPrint("Some columns data is missing b/c of data type", 1)
    
    # ------------------------------<<<  returning the result  >>>------------------------------- #
    LevBasedPrint('', 1, 1)
    return temp_cat, temp_num
    # ------------------------------------------------------------------------------------------- #



# --------------------------------------------- DataFrameScaling - V1 --------------------------------------------- #

######################################################### Algorithm to Preserve and Use offline Computer Stats for Data Scaling
def DataFrameScalingV1(dataframe, ColumnToIgnore, configuration, FeatureScale_LocID, Explicit_Scaler = None, Explicit_Task = None):
    '''
    Since A custom variant of Data Scaling is Used and this Model is to be preserved, to be used as for predict and also to be used as a way to point conceptual/data drift if it occurs.

    Each time a Scaling is to be done the basic information of the dataset at that pt will be saved with a unique location ID.
    The Dataset properties for Scaling are to be saved/added when Model is to be Trained and just to be read when the predict will be use.
    
    Scaling Settings: 'Normalized', 'Standard', 'Standard_Median', 'Nil'
    FeatureScale_LocID: is used to id the location where which data is to be used
    Explicit_Scaler: if mentioned Use the mentioned Scaler in place of the one mentioned in config file
    '''

    config = configuration  ## See if copy can be made
    AllFeature = dataframe.columns
    temp_df = dataframe.loc[:,[ col for col in AllFeature if col not in ColumnToIgnore ]].copy()
    new_df =  dataframe.loc[:, ColumnToIgnore].copy()
    
    Scaler = config['DataProcessing_General']['GlobalFeatureScaling']
    if Explicit_Scaler is not None:
        Scaler = Explicit_Scaler
    if Explicit_Task is None:
        Explicit_Task = config['aim']['Task']

    ScalingInformationFileName = config['input']['ModelsSaving_dir'] + config['input']['TrainTestDataScalingInfoFile']
    RemovedIndexFromDF = []
    
    ## Dataset Information is to be preserved
    if(Explicit_Task == 'TrainTest'):
        print('>>> Scaling While Saving Parameters, "' + Scaler + '"  <<<')
        ## Computing Measures used for Scaling
        ScalerOffResultFile = {}
        temp_dict ={}
        for col in temp_df.columns:
            temp_dict[col] = {'Min': temp_df[col].min(),
                              'Median': temp_df[col].median(), 
                              'Max': temp_df[col].max(), 
                              'Mean': temp_df[col].mean(), 
                              'Std': temp_df[col].std()}

        ScalerOffResultFile[FeatureScale_LocID] = temp_dict
        InfoForScaling = temp_dict

        ### Preserving the Information Locally i.e saving model data 
        #if(os.path.exists(ScalingInformationFileName) == True):
        #    # Try reading the key data
        #    file = open(ScalingInformationFileName, 'r')
        #    data = json.load(file)
        #    file.close()
        #    if(FeatureScale_LocID in list(data.keys())):
        #        # overwrite that key information
        #        #ActionItem = 'a'
        #        data[FeatureScale_LocID] = InfoForScaling
        #    else:
        #        # append that information
        #        #ActionItem = 'a'
        #        data[FeatureScale_LocID] = InfoForScaling
        #else:
        #    #do something else
        #    #ActionItem = 'w' # can be 'a' too
        #    data[FeatureScale_LocID] = InfoForScaling
        if(os.path.exists(ScalingInformationFileName) == True):
            file = open(ScalingInformationFileName, 'r')
            data = json.load(file)
            file.close()
            data[FeatureScale_LocID] = InfoForScaling
        else:
            data = ScalerOffResultFile
            

        ## Preserving the Information Locally i.e saving model data 
        file = open(ScalingInformationFileName, 'w+')
        DictToWrite = json.dumps(data)
        file.write(DictToWrite)
        file.close()

    elif(Explicit_Task == 'GlTest'):
        print('<<< Scaling Using Saved Parameters, "' + Scaler + '" >>>') 
        if(os.path.exists(ScalingInformationFileName) == True):
            # Try reading the key data
            file = open(ScalingInformationFileName, 'r')
            data = json.load(file)
            file.close()
            if(FeatureScale_LocID in list(data.keys())):
                ## read information for that key
                InfoForScaling = data[FeatureScale_LocID]
            else:
                print('Key is not present in the saved Scaling model file. Exiting')
                sys.exit(1)
        else:
            print('Scaling Model Saved Data Information file doesn\'t exits. Exiting')
            sys.exit(1)
    else:
        print('config[\'aim\'][\'Task\'] is indefinate. Exiting.')
        sys.exit(1)


    # Variable 'InfoForScaling' has distribution information
    
    if(Scaler == 'Nil'):
        print('No Scaling Done')
        return temp_df
    
    ## Using the Information received from 'InfoForScaling' for Scaling the Dataset
    print('Scaling Feature\n+', '-'*100)
    for col in temp_df.columns:
        print('|\t', col)
        li = list(temp_df[col])
        if(Scaler == 'Normalized'):
            new_df[col] = [ (elem - InfoForScaling[col]['Min']) / (InfoForScaling[col]['Max'] - InfoForScaling[col]['Min']) for elem in li ] 
        elif(Scaler == 'Standard'):
            new_df[col] = [ (elem - InfoForScaling[col]['Mean']) / InfoForScaling[col]['Std'] for elem in li ] 
        elif(Scaler == 'Standard_Median'):
            new_df[col] = [ (elem - InfoForScaling[col]['Median']) / InfoForScaling[col]['Std'] for elem in li ] 
        else:
            print('This Passed Scaler is Not Defined')
            continue
            #print('Scaler to Use Undefined')
    print('+', '-'*100)
    #display(temp_df.head(10))
    #display(new_df.head(10))
    
    ################################################
    ## Additional Extension --- Highlight observation whose values lies outside from that of TrainSet --- mark as outlier (Conceptual Drift)
    
    #### Conceptual Drift ---->  Currently only with 'normalized'
    ConcpDftDF = None
    if((Explicit_Task == 'GlTest') & (Scaler == 'Normalized')):
        IndexOutsideRange = []
        print('Checking Conceptual Drift\n+', '-'*100)
        for col in temp_df.columns:
            ValOutsideRange = [ True if((obs < 0)|(obs > 1)) else False for obs in new_df[col] ]
            print('|\tFeature "'+col+'" contains '+ str(sum(ValOutsideRange)) + ' Observation Outside the Range') 
            IndOutRange = [ ind for ind in range(len(new_df[col])) if((new_df[col][ind] < 0)|(new_df[col][ind] > 1)) ]
            [ IndexOutsideRange.append(ind) for ind in IndOutRange ]
        print('|\tIndex Which Are Outside The Defined Range :', IndexOutsideRange, '\n+', '-'*100)

        RemovedIndexFromDF = IndexOutsideRange
        ConcpDftDF = new_df.iloc[IndexOutsideRange,:].reset_index(drop=True)
        new_df = new_df.iloc[[ i for i in range(len(new_df)) if i not in IndexOutsideRange ], :].reset_index(drop=True)
        
        if len(ConcpDftDF) > 0:
            ConceptDriftFilePath = config['input']['ConceptualDriftDatabase']
            # if os.path.exists(ConceptDriftFilePath):   ### Present in Data Pre Processing
            #     os.remove(ConceptDriftFilePath)  # To be used only once when the code is Run for the first time 

            if os.path.exists(ConceptDriftFilePath):
                tempDF = pd.read_csv(ConceptDriftFilePath)
                ConcpDftDF = ConcpDftDF.append(tempDF, ignore_index=True, sort = False)#.sample(frac =1)#.reset_index(drop =True)

            ### With Index as in geral drop_duplicate doesn't seems to work 
            # ConcpDftDF['SID'] = ConcpDftDF['SID'].astype(str)
            # ConcpDftDF.index = ConcpDftDF[['SID', 'BinsBackFromCurrent', 'apidata__zpsbd6']].apply(lambda x: ('|').join(x), axis=1)
            if(len(ConcpDftDF.columns) >= 3):
                I1 = ConcpDftDF[ConcpDftDF.columns[0]].astype('str')
                I2 = ConcpDftDF[ConcpDftDF.columns[1]].astype('str')
                I3 = ConcpDftDF[ConcpDftDF.columns[2]].astype('str')
                ConcpDftDF.index = [ "||".join([I1[ind], I2[ind], I3[ind]]) for ind in range(len(I1)) ]
            elif(len(ConcpDftDF.columns) >= 2):
                I1 = ConcpDftDF[ConcpDftDF.columns[0]].astype('str')
                I2 = ConcpDftDF[ConcpDftDF.columns[1]].astype('str')
                ConcpDftDF.index = [ "||".join([I1[ind], I2[ind]]) for ind in range(len(I1)) ]
            else:
                ConcpDftDF.index = ConcpDftDF[ConcpDftDF.columns[0]].astype('str')
            # ConcpDftDF.drop_duplicates(subset = FeatureToIgnore, keep = 'first', inplace = True)#.reset_index(drop=True)
            ConcpDftDF.drop_duplicates(keep = 'first', inplace = True)#.reset_index(drop=True)
            ConcpDftDF.reset_index(drop=True, inplace=True)
            ConcpDftDF.to_csv(ConceptDriftFilePath,index=False)
            ## ConcpDftDF.groupby(by= FeatureToIgnore).first()
            print('Conceptual Drift based observations were present and they have been saved.')
            # print(ConcpDftDF.shape)
            # display(ConcpDftDF)#.head()
    
    return new_df, ConcpDftDF #, RemovedIndexFromDF

# def DataFrameNormalization(temp_df):
#         return (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())
    # ------------------------------------------------------------------------------------------- #



# ----------------------------------------------- GeneralStats - V1 ----------------------------------------------- #
from scipy.stats import mstats
from sklearn.preprocessing import minmax_scale
def GeneralStats(df_series):
    '''
    This Function uses 'pandas series' to compute and print various statistics on the series
    # https://docs.scipy.org/doc/scipy-0.13.0/reference/stats.mstats.html
    '''
    print("\nGeneral Statistics")
    series = minmax_scale(df_series, feature_range=(1e-10, 1))
    print("Zscore per point", [mstats.zmap(i, series) for i in series] [0:4] + ["....."])
    print("Zscore series", mstats.zscore(series)[0:4] )
    print("Describing Series", mstats.describe(series) )
    print("Trimmed Min", mstats.tmin(series) )
    print("Trimmed Max", mstats.tmax(series) )
    print("Geometric Mean", mstats.gmean(series) )
    print("Harmonic Mean", mstats.hmean(series) )
    # ------------------------------------------------------------------------------------------- #



# ----------------------------------------------------------------------------------------------------------------- #