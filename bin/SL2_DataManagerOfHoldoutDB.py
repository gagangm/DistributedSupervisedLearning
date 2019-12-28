
# Holdout DB
'''
Description: 
    This file provide some function that revolves around preserving and ussing critical class Observations.
Function this file Contains:
    - GenerateHoldoutDB: Used to gather critical class observation from InputDF and preserve it in HoldoutDB.
    - AddObsFromHoldoutDB: Used to get observation from HoldoutDB and mmix these with InputDF.
'''

# ----------------------------------------------- Loading Libraries ----------------------------------------------- #
import os, sys, time, ast
import pandas as pd
from SL0_GeneralFunc import LevBasedPrint, AddRecommendation, CreateKey

# ----------------------------------------------- GenerateHoldoutDB ----------------------------------------------- #

def GenerateHoldoutDB(DF, SeriesCriticalClass, config):
    '''
    Storing critical class Observations to a DB
    SeriesCriticalClass: containg True label for the observations from which randomlly observation needs to be saved.
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>----------- #
    DF = DF.copy()
    CycleType = config['IterationAim']['CycleType']
    RunInTrain = ast.literal_eval(config['CreateHoldoutDB']['EnableInTrainCycle'])
    RunInPredict = ast.literal_eval(config['CreateHoldoutDB']['EnableInPredictCycle'])
    ModuleName = config['Config']['ModuleSettingRuleName']
    KeyFormat = ast.literal_eval(config['DataProcessing_General']['KeyFormat'])
    HoldoutDBSavingLoc = config['InputPaths']['CriticalClassHoldoutDB']
    HDB_SigToPPreserve = float(config['CreateHoldoutDB']['FracOrCntCritClassSigToPreservePerIteration'])
    HDB_UpperLimitOnDbSize = int(config['CreateHoldoutDB']['MaxObsThatCanBeKeptInDB'])
    FeatureProcess_Dict = ast.literal_eval(config['DataProcessing_General']['FeaturesProcessing'])
    FeatureToIgnore = [ key for key in FeatureProcess_Dict.keys() if FeatureProcess_Dict[key]['Usage'] == 'Identification' ]
    LevBasedPrint('Inside "'+GenerateHoldoutDB.__name__+'" function and configurations for this has been set.',1,1)
    
    # -------------------------------<<<  Allowed To Execute  >>>-------------------------------- #
    if ((CycleType == 'TrainTest') & RunInTrain):
        msg = 'Alllowed to Run in "Train" Cycle'
    elif ((CycleType == 'GlTest') & RunInPredict):
        msg = 'Alllowed to Run in "Predict" Cycle'
    else:
        msg = 'Not allowed to run in this "{}" cycle'.format(CycleType)
        LevBasedPrint(msg, 1)
        return pd.DataFrame()
    
    # ----------------<<<  Randomly selecting the critical class observation  >>>---------------- #
    if HDB_SigToPPreserve == 0: 
        msg = 'Configuration has been set to NOT make use of this functionality. Hence skipping this step.'
        LevBasedPrint(msg, 1)
        return pd.DataFrame()
    if HDB_SigToPPreserve <= 1:
        tempSampleDF = DF.loc[SeriesCriticalClass,:].sample(frac = HDB_SigToPPreserve).reset_index(drop=True)
    else:
        tempSampleDF = DF.loc[SeriesCriticalClass,:].sample(n = HDB_SigToPPreserve).reset_index(drop=True)
    
    # -----------------------------<<<  Reading Already Saved DB  >>>---------------------------- #
    SampleDF = pd.read_csv(HoldoutDBSavingLoc) if os.path.exists(HoldoutDBSavingLoc) else pd.DataFrame(columns = tempSampleDF.columns)
    
    # ---------------------------<<<  Appending Observations to DB  >>>-------------------------- #
    SampleDF = SampleDF.append(tempSampleDF, ignore_index=True, sort = False).sample(frac =1).reset_index(drop =True)
    
    # ---------------------<<<  Removing Duplicate Observations From DB  >>>--------------------- #
    if ModuleName == 'ICLSSTA':
        SampleDF['BinsBackFromCurrent'] = 'Bin_XX'
        ## with index as general drop_duplicate doesn't seems to work
        SampleDF.index = CreateKey(SampleDF, KeyFormat)
        SampleDF = SampleDF.drop_duplicates(subset = FeatureToIgnore, keep = 'first')
        
        # ## Deleting based on Timestamp
        # if 'RecentHit_TimeStamp' in SampleDF.columns:
        #     DuplicateObsIndex = DF.loc[DF.duplicated(KeyFormat+['RecentHit_TimeStamp'], keep='first'), :].index
        #     DF.drop(index=DuplicateObsIndex, inplace=True, axis=0)
        # ##  OtherWay: DF.drop_duplicates(subset=['Identifier_KEY', 'LastPresenceTimestamp'], inplace=True)
        
        SampleDF.reset_index(drop=True, inplace=True)
    
    # ------------------------------<<<  Limiting DB upper size  >>>----------------------------- #
    if len(SampleDF) >= HDB_UpperLimitOnDbSize:
        SampleDF = SampleDF.sample(n = HDB_UpperLimitOnDbSize).reset_index(drop=True)
    
    # -----------------------------<<<  Saving the Modified DB  >>>------------------------------ #
    SampleDF.to_csv(HoldoutDBSavingLoc, index = False)
    
    # ---------------------------------------<<<  xyz  >>>--------------------------------------- #
    LevBasedPrint('Adding Observation to Holdout DB | Complete',1)
    LevBasedPrint('',1,1)
    return SampleDF
    # ------------------------------------------------------------------------------------------- #


# ---------------------------------------------- AddObsFromHoldoutDB ---------------------------------------------- #
def AddObsFromHoldoutDB (InputDF, config):
    '''
    Using some logic access a certain number of observation from the holdout DB
    '''
    # -----------<<<  Setting constant values that are to be used inside function  >>>----------- #
    DF = InputDF.copy()
    CurrTimestamp = float(time.time())
    CycleType = config['IterationAim']['CycleType']
    RunInTrain = ast.literal_eval(config['AddingObsFromHoldoutDB']['EnableInTrainCycle'])
    RunInPredict = ast.literal_eval(config['AddingObsFromHoldoutDB']['EnableInPredictCycle'])
    ModuleName = config['Config']['ModuleSettingRuleName']
    HDB_SavedLoc = config['InputPaths']['CriticalClassHoldoutDB']
    ObsMixFromHDB = ast.literal_eval(config['AddingObsFromHoldoutDB']['AppendingMethodology'])
    ObsMixFromHDB_Setting = ObsMixFromHDB['Methodology']
    ObsMixFromHDB_Value = float(ObsMixFromHDB['Value'])
    FeatureProcess_Dict = ast.literal_eval(config['DataProcessing_General']['FeaturesProcessing'])
    AllFeatures = [ key for key in FeatureProcess_Dict.keys() ]
    if ModuleName == 'ICLSSTA': TimeDiffToConsider_Hr = float(config['Config']['ICLSSSTA_ObsFromHoldoutDBToBeOlderThanToMixed_Hr'])
    LevBasedPrint('Inside "'+AddObsFromHoldoutDB.__name__+'" function and configurations for this has been set.',1,1)
    
    # -------------------------------<<<  Allowed To Execute  >>>-------------------------------- #
    if ((CycleType == 'TrainTest') & RunInTrain):
        msg = 'Allowed to Run in "Train" Cycle'
    elif ((CycleType == 'GlTest') & RunInPredict):
        msg = 'Alllowed to Run in "Predict" Cycle'
    else:
        msg = 'Not allowed to run in this "{}" cycle'.format(CycleType)
        LevBasedPrint(msg, 1)
        return DF
    
    # ---------------------------------------<<<  xyz  >>>--------------------------------------- #
    if ObsMixFromHDB_Value == 0:
        msg = 'Configuration has been set to NOT make use of this functionality. Hence skipping this step.'
        LevBasedPrint(msg, 1)
        return DF
    
    # --------------------------------<<<  Loading Holdout DB  >>>------------------------------- #
    if os.path.exists(HDB_SavedLoc):
        HDB_DF = pd.read_csv(HDB_SavedLoc)#, sep='|', encoding='utf-8')
    else:
        txt = 'HoldoutDB doesn\'t exist. Hene no observation is added to InputDF'
        LevBasedPrint('Recommendation: '+ txt, 1)
        AddRecommendation(txt, config)
        return DF
    
    # ----------<<<  Checking consistency in columns in the Input DF and HoldoutDB  >>>---------- #
    if False in [ True for colI in DF.columns if colI in HDB_DF.columns ]:
        txt = 'Data structure of the InputDF and HoldoutDB is inconsistent. Hene returning InputDF  without addding any observation.'
        LevBasedPrint('Recommendation: '+ txt, 1)
        AddRecommendation(txt, config)
        return DF
    
    # --------------------<<<  Filtering Observation that can be selected  >>>------------------- #
    if ModuleName == 'ICLSSTA':
        ## Selecting Those Signature That are Older than the provided Time
        NotRecentObsIndex = [ True if(CurrTimestamp - ele)/(60*60) > TimeDiffToConsider_Hr else False for ele in HDB_DF['RecentHit_TimeStamp'] ]        
        HDB_DF = HDB_DF[NotRecentObsIndex].reset_index(drop =True)
    LevBasedPrint('"{}" Observations are available in HoldoutDB after filtering.'.format(len(HDB_DF)), 1)
    if len(HDB_DF) == 0:
        txt = 'There are no observation in HoldoutDB after filtering. Hene no observation is added to InputDF'
        LevBasedPrint('Recommendation: '+ txt, 1)
        AddRecommendation(txt, config)
        return DF
    
    # -------------------<<<  Getting the number of observation to extract  >>>------------------ #
    if ObsMixFromHDB_Setting == 'ObsFromHoldoutDB': ## should be count
        ObsThatAreRequestedToBeAdded = int(ObsMixFromHDB_Value) if ObsMixFromHDB_Value >= 1 else 'ValueError'
    elif ObsMixFromHDB_Setting == 'FracObsToTotalObsInIteration': ## should be a fraction
        ObsThatAreRequestedToBeAdded = int(len(DF) * ObsMixFromHDB_Value) if ObsMixFromHDB_Value <= 1 else 'ValueError'
    else:
        txt = 'Exception: Wrong Configuration has been passed in "AppendingMethodology -- Methodology".'
        LevBasedPrint('Recommendation: '+ txt, 1)
        AddRecommendation(txt, config)
        raise Exception(txt)
    if ObsThatAreRequestedToBeAdded == 'ValueError': 
        txt = 'Exception: Wrong Configuration has been passed in "AppendingMethodology -- Value".'
        LevBasedPrint('Recommendation: '+ txt, 1)
        AddRecommendation(txt, config)
        raise Exception(txt)
    
    if ObsThatAreRequestedToBeAdded > len(HDB_DF):
        ObsThatWillBeAdded  = len(HDB_DF)
        txt = 'Number of observation that are requested to be added is greater than the number of observations that are avalable in holdout DB hence setting it to available observation in Holdout DB.'
        LevBasedPrint('Recommendation: '+ txt, 1)
        AddRecommendation(txt, config)
        
        LevBasedPrint('Observation in InputDF: {}'.format(len(DF)), 1)
        LevBasedPrint('Observation that were requested to be added from HoldoutDB: {}'.format(ObsThatAreRequestedToBeAdded), 1)
        LevBasedPrint('Observation that will be finally added to InputDF: {}'.format(ObsThatWillBeAdded), 1)
    else:
        ObsThatWillBeAdded = ObsThatAreRequestedToBeAdded
        LevBasedPrint('Observation in InputDF: {}'.format(len(DF)), 1)
        LevBasedPrint('Observation that are requested and will finally added to InputDF: {}'.format(ObsThatWillBeAdded), 1)
    
    # -------------------<<<  Getting the Observation that are to be added  >>>------------------ #
    ObsToAdd = HDB_DF.sample(n=ObsThatWillBeAdded).reset_index(drop =True)
    
    # ------------------<<<  Appending and mixing observation to InputDF  >>>-------------------- #
    DF = DF.append(ObsToAdd, ignore_index=True, sort =False).sample(frac =1).reset_index(drop =True)
    
    # ------------------------------<<<  returning the result  >>>------------------------------- #
    LevBasedPrint('Input dataframe shape changed from {} --> {}'.format(str(InputDF.shape), str(DF.shape)), 1)
    LevBasedPrint('Getting Observation from Holdout DB and Mixing in InputDF | Complete', 1)
    LevBasedPrint('',1,1)
    return DF
    # ------------------------------------------------------------------------------------------- #


# ----------------------------------------------------------------------------------------------------------------- #