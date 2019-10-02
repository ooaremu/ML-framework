'''
Created on Oct 14, 2017
used to create figures for RML paper
@author: aremu
'''
import sys
sys.path.append('./')
import pandas as pd
import scipy.stats as scy_stats
import numpy as np
from sklearn.preprocessing import scale
#from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#import math as mth
import matplotlib.pyplot as plt
from sklearn.preprocessing.data import normalize, minmax_scale, StandardScaler,\
    RobustScaler
import os
#from scipy import reciprocal
from IPython.extensions.autoreload import update_class
from matplotlib.pyplot import axis
import itertools
from root.nested.ExpControl import ExpControl 
from root.nested.ExpFeatSelc import  ExpFeatSelc
from root.nested.ExpDimRed import ExpDimRed
from root.nested.GenDimRed import GenDimRed
from scipy import stats, integrate
import seaborn as sns
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from mpl_toolkits.mplot3d import Axes3D
sns.set(color_codes=True)
# -----------------------------------------------------------------------#
FD0002 ='~/Research/Cpp_PHM/CMAPSSData/train_FD002.txt' #FD0002- 6 conditions/fault mode 1 (HPC Degradation)
FD0004 ='~/Research/Cpp_PHM/CMAPSSData/train_FD004.txt' #FD0004- 6 conditions/Fault mode 1 and Fault mode 2 (HPC Degradation, Fan Degradation)

#different condtions persent
FD0001 ='~/Research/Cpp_PHM/CMAPSSData/train_FD001.txt' #FD0001- 1 conditions/fault mode 1 (HPC Degradation)
FD0003 ='~/Research/Cpp_PHM/CMAPSSData/train_FD003.txt' #FD0003- 1 conditions/Fault mode 1 and Fault mode 2 (HPC Degradation, Fan Degradation)
# -----------------------------------------------------------------------#
## Class setup
Control = ExpControl('ControlEvalRML')
DimRedct_control=ExpDimRed('ControlEvalRML')
DimReduct_Eval = ExpDimRed('DimReduct_RML_DimReductEval')
########################START: General Control Evaluation########################################

# import data
#6 conditions
OD = Control.dset_import(dataset_1=FD0002,fault_1=1)
OD2 = Control.dset_import(dataset_1=FD0004,fault_1=2)

# pre-processing
OD_prp,OD_SnRWgts = Control.sys_preprocess(OD)
OD2_prp,OD2_SnRWgts = Control.sys_preprocess(OD2)

# index condition data
OD_cond_idxd = ExpControl('format').sep_per_OPcond(OD)
#OD_cond_idxd.to_csv('/Users/aremu/Research/Cpp_PHM/Dimension_reduction/wolfram/OD1_wcond.csv', sep='\t', encoding='utf-8')
OD2_cond_idxd = ExpControl('format').sep_per_OPcond(OD2)
#OD2_cond_idxd.to_csv('/Users/aremu/Research/Cpp_PHM/Dimension_reduction/wolfram/OD2_wcond.csv', sep='\t', encoding='utf-8')

########################START: Dim Reductionn########################################
#data with 1 faults _ 6 condtion
OD_SpcEAll = ExpDimRed('RML_SpcE').SpcE(OD_prp,plot='yes')#,saveto='RML/1fault');
#OD_SpcEAll.to_pickle('/Users/aremu/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_RML_train_FD002')
plt.close('all')
#data with 2 faults _ 6 condtion
OD2_SpcEAll = ExpDimRed('RML_SpcE').SpcE(OD2_prp,plot='yes')#,saveto='RML/2fault');
#OD2_SpcEAll.to_pickle('/Users/aremu/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_RML_train_FD004')

plt.close('all')
##### or #######
# OD_SpcEAll = ExpDimRed('RML_ISO').IsoMap(OD_prp,plot='yes')#,saveto='RML/1fault');
# plt.close('all')
# #data with 2 faults _ 6 condtion
# OD2_SpcEAll = ExpDimRed('RML_ISO').IsoMap(OD2_prp,plot='yes')#,saveto='RML/2fault');


# General dim reduct method
OD_IsoMap_gen,errorIsoMap_gen = GenDimRed('General_IsoMap').IsoMap(OD_prp,plot='yes')#,saveto='RML/1general');
OD_LLE_gen,errorLLE_gen = GenDimRed('General_LLE').LLE(OD_prp,plot='yes')#,saveto='RML/1general');
OD_SpcE_gen = GenDimRed('General_SpcE').SpcE(OD_prp,plot='yes')#,saveto='RML/1general');
plt.close('all')

#General dim reduct method (data with 2 faults)
OD2_IsoMap_gen,error2IsoMap_gen = GenDimRed('General_IsoMap').IsoMap(OD2_prp,plot='yes')#,saveto='RML/2faults_general');
OD2_LLE_gen,error2LLE_gen = GenDimRed('General_LLE').LLE(OD2_prp,plot='yes')#,saveto='RML/2faults_general');
OD2_SpcE_gen = GenDimRed('General_SpcE').SpcE(OD2_prp,plot='yes')#,saveto='RML/2faults_general');
plt.close('all')
########################################################################################
#6 conditions 1 faults
# all_dsetsDimReduct_Eval = {"Original Data":OD_cond_idxd,"RML":OD_SpcEAll,"Isomap":OD_IsoMap_gen,
#                            "LLE":OD_LLE_gen,"LEM":OD_SpcE_gen}
# 
# # all_dsetsDimReduct_Eval = {"Original Data":OD_cond_idxd,"Original Data2":OD2_cond_idxd,"Original Data3":OD_cond_idxd,
# #                            "Original Data5":OD2_cond_idxd,"Original Data4":OD_cond_idxd}
# # Clssfy_analysis_results_all_FailNom_DimRedct = DimReduct_Eval.fail_classify(all_dsetsDimReduct_Eval,plot=True)
# #all_dsetsDimReduct_Eval = {"Original Data Set":OD_cond_idxd}
# #all_dsetsDimReduct_Eval = {"RML":OD_SpcEAll}
# 
# 
# #6 conditions 2 faults
# all_dsetsDimReduct_Eval_OD2 = {"Original Data":OD2_cond_idxd,"RML":OD2_SpcEAll,"Isomap":OD2_IsoMap_gen,
#                                "LLE":OD2_LLE_gen,"LEM":OD2_SpcE_gen}

# all_dsetsDimReduct_Eval_OD2 = {"Original Data":OD2_cond_idxd}
# all_dsetsDimReduct_Eval_OD2 = {"RML":OD2_SpcEAll}

# 
# #General Data 1 faults
# all_dsetsDimReduct_Eval_gen = {"Isomap":OD_IsoMap_gen}
# all_dsetsDimReduct_Eval_gen = {"LLE":OD_LLE_gen}
# all_dsetsDimReduct_Eval_gen = {"LEM":OD_SpcE_gen}
# 
# #General Data 2 faults
# all_dsetsDimReduct_Eval_gen = {"Isomap":OD2_IsoMap_gen}
# all_dsetsDimReduct_Eval_gen = {"LLE":OD2_LLE_gen}
# all_dsetsDimReduct_Eval_gen = {"LEM":OD2_SpcE_gen}


if os.path.exists('/Users/uqoaremu/Research/Cpp_PHM/')==1:
    orig_jets= 'uqoaremu'
else:
    orig_jets= 'aremu'


all_dsetsDimReduct_Eval = {"Original Data Set":pd.read_pickle('/Users/{0}/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_train_FD002'.format(orig_jets)),
                  "RML":pd.read_pickle('/Users/{0}/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_RML_train_FD002'.format(orig_jets)),
                  "Isomap":pd.read_pickle('/Users/{0}/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_isomap_FD002'.format(orig_jets)),
                  "LLE":pd.read_pickle('/Users/{0}/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_LLE_FD002'.format(orig_jets)),
                  "LEM":pd.read_pickle('/Users/{0}/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_SpcE_FD002'.format(orig_jets))}

all_dsetsDimReduct_Eval_OD2 = {"Original Data Set":pd.read_pickle('/Users/{0}/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_train_FD004'.format(orig_jets)),
                  "RML":pd.read_pickle('/Users/{0}/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_RML_train_FD004'.format(orig_jets)),
                  "Isomap":pd.read_pickle('/Users/{0}/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_isomap_FD004'.format(orig_jets)),
                  "LLE":pd.read_pickle('/Users/{0}/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_LLE_FD004'.format(orig_jets)),
                  "LEM":pd.read_pickle('/Users/{0}/Research/Cpp_PHM/svd_CMAPPSDATA/CMAPPS_SpcE_FD004'.format(orig_jets))}

=
# -----------------------------------------------------------------------#

#PRediction

# Regression for prediction 6 conditions 1 faults
Regression_analysis_results_all_DimRedct = DimReduct_Eval.regres_pred(all_dsetsDimReduct_Eval,plot=True,saveto='RML/regrespredictRsltr')

# Regression for prediction 6 conditions 2 faults
Regression_analysis_results_all_DimRedct2faults = DimReduct_Eval.regres_pred(all_dsetsDimReduct_Eval_OD2,plot=True,saveto='RML/2faults_regrespredictRsltr')

=
# -----------------------------------------------------------------------#
# Aanlysis of 6 conditions 1 faults
# Cluster (fail/norm)
# Clsrt_analysis_results_all_cond_DimRedct = DimReduct_Eval.failnom_clust(all_dsetsDimReduct_Eval, plot=True,saveto='RML/clstrfailnomRslt')
# #Clsrt_analysis_results_all_cond_DimRedct_gen = DimReduct_Eval.failnom_clust(all_dsetsDimReduct_Eval_gen, plot=True,saveto='RML/clstrfailnomRslt_GenEval')

# Classification (nom/fail)
Clssfy_analysis_results_all_FailNom_DimRedct = DimReduct_Eval.fail_classify(all_dsetsDimReduct_Eval,plot=True,saveto='RML/clsfyFailRsltr_proper')
#Clssfy_analysis_results_all_FailNom_DimRedct_gen = DimReduct_Eval.fail_classify(all_dsetsDimReduct_Eval_gen,plot=True,saveto='RML/clsfyFailRslt_GenEval')

# Anomaly Detection
eval_results_DimRedct = DimReduct_Eval.anomalydetec(all_dsetsDimReduct_Eval,saveto='RML/anomdtcRslt_proper')
#eval_results_DimRedct_gen = DimReduct_Eval.anomalydetec(all_dsetsDimReduct_Eval_gen,saveto='RML/anomdtcRslt_GenEval')


# -----------------------------------------------------------------------#
# Aanlysis of 6 conditions 2 faults
# Cluster (fail/norm)
# Clsrt_analysis_results_all_cond_DimRedct2faults = DimReduct_Eval.failnom_clust(all_dsetsDimReduct_Eval_OD2, plot=True,saveto='RML/2faults_clstrfailnomRslt')
# #Clsrt_analysis_results_all_cond_DimRedct_gen = DimReduct_Eval.failnom_clust(all_dsetsDimReduct_Eval_gen, plot=True,saveto='RML/clstrfailnomRslt_2GenEval')

# Classification (nom/fail)
Clssfy_analysis_results_all_FailNom_DimRedct2faults = DimReduct_Eval.fail_classify(all_dsetsDimReduct_Eval_OD2,plot=True,saveto='RML/2faults_clsfyFailRslt_proper')
#Clssfy_analysis_results_all_FailNom_DimRedct_gen = DimReduct_Eval.fail_classify(all_dsetsDimReduct_Eval_gen,plot=True,saveto='RML/clsfyFailRslt_2GenEval')


# Anomaly Detection
eval_results_DimRedct2faults = DimReduct_Eval.anomalydetec(all_dsetsDimReduct_Eval_OD2,saveto='RML/2faults_anomdtcRslt_proper')
#eval_results_DimRedct_gen = DimReduct_Eval.anomalydetec(all_dsetsDimReduct_Eval_gen,saveto='RML/anomdtcRslt_2GenEval')
# -----------------------------------------------------------------------#

