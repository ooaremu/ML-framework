'''
Created on 22Sep.,2017

@author: uqoaremu
'''
import pandas as pd
import scipy.stats as scy_stats
import numpy as np
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing.data import normalize, minmax_scale, StandardScaler,RobustScaler
from IPython.extensions.autoreload import update_class
from matplotlib.pyplot import axis
import itertools
import os
from scipy.stats._continuous_distns import logistic
from numpy import rate, str
from builtins import float
import time
from sklearn import cluster, metrics
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import distance
from matplotlib.colors import ListedColormap
import matplotlib.cm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
# from sklearn.cross_validation import KFold   #For K-fold cross validation
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
# from root.nested.ExpControl import  ExpControl
from scipy import stats, integrate
import seaborn as sns
from sklearn.feature_selection.univariate_selection import SelectKBest,\
    f_classif, SelectPercentile, SelectFpr, SelectFdr,\
    SelectFwe,GenericUnivariateSelect, f_regression
from scipy.stats._continuous_distns import chi2
from sklearn.pipeline import FeatureUnion
import string
from sklearn.svm.classes import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.gpr import GaussianProcessRegressor
from sklearn.feature_selection.rfe import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.feature_selection.from_model import SelectFromModel
from sklearn.ensemble.voting_classifier import VotingClassifier
from pandas.core import series
from statsmodels.stats.outliers_influence import variance_inflation_factor
from _tracemalloc import stop
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from matplotlib import offsetbox
from sklearn import (manifold, decomposition, ensemble,
                     discriminant_analysis, random_projection)
# from root.nested.ExpControl import ExpControl 
from sklearn.decomposition.kernel_pca import KernelPCA
from cmath import sqrt
from decorator import append
from scipy.stats.stats import pearsonr, spearmanr
#from plotly.graph_objs.graph_objs import Scatter3d

# -----------------------------------------------------------------------#



class ExpDimRed(object):
    '''
    This is used for various dimension specific to condition and behavior separation
    http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_kernel_pca.html#sphx-glr-auto-examples-decomposition-plot-kernel-pca-py
    http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#sphx-glr-auto-examples-plot-compare-reduction-py
    http://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html#sphx-glr-auto-examples-manifold-plot-mds-py
    '''


    def __init__(self, experimentname):
        self.experimentname = experimentname
    
    def sep_per_OPcond(self,dset,cond=6):
        
        OD_d1 = dset
        op_indx = KMeans(n_clusters=cond,random_state=0).fit(OD_d1[[ 'op_set1', 'op_set2', 'op_set3']]).labels_.astype(np.int) #index of condtions
        op_indx = op_indx+1 # avoid 0's
        OD_d1 = OD_d1.drop([ 'op_set1', 'op_set2', 'op_set3'],axis=1)
        OD_d1.insert(2, 'Opcond', op_indx) # machine data with op condition index

        return OD_d1
    
    def SpcE(self,dset,plot=None,saveto=None,conds=6):
        '''
        Spectral embedding using ML-Framework
        '''
        OD_redct = pd.DataFrame()
        #reduce dimension per condition
        t0 = time.time()
        error_Units = []
        for i in dset.unit.unique():
            OD_conds = ExpDimRed('DimReduct').sep_per_OPcond(dset,cond=conds)
            OD_per_conds_unit=pd.DataFrame()
            error_conds = []
            for j in OD_conds.Opcond.unique():
                gtent = OD_conds.loc[OD_conds['Opcond']==j].loc[OD_conds['unit']==i]
    
                reducts1 = gtent.drop(['unit', 'cycle', 'Opcond', 'faults'],axis=1)
                #print("Computing PCA projection")
                clf = manifold.SpectralEmbedding(n_components=10, random_state=0, eigen_solver="arpack",n_jobs=1,affinity='rbf')
                reducts = clf.fit_transform(reducts1)
                reducts = pd.DataFrame(reducts,columns=['pca1','pca8','pca3','pca4','pca5','pca6','pca7', "pca2", "pca9", "pca10"])
                #reducts = reducts.drop(['sm8','pca3','pca4','pca5','pca6', "sm7", "sm9", "sm10"], axis=1)
                
                gtent_reduced = pd.DataFrame(gtent[['unit', 'cycle', 'Opcond', 'faults']].values,columns=['unit', 'cycle', 'Opcond', 'faults'])
                gtent_reduced = pd.concat([gtent_reduced,reducts],axis=1)
                OD_per_conds_unit = pd.concat([OD_per_conds_unit,gtent_reduced],axis=0)
                
                #reprojct
                covDat = np.cov(reducts)
                loss= 1/np.sum(covDat**2)
                error_conds = np.append(error_conds, loss)
            # sort back into  per cycle time
            OD_per_conds_unitsort = OD_per_conds_unit.sort_values(by=['cycle'], ascending=[True])
            OD_per_conds_unitsort = pd.DataFrame(OD_per_conds_unitsort.values,columns=OD_per_conds_unitsort.columns)
            error_conds_frm = pd.DataFrame(error_conds)
            avg_error_conds = error_conds_frm.mean(axis=0)
            
            
            OD_redct = pd.concat([OD_redct,OD_per_conds_unitsort],axis=0)
            error_Units = np.concatenate(([error_Units,avg_error_conds]))
        t1 = time.time()   
        error_Units_Frame = pd.DataFrame(error_Units)         
        
        # plot
        if plot is not None:
            plt.figure(figsize=[30,10])
            plt.suptitle('{0} Dimension Reduction of Data set with {1} fault(s). Run Time: {2}s '.format(self.experimentname,OD_conds.faults.max(),round(t1-t0,2)))#,round(error_Units_Frame.mean()[0],3)))#

            ax = plt.subplot(2,2, 1)
            ax.set_title("Life Cycle of All Units per Feature 1")
            for ii in OD_conds.unit.unique():
                ax.scatter(OD_redct.loc[OD_redct['unit']==ii].cycle,OD_redct.loc[OD_redct['unit']==ii].pca1,s=8)
            #ax.set_xlim(-2,3)
            #ax.set_ylim(-2,2.5)
            #ax.text(1.007-.003, .347+.003,'{0} samples'.format(X_tst.shape[0]),size=8, verticalalignment='bottom',horizontalalignment='right')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Feature 1')
            #ax.set_xticks(())
            ##ax.set_yticks(())
            ax.axis('tight')
            
            ax = plt.subplot(2,2, 2)
            ax.set_title("Life Cycle of All Units per Feature 2")
            for ii in OD_conds.unit.unique():
                ax.scatter(OD_redct.loc[OD_redct['unit']==ii].cycle,OD_redct.loc[OD_redct['unit']==ii].pca2,s=8)
            #ax.set_xlim(-2,3)
            #ax.set_ylim(-2,2.5)
            #ax.text(1.007-.003, .347+.003,'{0} samples'.format(X_tst.shape[0]),size=8, verticalalignment='bottom',horizontalalignment='right')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Feature 2')
            #ax.set_xticks(())
            #ax.set_yticks(())
            ax.axis('tight')
            
            ax = plt.subplot(2,2, 3)
            ax.set_title("Life Cycle of Unit 4 per Feature 1")
            ax.plot(OD_redct.loc[OD_redct['unit']==4].cycle,OD_redct.loc[OD_redct['unit']==4].pca1)
            #ax.set_xlim(-2,3)
            #ax.set_ylim(-2,2.5)
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Feature 1')
            #ax.set_xticks(())
            #ax.set_yticks()
            ax.axis('tight')
            
            ax = plt.subplot(2,2, 4)
            ax.set_title("Life Cycle of Unit 4 per Feature 2")
            ax.plot(OD_redct.loc[OD_redct['unit']==4].cycle,OD_redct.loc[OD_redct['unit']==4].pca2)
            #ax.set_xlim(-2,3)
            #ax.set_ylim(-2,2.5)
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Feature 2')
            #ax.set_xticks(())
            #ax.set_yticks()
            ax.axis('tight')   

            plt.tight_layout()
            plt.subplots_adjust(top=.90)
            plt.show()
        if saveto is not None:
            picsave= '{0}_{1}.png'.format(saveto,self.experimentname)
            if plot is not None:
                plt.savefig(picsave, dpi=300)
        
        return OD_redct
    
    def SpcE_test(self,dset,plot=None,saveto=None,conds=6):
        '''
        Spectral embedding
        '''
        OD_redct = pd.DataFrame()
        #reduce dimension per condition
        t0 = time.time()
        error_Units = []
        for i in dset.unit.unique():
            #conds = dset.loc[dset['unit']==i].op_set3.unique().shape[0]
            print(dset.loc[dset['unit']==i])
            print(dset.loc[dset['unit']==i].shape[0])
            if dset.loc[dset['unit']==i].shape[0] <= 20:
                conds = 1
            elif dset.loc[dset['unit']==i].shape[0] <= 23 and dset.loc[dset['unit']==i].shape[0] > 20:
                conds = 1
            elif dset.loc[dset['unit']==i].shape[0] <= 25 and dset.loc[dset['unit']==i].shape[0] > 23:
                conds = 1
            elif dset.loc[dset['unit']==i].shape[0] == 28 and i ==156 or i == 239:
                conds = 1
            elif dset.loc[dset['unit']==i].shape[0] == 29 and i ==246 or i == 164:
                conds = 1
            elif dset.loc[dset['unit']==i].shape[0] <= 30 and dset.loc[dset['unit']==i].shape[0] > 25:
                conds = 2
            elif dset.loc[dset['unit']==i].shape[0] == 33:
                conds = 1 #3
            elif dset.loc[dset['unit']==i].shape[0] == 34 and i == 144 or i == 149 or i==169:
                conds = 1 #3
            elif dset.loc[dset['unit']==i].shape[0] == 37 and i == 59:
                conds = 1
            elif dset.loc[dset['unit']==i].shape[0] == 37 and i == 46:
                conds = 3
            elif dset.loc[dset['unit']==i].shape[0] <= 35 and dset.loc[dset['unit']==i].shape[0] > 30:
                conds = 2
            elif dset.loc[dset['unit']==i].shape[0] == 39 and i == 225:
                conds = 1
            elif dset.loc[dset['unit']==i].shape[0] == 42 and i == 241:
                conds = 2
            elif dset.loc[dset['unit']==i].shape[0] <= 40 and dset.loc[dset['unit']==i].shape[0] > 35:
                conds = 3
            elif dset.loc[dset['unit']==i].shape[0] == 43 and i == 224:
                conds = 2
            elif dset.loc[dset['unit']==i].shape[0] == 44 and i == 228:
                conds = 2
            elif dset.loc[dset['unit']==i].shape[0] == 51 and i == 5:
                conds = 3
            elif dset.loc[dset['unit']==i].shape[0] == 51 and i == 97:
                conds = 4
            elif dset.loc[dset['unit']==i].shape[0] <= 45 and dset.loc[dset['unit']==i].shape[0] > 40:
                conds = 5
            elif dset.loc[dset['unit']==i].shape[0] == 47 and i== 186:
                conds = 2
            elif dset.loc[dset['unit']==i].shape[0] == 48 and i== 170:
                conds = 4
            elif dset.loc[dset['unit']==i].shape[0] == 48 and i ==76:
                conds = 4
            elif dset.loc[dset['unit']==i].shape[0] <= 52 and dset.loc[dset['unit']==i].shape[0] >45:
                conds = 5
            elif dset.loc[dset['unit']==i].shape[0] == 63 and i == 166:
                conds = 1
            else:
                conds = conds
                
            OD_conds = ExpDimRed('DimReduct').sep_per_OPcond(dset,cond=conds)
            OD_per_conds_unit=pd.DataFrame()
            error_conds = []
            for j in OD_conds.Opcond.unique():
                print(i)
                print(j)
                
                gtent = OD_conds.loc[OD_conds['Opcond']==j].loc[OD_conds['unit']==i]
    
                reducts1 = gtent.drop(['unit', 'cycle', 'Opcond', 'faults'],axis=1)
                #print("Computing PCA projection")
                print(reducts1.shape[0])
                clf = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack",n_jobs=1,affinity='rbf',n_neighbors=gtent.shape[0]-1)
                reducts = clf.fit_transform(reducts1)
                reducts = pd.DataFrame(reducts,columns=['pca1','pca2'])
                #reducts = reducts.drop(['sm8','pca3','pca4','pca5','pca6', "sm7", "sm9", "sm10"], axis=1)
                
                gtent_reduced = pd.DataFrame(gtent[['unit', 'cycle', 'Opcond', 'faults']].values,columns=['unit', 'cycle', 'Opcond', 'faults'])
    
                gtent_reduced = pd.concat([gtent_reduced,reducts],axis=1)
    
                OD_per_conds_unit = pd.concat([OD_per_conds_unit,gtent_reduced],axis=0)
                #reprojct
                covDat = np.cov(reducts)
                loss= 1/np.sum(covDat**2)
                error_conds = np.append(error_conds, loss)
            # sort back into  per cycle time
            OD_per_conds_unitsort = OD_per_conds_unit.sort_values(by=['cycle'], ascending=[True])
            OD_per_conds_unitsort = pd.DataFrame(OD_per_conds_unitsort.values,columns=OD_per_conds_unitsort.columns)
            error_conds_frm = pd.DataFrame(error_conds)
            avg_error_conds = error_conds_frm.mean(axis=0)
            
            
            OD_redct = pd.concat([OD_redct,OD_per_conds_unitsort],axis=0)
            error_Units = np.concatenate(([error_Units,avg_error_conds]))
        t1 = time.time()   
        error_Units_Frame = pd.DataFrame(error_Units)         
        
        # plot
        if plot is not None:
            plt.figure(figsize=[30,10])
            plt.suptitle('{0} Dimension Reduction of Data set with {1} fault(s). Run Time: {2}s '.format(self.experimentname,OD_conds.faults.max(),round(t1-t0,2)))#,round(error_Units_Frame.mean()[0],3)))#
            unit = 213
            ax = plt.subplot(2,2, 1)
            ax.set_title("Life Cycle of All Units per Feature 1")
            for ii in OD_conds.unit.unique():
                ax.scatter(OD_redct.loc[OD_redct['unit']==ii].cycle,OD_redct.loc[OD_redct['unit']==ii].pca1,s=8)
            #ax.set_xlim(-2,3)
            #ax.set_ylim(-2,2.5)
            #ax.text(1.007-.003, .347+.003,'{0} samples'.format(X_tst.shape[0]),size=8, verticalalignment='bottom',horizontalalignment='right')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Feature 1')
            #ax.set_xticks(())
            ##ax.set_yticks(())
            ax.axis('tight')
            
            ax = plt.subplot(2,2, 2)
            ax.set_title("Life Cycle of All Units per Feature 2")
            for ii in OD_conds.unit.unique():
                ax.scatter(OD_redct.loc[OD_redct['unit']==ii].cycle,OD_redct.loc[OD_redct['unit']==ii].pca2,s=8)
            #ax.set_xlim(-2,3)
            #ax.set_ylim(-2,2.5)
            #ax.text(1.007-.003, .347+.003,'{0} samples'.format(X_tst.shape[0]),size=8, verticalalignment='bottom',horizontalalignment='right')
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Feature 2')
            #ax.set_xticks(())
            #ax.set_yticks(())
            ax.axis('tight')
            
            ax = plt.subplot(2,2, 3)
            ax.set_title("Life Cycle of Unit {0} per Feature 1".format(unit))
            ax.plot(OD_redct.loc[OD_redct['unit']==unit].cycle,OD_redct.loc[OD_redct['unit']==213].pca1)
            #ax.set_xlim(-2,3)
            #ax.set_ylim(-2,2.5)
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Feature 1')
            #ax.set_xticks(())
            #ax.set_yticks()
            ax.axis('tight')
            
            ax = plt.subplot(2,2, 4)
            ax.set_title("Life Cycle of Unit {0} per Feature 2".format(unit))
            ax.plot(OD_redct.loc[OD_redct['unit']==unit].cycle,OD_redct.loc[OD_redct['unit']==213].pca2)
            #ax.set_xlim(-2,3)
            #ax.set_ylim(-2,2.5)
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Feature 2')
            #ax.set_xticks(())
            #ax.set_yticks()
            ax.axis('tight')   

            plt.tight_layout()
            plt.subplots_adjust(top=.90)
            plt.show()
        if saveto is not None:
            picsave= '{0}_{1}.png'.format(saveto,self.experimentname)

            if plot is not None:
                plt.savefig(picsave, dpi=300)
        
        return OD_redct

