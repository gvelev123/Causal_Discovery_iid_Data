import pickle
import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.1"
import pandas as pd
import rpy2
from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
saveRDS = robjects.r['saveRDS']
import numpy as np

def python_pickle_to_rds_r(frames_list,save_path):
    for data_item_index in range(0,len(frames_list)):
        pc_stable_estimation=frames_list[data_item_index][-1]['PC_stable']
        frames_list=frames_list[:5]
        with localconverter(robjects.default_converter + pandas2ri.converter):
            frames_list[data_item_index][1]=robjects.conversion.py2rpy(pd.DataFrame(frames_list[data_item_index][1]))
            frames_list[data_item_index][2]=robjects.conversion.py2rpy(pd.DataFrame(frames_list[data_item_index][2]))
            frames_list[data_item_index][3]=robjects.conversion.py2rpy(pd.DataFrame(frames_list[data_item_index][3]))
            pc_stable_estimation_rds=robjects.conversion.py2rpy(pd.DataFrame(pc_stable_estimation))
            frames_list[data_item_index][-1]=pc_stable_estimation_rds
    saveRDS(frames_list,save_path)

def extract_beta_results(current_betas):
    res_frame=[]
    for beta_item in current_betas:

        h_values=beta_item[-1][1]
        true_wa_std=beta_item[2].flatten()[np.flatnonzero(beta_item[2])].std()

        index_10_p=int(len(h_values)*0.1)
        index_20_p=int(len(h_values)*0.2)
        index_30_p=int(len(h_values)*0.3)
        index_40_p=int(len(h_values)*0.4)
        index_50_p=int(len(h_values)*0.5)
        index_60_p=int(len(h_values)*0.6)
        index_70_p=int(len(h_values)*0.7)
        index_80_p=int(len(h_values)*0.8)
        index_90_p=int(len(h_values)*0.9)

        res_frame.append([beta_item[0][-3],
                          #Retrieve information about h values: 
                          h_values[0], h_values[index_10_p], h_values[index_20_p],
                          h_values[index_30_p],h_values[index_40_p],h_values[index_50_p],
                          h_values[index_60_p],h_values[index_70_p],h_values[index_80_p],
                          h_values[index_90_p],h_values[-1],
                          #retrieve information about weighted adjacency matrix: num edges
                          len(np.flatnonzero(beta_item[-1][2][0])),len(np.flatnonzero(beta_item[-1][2][index_10_p])),
                          len(np.flatnonzero(beta_item[-1][2][index_20_p])),len(np.flatnonzero(beta_item[-1][2][index_30_p])),
                          len(np.flatnonzero(beta_item[-1][2][index_40_p])),len(np.flatnonzero(beta_item[-1][2][index_50_p])),
                          len(np.flatnonzero(beta_item[-1][2][index_60_p])),len(np.flatnonzero(beta_item[-1][2][index_70_p])),
                          len(np.flatnonzero(beta_item[-1][2][index_80_p])),len(np.flatnonzero(beta_item[-1][2][index_90_p])),
                          len(np.flatnonzero(beta_item[-1][2][-1])),
                          #retrieve information about weighted adjacency: difference in std. dev of nonzero elements:
                          np.abs(true_wa_std-beta_item[-1][2][0].flatten()[np.flatnonzero(beta_item[-1][2][0])].std()),
                          np.abs(true_wa_std-beta_item[-1][2][index_10_p].flatten()[np.flatnonzero(beta_item[-1][2][index_10_p])].std()),
                          np.abs(true_wa_std-beta_item[-1][2][index_20_p].flatten()[np.flatnonzero(beta_item[-1][2][index_20_p])].std()),
                          np.abs(true_wa_std-beta_item[-1][2][index_30_p].flatten()[np.flatnonzero(beta_item[-1][2][index_30_p])].std()),
                          np.abs(true_wa_std- beta_item[-1][2][index_40_p].flatten()[np.flatnonzero(beta_item[-1][2][index_40_p])].std()),
                          np.abs(true_wa_std-beta_item[-1][2][index_50_p].flatten()[np.flatnonzero(beta_item[-1][2][index_50_p])].std()),
                          np.abs(true_wa_std-beta_item[-1][2][index_60_p].flatten()[np.flatnonzero(beta_item[-1][2][index_60_p])].std()),
                          np.abs(true_wa_std-beta_item[-1][2][index_70_p].flatten()[np.flatnonzero(beta_item[-1][2][index_70_p])].std()),
                          np.abs(true_wa_std-beta_item[-1][2][index_80_p].flatten()[np.flatnonzero(beta_item[-1][2][index_80_p])].std()),
                          np.abs(true_wa_std-beta_item[-1][2][index_90_p].flatten()[np.flatnonzero(beta_item[-1][2][index_90_p])].std()),
                          np.abs(true_wa_std-beta_item[-1][2][-1].flatten()[np.flatnonzero(beta_item[-1][2][-1])].std())
                          #retrieve information about weighted adjacency: compute evaluation criteria
    #                     beta_item[-1][2][0].flatten()[np.flatnonzero(beta_item[-1][2][0])].mean(),
    #                     beta_item[-1][2][0].flatten()[np.flatnonzero(beta_item[-1][2][0])].std(),
                         ])
    res_frame=pd.DataFrame(np.array(res_frame),columns=['Beta_Upper_Limit','H_Start','H_10%','H_20%','H_30%',
                                                               'H_40%','H_50%','H_60%','H_70%','H_80%','H_90%','H_End',
                                                               'NumEdges_Start','NumEdges_10%','NumEdges_20%','NumEdges_30%',
                                                               'NumEdges_40%','NumEdges_50%','NumEdges_60%','NumEdges_70%','NumEdges_80%','NumEdges_90%','NumEdges_End',
                                                              'Diff_Std_Start','Diff_Std_10%','Diff_Std_20%','Diff_Std_30%',
                                                               'Diff_Std_40%','Diff_Std_50%','Diff_Std_60%','Diff_Std_70%','Diff_Std_80%','Diff_Std_90%','Diff_Std_End'])
    return res_frame

def prepare_h_grouping(h_grouping):
    #H Values Grouping:
    h_grouping_tr=[]
    h_grouping_tr_training=[]
    beta_values=[]
    for row_idx in range(0,h_grouping.shape[0]):
        h_grouping_tr.extend(list(h_grouping.iloc[row_idx,:-1].values))
        h_grouping_tr_training.extend(list(h_grouping.iloc[row_idx,:-1].keys()))
        beta_values.extend([h_grouping.iloc[row_idx,-1]]*len(h_grouping.iloc[row_idx,:-1].values))
    h_grouping=pd.DataFrame({'Beta_Upper_Limit':np.array(beta_values).astype(np.int16),
                             'H_Values':np.array(h_grouping_tr),
                             'Stage_Training_Process':np.array(h_grouping_tr_training)})
    #Filter out beta values higher than 8.0:
    h_grouping=h_grouping[h_grouping['Beta_Upper_Limit']<=8.0]
    h_grouping['Stage_Training_Process']=h_grouping['Stage_Training_Process'].map({'H_Start':'Start','H_10%':'10%','H_20%':'20%',
                                                                              'H_30%':'30%','H_40%':'40%','H_50%':'50%',
                                                                              'H_60%':'60%','H_70%':'70%','H_80%':'80%',
                                                                              'H_90%':'90%','H_End':'End'})
    return h_grouping

    