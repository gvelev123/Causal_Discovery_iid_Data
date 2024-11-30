import sys
sys.path.append("..")

from Src.Utils import extract_beta_results,prepare_h_grouping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
import pandas as pd
import numpy as np
from seaborn import violinplot,swarmplot,barplot,boxplot,heatmap,kdeplot,lineplot
import statsmodels.api as sm
import pickle
        

        


def plot_sf_er_graph_NodesEdgesMapping(er_frames_descriptions,
                                      sf_frames_descriptions):
    er_edges_frame=er_frames_descriptions[['Number_Nodes','Edge_Density','Number_Edges']]
    er_grouped=er_edges_frame.groupby(['Number_Nodes','Edge_Density']).mean()
    er_grouped['Number_Edges']=np.ceil(er_grouped['Number_Edges'].values)
    nr_nodes_density=[[ind[0],ind[1]] for ind in er_grouped.index] 
    er_grouped[['Number_Nodes','Edge_Density']]=np.array(nr_nodes_density)

    #SF graphs
    sf_edges_frame=sf_frames_descriptions[['Number_Nodes','K','Number_Edges']]
    sf_grouped=sf_edges_frame.groupby(['Number_Nodes','K']).mean()
    sf_grouped['Number_Edges']=np.ceil(sf_grouped['Number_Edges'].values)
    nr_nodes_k=[[ind[0],ind[1]] for ind in sf_grouped.index] 
    sf_grouped[['Number_Nodes','K']]=np.array(nr_nodes_k)


    unique_nr_nodes = er_grouped['Number_Nodes'].unique()
    nodes_edges_mapping = {
        '0.2': er_grouped[er_grouped['Edge_Density']==0.2]['Number_Edges'].values,
        '0.3': er_grouped[er_grouped['Edge_Density']==0.3]['Number_Edges'].values,
        '0.4': er_grouped[er_grouped['Edge_Density']==0.4]['Number_Edges'].values,}

    edge_density_colors={'0.2':'black',
                    '0.3':'silver',
                    '0.4':'whitesmoke'}

    x = np.arange(len(unique_nr_nodes))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    #plt.figure(figsize=(20,30))
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
    plt.figtext(0.02, 0.98, "a)", fontsize=13,weight='bold')
    plt.figtext(0.52, 0.98, "b)", fontsize=13,weight='bold')


    for attribute, measurement in nodes_edges_mapping.items():
        offset = width * multiplier
        rects = ax[0].bar(x + offset, measurement, width, label=attribute,color=edge_density_colors[attribute],
                          edgecolor='0.01')
        ax[0].bar_label(rects, padding=3)
        multiplier += 1



    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0].set_ylabel('Avg. Number Edges',fontsize=14)
    ax[0].set_xlabel('Node Size',fontsize=14)
    ax[0].set_xticks(x + width, unique_nr_nodes)
    ax[0].legend(title='Edge Density of ER & SF Graphs',loc='upper center', bbox_to_anchor=(0.17, -0.1),
              fancybox=True, shadow=True, ncol=3)
    ax[0].set_ylim(0, 2100)

    #SF Graphs:
    unique_nr_nodes = sf_grouped['Number_Nodes'].unique()
    nodes_edges_mapping={}
    edge_density_colors={}
    for i in range(0,3):
        key=str(sf_grouped[sf_grouped['Number_Nodes']==10].iloc[i,
             2])+'_'+str(sf_grouped[sf_grouped['Number_Nodes']==20].iloc[i,
             2])+'_'+str(sf_grouped[sf_grouped['Number_Nodes']==50].iloc[i,
             2])+'_'+str(sf_grouped[sf_grouped['Number_Nodes']==100].iloc[i,2])

        values=[sf_grouped[sf_grouped['Number_Nodes']==10].iloc[i,0],
        sf_grouped[sf_grouped['Number_Nodes']==20].iloc[i,0],
        sf_grouped[sf_grouped['Number_Nodes']==50].iloc[i,0],
        sf_grouped[sf_grouped['Number_Nodes']==100].iloc[i,0]]
        nodes_edges_mapping[key]=values

    edge_density_colors={'1_2_5_10': 'black',
                         '2_3_8_15':'silver',
                         '3_4_10_20': 'whitesmoke'}
    x = np.arange(len(unique_nr_nodes))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0


    for attribute, measurement in nodes_edges_mapping.items():
        offset = width * multiplier
        attributes=attribute.split('_')
        rects = ax[1].bar(x + offset, measurement, width, 
                           label=(0.2 if attribute =='1_2_5_10' else (0.3 if attribute=='2_3_8_15' else 0.4)),
                         color=edge_density_colors[attribute],edgecolor='0.01')
        ax[1].bar_label(rects, padding=3)
        multiplier += 1

    ax[1].set_xlabel('Node Size',fontsize=14)
    ax[1].set_xticks(x + width, unique_nr_nodes)
    ax[1].set_ylim(0, 1900)
    fig.tight_layout()
    plt.show()
    plt.close()
    print('\tFigure Caption: a) ER Graphs, b) SF Graphs')





class Sensitivity_Analysis():
    def __init__(self,
                ebm_explanations,
                results_df,
                ebm_threshold,
                read_path):
        
        self.ebm_merge=ebm_explanations
        self.all_csl_regrouped=results_df
        self.emb_importance_threshold=ebm_threshold
        self.read_path=read_path
        
        super(Sensitivity_Analysis,self).__init__()
        
    def plot_ebm_importance_interaction_scores(self):
        sensitivity_frame_reduced=self.ebm_merge[self.ebm_merge["Importance_Types"]!="interaction"]
        sensitivity_frame_reduced=sensitivity_frame_reduced.pivot(index="Simulation_Component",
                                                                  columns="CSL_Model",values="Importance_Scores")
        binary_heatmap=np.where(sensitivity_frame_reduced>=self.emb_importance_threshold,1.0,0.0)

        colors = ["silver","dimgrey"]
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors),gamma=1.5)

        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(14,3))
        plt.figtext(-0.05, 0.95, "a)", fontsize=14,weight='bold')
        heatmap(binary_heatmap,linewidth="1.5",linecolor="w",cmap=cmap,
                yticklabels=sensitivity_frame_reduced.index,annot=sensitivity_frame_reduced.values,fmt=".3f",annot_kws={"size":11},
               xticklabels=sensitivity_frame_reduced.columns,cbar=False,ax=ax)


        ax.set_xticklabels(ax.get_xticklabels(),rotation=25,fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
        ax.set_ylabel("Experimental \nFactors",fontsize=16)


        sensitivity_frame_reduced=self.ebm_merge[self.ebm_merge["Importance_Types"]=="interaction"]
        sensitivity_frame_reduced=sensitivity_frame_reduced.pivot(index="Simulation_Component",
                                                                  columns="CSL_Model",values="Importance_Scores")
        interaction_indices=[idx for idx in range(0,len(sensitivity_frame_reduced.index)) if sensitivity_frame_reduced.index[idx] in ["Nodes & Sample_Size","Nodes & Connectivity","Nodes & Scale","Scale & Sample_Size",
                                                                      "Scale & Graph_Type","Beta_Upper_Limit & Scale","Nodes & Transformation_Function"] ]
        sensitivity_frame_reduced=sensitivity_frame_reduced.iloc[interaction_indices,:].fillna(0.0)
        binary_heatmap=np.where(sensitivity_frame_reduced>=self.emb_importance_threshold,1.0,0.0)

        colors = ["silver","dimgrey"]
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors),gamma=1.5)

        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(14,4.7))
        plt.figtext(0.02, 0.99, "b)", fontsize=14,weight='bold')
        heatmap(binary_heatmap,linewidth="1.5",linecolor="w",cmap=cmap,
                yticklabels=sensitivity_frame_reduced.index,annot=sensitivity_frame_reduced.values,fmt=".4f",annot_kws={"size":11},
               xticklabels=sensitivity_frame_reduced.columns,cbar=True,cbar_kws={"orientation":"horizontal",
                                                                                "fraction":0.05,
                                                                                "label":"<0.01 >=0.01",
                                                                                "pad":0.28},ax=ax)
        colorbar=ax.collections[0].colorbar
        colorbar.set_label(label="<0.01                 >=0.01",fontsize=13)
        colorbar.set_ticks([])

        ax.set_xticklabels(ax.get_xticklabels(), rotation=25,fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
        ax.set_xlabel("CSL Model",fontsize=16)
        ax.set_ylabel("Interaction \nEffects",fontsize=16)
        fig.tight_layout()

        plt.show()
        plt.close()
        print('\tFigure Caption: a) Importance Scores for experimental factors included in the simulation framework, \n \t\t\tb) EBM scores for estimated interaction effects.')
        
    def plot_nodes_connectivity(self):
        self.all_csl_regrouped['Nodes_Connectivity_Str']=np.array([str(self.all_csl_regrouped.iloc[row_idx,:]['Nodes'])+', \n'+str(self.all_csl_regrouped.iloc[row_idx,:]['Connectivity']) for row_idx in range(0,self.all_csl_regrouped.shape[0])])
        self.all_csl_regrouped['Nodes_GraphType_Str']=np.array([str(self.all_csl_regrouped.iloc[row_idx,:]['Nodes'])+', \n'+str(self.all_csl_regrouped.iloc[row_idx,:]['Graph_Type']) for row_idx in range(0,self.all_csl_regrouped.shape[0])])

        all_cols_components=['CSL_Model','Nr_Run','Nodes','Connectivity','Transformation_Function','Beta_Upper_Limit','Scale',
         'Graph_Type','Sample_Size','DOS']

        #Sensitivity Nodes:
        current_component='Nodes'
        values=[[10,20],[10,50],[10,100]]
        remaining_components_current=all_cols_components.copy()
        remaining_components_current.remove(current_component)
        current_component_visualize=self.all_csl_regrouped[self.all_csl_regrouped[current_component]==values[0][0]][remaining_components_current].drop(['DOS'],axis=1)


        for values_index in range(0,len(values)):
            current_component_first_value=self.all_csl_regrouped[self.all_csl_regrouped[current_component]==values[values_index][0]][remaining_components_current]
            current_component_first_value.index=np.arange(0,current_component_first_value.shape[0],1)

            current_component_second_value=self.all_csl_regrouped[self.all_csl_regrouped[current_component]==values[values_index][1]][remaining_components_current]
            current_component_second_value.index=np.arange(0,current_component_second_value.shape[0],1)

            if current_component_first_value.drop(['DOS'],axis=1).equals(current_component_second_value.drop(['DOS'],axis=1))==True:
                current_delta_DOS=current_component_first_value[['DOS']]-current_component_second_value[['DOS']]
                current_component_visualize['Delta_DOS_'+str(values[values_index][0])+'_'+str(values[values_index][1])+'_'+current_component]=np.abs(current_delta_DOS['DOS'].values)#current_delta_DOS[current_delta_DOS['DOS']>=0.0].dropna()#

        print(current_component_visualize.columns)
        current_component_visualize['Sum_Delta_DOS_'+current_component]=current_component_visualize['Delta_DOS_10_20_Nodes'].values+current_component_visualize['Delta_DOS_10_50_Nodes'].values+current_component_visualize['Delta_DOS_10_100_Nodes'].values
        nodes_sorted=self.ebm_merge[self.ebm_merge['Simulation_Component']=='Nodes'].sort_values(by=['Importance_Scores'],ascending=False)
        criterion=list(nodes_sorted[nodes_sorted['Importance_Scores'].values>0.01]['CSL_Model'].values)


        #Inteaction Effect between nodes and sample size:
        #Redundant: to put in a function:
        sensitivity_frame_reduced=self.ebm_merge[self.ebm_merge["Importance_Types"]=="interaction"]
        sensitivity_frame_reduced=sensitivity_frame_reduced.pivot(index="Simulation_Component",
                                                                  columns="CSL_Model",values="Importance_Scores")
        interaction_indices=[idx for idx in range(0,len(sensitivity_frame_reduced.index)) if sensitivity_frame_reduced.index[idx] in ["Nodes & Sample_Size","Nodes & Connectivity","Nodes & Scale","Scale & Sample_Size",
                                                                      "Scale & Graph_Type","Beta_Upper_Limit & Scale","Nodes & Transformation_Function"] ]
        sensitivity_frame_reduced=sensitivity_frame_reduced.iloc[interaction_indices,:].fillna(0.0)
        ############################################################################################

        criterion_interaction=list(sensitivity_frame_reduced.loc["Nodes & Sample_Size"][sensitivity_frame_reduced.loc["Nodes & Sample_Size"]>=self.emb_importance_threshold].index)


        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(21,8))
        plt.figtext(0.02, 0.99, "a)", fontsize=15,weight='bold')
        plt.figtext(0.52, 0.99, "b)", fontsize=15,weight='bold')

        sns.boxplot(x='CSL_Model',
                            y='Sum_Delta_DOS_'+current_component,
                    #hue='Sample_Size',
                            data=current_component_visualize[current_component_visualize['CSL_Model'].isin(criterion)],
                    palette=["whitesmoke"],ax=ax[0])
                            #palette=['dimgrey','silver'])
        #plt.legend(bbox_to_anchor=(0.12, -0.1))
        #ax[0].set_title('Sensitivity to Node Size',fontsize=14)
        ax[0].tick_params(axis='y', which='major', labelsize=15, width=2.5, length=10)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=40,fontsize=14)
        ax[0].set_ylabel("Δ DOS Sum\n(Nodes ∈ [10, 20, 50, 100])",fontsize=16)
        ax[0].set_xlabel("CSL Model",fontsize=17)

        criterion_interaction=list(sensitivity_frame_reduced.loc["Nodes & Sample_Size"][sensitivity_frame_reduced.loc["Nodes & Sample_Size"]>=0.01].index)
        nodes_sample_size_frame=self.all_csl_regrouped[(self.all_csl_regrouped["CSL_Model"].isin(criterion_interaction))&(self.all_csl_regrouped["Nodes"].isin([10,100]))]
        nodes_sample_size_frame["Nodes_Sample_Size"]=np.array(["\n"+str(nodes_sample_size_frame["Nodes"].values[idx])+" Nodes,\n "+str(nodes_sample_size_frame["Sample_Size"].values[idx]) for idx in range(0,nodes_sample_size_frame.shape[0])])
        nodes_sample_size_frame=nodes_sample_size_frame.groupby(by=["CSL_Model","Nodes_Sample_Size"])["DOS"].mean()
        nodes_sample_size_frame=pd.DataFrame({"CSL_Model":np.array([nodes_sample_size_frame.index[idx][0] for idx in range(0,nodes_sample_size_frame.shape[0])]),
        "Nodes & Sample Size":np.array([nodes_sample_size_frame.index[idx][1] for idx in range(0,nodes_sample_size_frame.shape[0])]),
        "Avg. DOS":nodes_sample_size_frame.values})

        sns.barplot(x='CSL_Model',
                            y='Avg. DOS',
                    hue="Nodes & Sample Size",
                            data=nodes_sample_size_frame,
                    palette=['black','dimgray','silver','whitesmoke'],edgecolor='0.01',ax=ax[1],
                   orientation="vertical")
        ax[1].set_ylabel("Avg. DOS",fontsize=16)
        ax[1].set_xlabel("CSL Model",fontsize=16)
        ax[1].tick_params(axis='y', which='major', labelsize=15, width=2.5, length=10)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=40,fontsize=14)
        #ax[1].set_yticklabels(ax[1].get_yticklabels(), fontsize=13)
        ax[1].legend(title='Nodes & Sample Size',
                     loc=(0.01,-0.43),mode="expand",ncol=4,fontsize=13,title_fontsize=15)
        fig.tight_layout()
        plt.show()
        plt.close()


        
        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(19,7.3))
        plt.figtext(0.02, 0.99, "c)", fontsize=15,weight='bold')
        plt.figtext(0.52, 0.99, "d)", fontsize=15,weight='bold')

        #Sensitivity Connectivity:
        current_component='Connectivity'
        values=[[0.2,0.3],[0.2,0.4]]
        current_component_visualize=self.all_csl_regrouped[self.all_csl_regrouped[current_component]==values[0][0]][remaining_components_current].drop(['DOS'],axis=1)

        remaining_components_current=all_cols_components.copy()
        remaining_components_current.remove(current_component)

        for values_index in range(0,len(values)):
            current_component_first_value=self.all_csl_regrouped[self.all_csl_regrouped[current_component]==values[values_index][0]][remaining_components_current]
            current_component_first_value.index=np.arange(0,current_component_first_value.shape[0],1)

            current_component_second_value=self.all_csl_regrouped[self.all_csl_regrouped[current_component]==values[values_index][1]][remaining_components_current]
            current_component_second_value.index=np.arange(0,current_component_second_value.shape[0],1)

            if current_component_first_value.drop(['DOS'],axis=1).equals(current_component_second_value.drop(['DOS'],axis=1))==True:
                current_delta_DOS=current_component_first_value[['DOS']]-current_component_second_value[['DOS']]
                current_component_visualize['Delta_DOS_'+str(values[values_index][0])+'_'+str(values[values_index][1])+'_'+current_component]=np.abs(current_delta_DOS['DOS'].values)

        current_component_visualize['Sum_Delta_DOS_'+current_component]=current_component_visualize['Delta_DOS_0.2_0.3_Connectivity'].values+current_component_visualize['Delta_DOS_0.2_0.4_Connectivity'].values

        connectivity_sorted=self.ebm_merge[self.ebm_merge['Simulation_Component']=='Connectivity'].sort_values(by=['Importance_Scores'],ascending=False)
        connectivity_sorted['Threshold_Importance_Score']=np.where(connectivity_sorted['Importance_Scores'].values>=self.emb_importance_threshold,'>='+str(0.01),'<'+str(0.01))
        criterion=list(connectivity_sorted[connectivity_sorted['Threshold_Importance_Score']=='>='+str(0.01)]['CSL_Model'].values)
        current_component_visualize=current_component_visualize[current_component_visualize['CSL_Model'].isin(criterion)]

        boxplot(y='Sum_Delta_DOS_'+current_component,
                x='CSL_Model',
                #hue='Graph_Type',
                data=current_component_visualize,ax=ax[0],
               palette=['whitesmoke'])

        ax[0].tick_params(axis='y', which='major', labelsize=15, width=2.5, length=10)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=40,fontsize=14)
        ax[0].set_ylabel("Δ DOS Sum\n(Edge Density ∈ [0.2, 0.3, 0.4]",fontsize=16)
        ax[0].set_xlabel("CSL Model",fontsize=16)


        #Interaction Connectivity & Node Size:
        nodes_connectivity_sorted=self.ebm_merge[(self.ebm_merge['Simulation_Component']=='Nodes & Connectivity')&(self.ebm_merge['Importance_Scores'].values>=0.01)].sort_values(by=['Importance_Scores'],ascending=False)
        criterion=list(nodes_connectivity_sorted["CSL_Model"].value_counts().index)
        nodes_connectivity_interaction=self.all_csl_regrouped[(self.all_csl_regrouped["CSL_Model"].isin(criterion))&(self.all_csl_regrouped["Nodes"].isin([10,100]))&(self.all_csl_regrouped["Connectivity"].isin([0.2,0.4]))]
        nodes_connectivity_interaction=nodes_connectivity_interaction.groupby(by=["CSL_Model","Nodes_Connectivity_Str"])["DOS"].mean()
        nodes_connectivity_interaction=pd.DataFrame({"CSL_Model":np.array([nodes_connectivity_interaction.index[idx][0] for idx in range(0,nodes_connectivity_interaction.shape[0])]),
        "Node Size & Connectivity":np.array([nodes_connectivity_interaction.index[idx][1] for idx in range(0,nodes_connectivity_interaction.shape[0])]),
        "Avg. DOS":nodes_connectivity_interaction.values})


        sns.barplot(x='CSL_Model',
                    y='Avg. DOS',
                    hue="Node Size & Connectivity",
                    data=nodes_connectivity_interaction,
                    palette=['black','dimgray','silver','whitesmoke'],edgecolor='0.01',
                    orientation="vertical",ax=ax[1])
        ax[1].tick_params(axis='y', which='major', labelsize=15, width=2.5, length=10)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=40,fontsize=14)
        ax[1].set_ylabel("Avg. DOS",fontsize=16)
        ax[1].set_xlabel("CSL Model",fontsize=16)
        ax[1].legend(title='Nodes Size & Connectivity',
                     loc=(0.01,-0.45),mode="expand",ncol=6,fontsize=13,title_fontsize=14)


        fig.tight_layout()
        plt.show()
        plt.close()
        print('\tFigure Caption: a) & c) Sensitivity w.r.t. node size and connectivity, \n \t\t\tb) & d) EBM interaction effects of node size with sample size & connectivity, respectivelly.')
    
    def plot_interaction_effects_scale(self):
        self.all_csl_regrouped['Scale_Nodes_Str']=np.array([self.all_csl_regrouped.iloc[row_idx,:]['Scale']+', \n'+str(self.all_csl_regrouped.iloc[row_idx,:]['Nodes']) for row_idx in range(0,self.all_csl_regrouped.shape[0])])
        self.all_csl_regrouped['Scale_SampleSize_Str']=np.array(["\n"+self.all_csl_regrouped.iloc[row_idx,:]['Scale']+', \n'+str(self.all_csl_regrouped.iloc[row_idx,:]['Sample_Size']) for row_idx in range(0,self.all_csl_regrouped.shape[0])])
        self.all_csl_regrouped['Scale_GraphType_Str']=np.array([self.all_csl_regrouped.iloc[row_idx,:]['Scale']+', \n'+str(self.all_csl_regrouped.iloc[row_idx,:]['Graph_Type']) for row_idx in range(0,self.all_csl_regrouped.shape[0])])

        emb_filter_title='Importance Score\nthreshold:'
        emb_importance_threshold=0.01

        #Sensitivity Graph Type:
        graph_type_sorted=self.ebm_merge[self.ebm_merge['Simulation_Component']=='Graph_Type'].sort_values(by=['Importance_Scores'],ascending=False)
        graph_type_sorted['Threshold_Importance_Score']=np.where(graph_type_sorted['Importance_Scores'].values>=emb_importance_threshold,'>'+str(emb_importance_threshold),'<='+str(emb_importance_threshold))

        criterion=list(graph_type_sorted[graph_type_sorted['Threshold_Importance_Score']=='>'+str(emb_importance_threshold)]['CSL_Model'].values)
        graph_type_sensitive_models=self.all_csl_regrouped[self.all_csl_regrouped['CSL_Model'].isin(criterion)]

        grouped_graph_type_sensitive_models=graph_type_sensitive_models.groupby(by=['Graph_Type','CSL_Model'])['DOS'].mean()
        grouped_graph_type_sensitive_models=pd.DataFrame({'Graph_Type':np.array([idx[0] for idx in grouped_graph_type_sensitive_models.index]),
                    'CSL_Model':np.array([idx[1] for idx in grouped_graph_type_sensitive_models.index]),'Mean_DOS':grouped_graph_type_sensitive_models.values})


        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(19,7.3))
        plt.figtext(0.02, 0.99, "a)", fontsize=15,weight='bold')
        plt.figtext(0.52, 0.99, "b)", fontsize=15,weight='bold')

        barplot(y='Mean_DOS',x='CSL_Model',hue='Graph_Type',data=grouped_graph_type_sensitive_models,ax=ax[0],
                palette=['grey','whitesmoke'],edgecolor='0.2')

        ax[0].tick_params(axis='y', which='major', labelsize=15, width=2.5, length=10)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=40,fontsize=14)
        ax[0].set_ylabel("Avg. DOS",fontsize=16)
        ax[0].set_xlabel("CSL Model",fontsize=16)

        ax[0].legend(title='Graph Type',
                    loc=(0.01,-0.65),mode="expand",ncol=2,fontsize=13,title_fontsize=14)


        graph_scale_sorted=self.ebm_merge[(self.ebm_merge['Simulation_Component']=='Scale & Graph_Type')&(self.ebm_merge['Importance_Scores'].values>=0.01)].sort_values(by=['Importance_Scores'],ascending=False)
        criterion=list(graph_scale_sorted["CSL_Model"].value_counts().index)
        graph_scale_interaction=self.all_csl_regrouped[self.all_csl_regrouped['CSL_Model'].isin(criterion)].groupby(by=["CSL_Model","Scale_GraphType_Str"])["DOS"].mean()
        graph_scale_interaction=pd.DataFrame({'CSL_Model':np.array([idx[0] for idx in graph_scale_interaction.index]),
                    'Scale & Graph Type':np.array([idx[1] for idx in graph_scale_interaction.index]),
                    'Avg. DOS':graph_scale_interaction.values})

        sns.barplot(x='CSL_Model',
                    y='Avg. DOS',
                    hue="Scale & Graph Type",
                    hue_order=["original, \nER","standardized, \nER",
                            "original, \nSF","standardized, \nSF"],
                    data=graph_scale_interaction,
                    palette=['black','dimgray','silver','whitesmoke'],edgecolor='0.01',
                    orientation="vertical",ax=ax[1])

        ax[1].tick_params(axis='y', which='major', labelsize=15, width=2.5, length=10)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=40,fontsize=14)
        ax[1].set_ylabel("Avg. DOS",fontsize=15)
        #ax[1].set_ylabel("\n\n\n\n\n\n\n\n\nAvg. DOS",fontsize=15)
        ax[1].set_xlabel("CSL Model",fontsize=15)
        ax[1].legend(title='Scale & Graph Type',
                    loc=(0.01,-0.65),mode="expand",ncol=4,fontsize=12,title_fontsize=13)

        fig.tight_layout()
        plt.show()
        plt.close()

        #Interaction Scale & Node Size:
        nodes_scale_sorted=self.ebm_merge[(self.ebm_merge['Simulation_Component']=='Nodes & Scale')&(self.ebm_merge['Importance_Scores'].values>=0.01)].sort_values(by=['Importance_Scores'],ascending=False)
        criterion=list(nodes_scale_sorted["CSL_Model"].value_counts().index)
        nodes_scale_interaction=self.all_csl_regrouped[(self.all_csl_regrouped["CSL_Model"].isin(criterion))&((self.all_csl_regrouped["Nodes"].isin([10,100])))]
        nodes_scale_interaction=nodes_scale_interaction.groupby(by=["CSL_Model","Scale_Nodes_Str"])["DOS"].mean()
        nodes_scale_interaction=pd.DataFrame({"CSL_Model":np.array([nodes_scale_interaction.index[idx][0] for idx in range(0,nodes_scale_interaction.shape[0])]),
        "Scale & Node Size":np.array([nodes_scale_interaction.index[idx][1] for idx in range(0,nodes_scale_interaction.shape[0])]),
        "Avg. DOS":nodes_scale_interaction.values})

        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(19,7.3))
        plt.figtext(0.02, 0.99, "c)", fontsize=15,weight='bold')
        plt.figtext(0.52, 0.99, "d)", fontsize=15,weight='bold')

        sns.barplot(x='CSL_Model',
                    y='Avg. DOS',
                    hue="Scale & Node Size",
                    data=nodes_scale_interaction,
                    palette=['black','dimgray','silver','whitesmoke'],edgecolor='0.01',
                    hue_order=["original, \n10","standardized, \n10",
                            "original, \n100","standardized, \n100"],
                    orientation="vertical",ax=ax[0])

        ax[0].tick_params(axis='y', which='major', labelsize=15, width=2.5, length=10)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=40,fontsize=14)
        ax[0].set_ylabel("Avg. DOS",fontsize=16)
        ax[0].set_xlabel("CSL Model",fontsize=16)
        ax[0].legend(title='Scale & Node Size',
                    loc=(0.01,-0.65),mode="expand",ncol=4,fontsize=13,title_fontsize=13)


        #Interaction Scale & Sample Size:
        scale_sampleSize_sorted=self.ebm_merge[(self.ebm_merge['Simulation_Component']=='Scale & Sample_Size')&(self.ebm_merge['Importance_Scores'].values>=0.01)].sort_values(by=['Importance_Scores'],ascending=False)
        criterion=list(scale_sampleSize_sorted["CSL_Model"].value_counts().index)
        scale_sampleSize_interaction=self.all_csl_regrouped[self.all_csl_regrouped["CSL_Model"].isin(criterion)].groupby(by=["CSL_Model","Scale_SampleSize_Str"])["DOS"].mean()
        scale_sampleSize_interaction=pd.DataFrame({"CSL_Model":np.array([scale_sampleSize_interaction.index[idx][0] for idx in range(0,scale_sampleSize_interaction.shape[0])]),
        "Scale & Sample Size":np.array([scale_sampleSize_interaction.index[idx][1] for idx in range(0,scale_sampleSize_interaction.shape[0])]),
        "Avg. DOS":scale_sampleSize_interaction.values})

        sns.barplot(x='CSL_Model',
                    y='Avg. DOS',
                    hue="Scale & Sample Size",
                    hue_order=["\noriginal, \nLarge_Sample_Size","\nstandardized, \nLarge_Sample_Size",
                            "\noriginal, \nSmall_Sample_Size","\nstandardized, \nSmall_Sample_Size"],
                    data=scale_sampleSize_interaction,
                    palette=['black','dimgray','silver','whitesmoke'],edgecolor='0.01',
                    orientation="vertical",ax=ax[1])

        ax[1].tick_params(axis='y', which='major', labelsize=15, width=2.5, length=10)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=40,fontsize=14)
        ax[1].set_ylabel("Avg. DOS",fontsize=16)
        ax[1].set_xlabel("CSL Model",fontsize=16)
        ax[1].legend(title='Scale & Sample Size',
                    loc=(0.01,-0.65),mode="expand",ncol=4,fontsize=13,title_fontsize=13)


        fig.tight_layout()
        plt.show()
        plt.close()

        #Interaction Beta Upper Limit & Scale:
        beta_scale_sorted=self.ebm_merge[(self.ebm_merge['Simulation_Component']=='Beta_Upper_Limit & Scale')&(self.ebm_merge['Importance_Scores'].values>=0.01)].sort_values(by=['Importance_Scores'],ascending=False)
        criterion=list(beta_scale_sorted["CSL_Model"].value_counts().index)
        beta_scale_interaction=self.all_csl_regrouped[self.all_csl_regrouped['CSL_Model'].isin(criterion)].groupby(by=["CSL_Model","Scale","Beta_Upper_Limit"])["DOS"].mean()
        beta_scale_interaction=pd.DataFrame({'CSL_Model':np.array([idx[0] for idx in beta_scale_interaction.index]),
                    'Scale':np.array([idx[1] for idx in beta_scale_interaction.index]),
                    'Beta_Upper_Limit':np.array([idx[2] for idx in beta_scale_interaction.index]),
                    'Avg. DOS':beta_scale_interaction.values})

        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(19,7.3))
        plt.figtext(0.02, 0.99, "e)", fontsize=15,weight='bold')
        plt.figtext(0.52, 0.99, "f)", fontsize=15,weight='bold')

        sns.barplot(x='CSL_Model',
                    y='Avg. DOS',
                    hue="Beta_Upper_Limit",
                    data=beta_scale_interaction[beta_scale_interaction["Scale"]=="original"],
                    palette=['black','dimgray','silver','whitesmoke'],edgecolor='0.01',
                    orientation="vertical",ax=ax[0])

        ax[0].tick_params(axis='y', which='major', labelsize=15, width=2.5, length=10)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=40,fontsize=14)
        ax[0].set_ylabel("Avg. DOS",fontsize=16)
        ax[0].set_xlabel("CSL Model",fontsize=16)
        ax[0].legend(title='Beta Upper Limit (original Scale)',
                    loc=(0.01,-0.45),mode="expand",ncol=4,fontsize=13,title_fontsize=13)


        sns.barplot(x='CSL_Model',
                    y='Avg. DOS',
                    hue="Beta_Upper_Limit",
                    data=beta_scale_interaction[beta_scale_interaction["Scale"]=="standardized"],
                    palette=['black','dimgray','silver','whitesmoke'],edgecolor='0.01',
                    orientation="vertical",ax=ax[1])

        ax[1].tick_params(axis='y', which='major', labelsize=15, width=2.5, length=10)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=40,fontsize=14)
        ax[1].set_ylabel("Avg. DOS",fontsize=16)
        ax[1].set_xlabel("CSL Model",fontsize=16)
        ax[1].legend(title='Beta Upper Limit (standardized Scale)',
                    loc=(0.01,-0.45),mode="expand",ncol=4,fontsize=13,title_fontsize=13)
        fig.tight_layout()
        plt.show()
        plt.close()

        print('\tFigure Caption: a) Sensitivity w.r.t. graph type, \n \t\t\tb), c) & d) EBM interaction effects of scale with graph type, node size and sample size, \nrespectivelly,  \n \t\t\te) & f) Sensitivity w.r.t. beta upper limit on original and standardized scale, respectivelly, ')

    def plot_densities(self, axes,approaches_array,data,remove_legends,fig,
                    linewidth,super_title=None):
        for model_idx in range(0,len(approaches_array)):
            current_model=data[data['CSL_Model']==approaches_array[model_idx]]

            plotted_graph=kdeplot(data=current_model[current_model["Transformation_Function"]!='Linear 30%, ReLU 70%'],x='DOS',hue='Transformation_Function',
            fill=True,ax=axes[model_idx],palette=['black','dimgray','whitesmoke'],linewidth=linewidth,
                ec="0.5")#('0.5' if "R2SortnRegress" not in approaches_array else 0.9))
            #violinplot(x='CSL_Model',
            #       y="DOS", hue="Graph_Type", inner="quart", data=all_csl_regrouped[all_csl_regrouped['CSL_Model'].isin(criterion)],
            #    split=True,ax=ax[1])

            axes[model_idx].set_xlabel('DOS',fontsize=(16 if "R2SortnRegress" not in approaches_array else 8))#,weight='bold')
            axes[model_idx].set_ylabel(('Density' if "R2SortnRegress" not in approaches_array else"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDensity"),
                                    fontsize=(16 if "R2SortnRegress" not in approaches_array else 8))#,weight='bold')
            axes[model_idx].tick_params(labelsize=(16 if "R2SortnRegress" not in approaches_array else 8))
            if remove_legends[model_idx]==True:
                axes[model_idx].get_legend().remove()
            else:#"Linear_30%_ReLU_70%"
                axes[model_idx].legend(title='Transformation_Function',
                labels=["Linear 10%,\nReLU 90%",
                        #"Linear 30%,\nReLU 70%",
                        "Linear 50%,\nReLU 50%",
                        "Linear 100%"],
                mode="expand",ncol=4,labelspacing=1,
                loc=(0.01,-0.42),fontsize=14,title_fontsize=14)
        fig.tight_layout()
        plt.show()
        plt.close()


    def plot_transformation_function(self):
        transformation_sorted=self.ebm_merge[self.ebm_merge['Simulation_Component']=='Transformation_Function'].sort_values(by=['Importance_Scores'],ascending=False)
        transformation_sorted=transformation_sorted[transformation_sorted['Importance_Scores']>=0.01]
        orientation='h'
        emb_filter_title='Importance Score\nthreshold:'
        all_csl_regrouped_trF_mapped=self.all_csl_regrouped.copy()
        all_csl_regrouped_trF_mapped["Transformation_Function"]=all_csl_regrouped_trF_mapped["Transformation_Function"].map({'Linear_100%':'Linear 100%',
                                                                                                                            'Linear_ReLU_50%':'Linear, ReLU 50%',
                                                                                                                            'Linear_30%_ReLU_70%':'Linear 30%, ReLU 70%',
                                                                                                                            'Linear_10%_ReLU_90%':'Linear 10%, ReLU 90%'})
        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(7.5,2.7))
        plt.figtext(0.29, 0.99, "a)", fontsize=8,weight='bold')
        combinatorial_approaches=['R2SortnRegress']
        self.plot_densities(axes=[ax],approaches_array=combinatorial_approaches,
                        data=all_csl_regrouped_trF_mapped,remove_legends=[True],
                    fig=fig,linewidth=0.7)
        fig.tight_layout()
        plt.show()
        plt.close()

        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,5))
        plt.figtext(0.02, 0.99, "b)", fontsize=15,weight='bold')
        plt.figtext(0.52, 0.99, "c)", fontsize=15,weight='bold')
        combinatorial_approaches=['GOLEM','DIRECT-LINGAM']
        self.plot_densities(axes=[ax[0],ax[1]],approaches_array=combinatorial_approaches,
                        data=all_csl_regrouped_trF_mapped,remove_legends=[True,True],
                    fig=fig,linewidth=1.5)

        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,6))
        plt.figtext(0.02, 0.99, "d)", fontsize=15,weight='bold')
        plt.figtext(0.52, 0.99, "e)", fontsize=15,weight='bold')
        combinatorial_approaches=['DAGMA','NoTears_MLP']
        self.plot_densities(axes=[ax[0],ax[1]],approaches_array=combinatorial_approaches,
                        data=all_csl_regrouped_trF_mapped,remove_legends=[False,True],
                    super_title='CSL Models sensitive to \nTransformation Function w.r.t DOS',
                    fig=fig,linewidth=1.5)


        fig.tight_layout()
        plt.show()
        plt.close()

        print('\tFigure Caption: Sensitivity w.r.t. nonlinear, nonidentifiable causal transformations for \n \t\t\ta), b), c), d) & e) R2SortnRegress, GOLEM, DIRECT-LINGAM, DAGMA and NoTears_MLP, respectively. ')
    
    def plot_ranking_summary(self):
        conditioned_grouped=self.all_csl_regrouped.copy()
        conditioned_grouped=conditioned_grouped[conditioned_grouped["Scale"]!="original"]

        rankings=[conditioned_grouped.groupby(by=['CSL_Model'])[['FScore']].mean().sort_values(by='FScore',
                ascending=False),
                conditioned_grouped.groupby(by=['CSL_Model'])[['TPR']].mean().sort_values(by='TPR',
                ascending=False),
                conditioned_grouped.groupby(by=['CSL_Model'])[['FPR']].mean().sort_values(by='FPR',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['Causal_Order_Divergence']].mean().sort_values(by='Causal_Order_Divergence',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['nSHD']].mean().sort_values(by='nSHD',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['nSID']].mean().sort_values(by='nSID',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['DOS']].mean().sort_values(by='DOS',
                ascending=False)]

        dos_ranking=rankings[-1].copy()
        for rnk in rankings:
            rnk[rnk.columns[0]]=np.arange(1,15,1)
        rankings_frame=pd.concat(rankings,axis=1)
        rankings_frame.columns=["FScore","TPR","FPR","COD","nSHD","nSID","DOS"]

        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(17,7))

        plt.figtext(0.02, 0.99, "a)", fontsize=15,weight='bold')
        plt.figtext(0.54, 0.99, "b)", fontsize=15,weight='bold')

        heatmap(rankings_frame.sort_values(by="DOS",ascending=True),annot=True,
            linecolor="black",linewidths=1,cbar=False,ax=ax[0],
                cmap=ListedColormap(['whitesmoke']),annot_kws={"size":14})
        ax[0].set_xlabel("Performance Indicator Ranking",fontsize=17)
        ax[0].set_ylabel("CSL Model",fontsize=17)
        ax[0].tick_params(axis='both', which='major', labelsize=14, width=2.5, length=10,rotation=0)

        spearman_DOS_corr=rankings_frame.corr(method="spearman").iloc[:-1,][['DOS']].sort_values(by="DOS",ascending=False)
        barplot(x=spearman_DOS_corr["DOS"],y=spearman_DOS_corr.index,ax=ax[1],
        orientation="horizontal",palette=["whitesmoke"],edgecolor="0.01")
        ax[1].set_ylabel("One-Dimensional Metrics",fontsize=17)
        ax[1].set_xlabel("Ranking Correlation with DOS",fontsize=14)
        ax[1].tick_params(axis='both', which='major', labelsize=14, width=2.5, length=10,rotation=0)
        ax[1].set(xlim=(0, 1.07))
        for index, row in spearman_DOS_corr.iterrows():
            ax[1].text(row.DOS,np.where(spearman_DOS_corr.index==row.name)[0][0], round(row.DOS, 3),
                    color='black', horizontalalignment='left',fontsize=14)

        fig.tight_layout()
        plt.show()
        plt.close()

        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(8,5.7))
        plt.figtext(0.15, 0.99, "c)", fontsize=10,weight='bold')

        barplot(x=dos_ranking["DOS"],y=dos_ranking.index,ax=ax,
        orientation="horizontal",edgecolor="0.01",
                palette=['dimgray','silver','dimgray',
                        'dimgray','dimgray','dimgray','silver',
                        'silver','silver','dimgray',
                        'silver','silver','dimgray','silver'])
        ax.set_ylabel("\n\n\n\n\n\n\nCSL Model",fontsize=10)
        ax.set_xlabel("DOS",fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9, width=2.5, length=10,rotation=0)

        ax.set(xlim=(0, 0.72))
        for index, row in dos_ranking.iterrows():
            ax.text(row.DOS,np.where(dos_ranking.index==row.name)[0][0], round(row.DOS, 3),
                    color='black', horizontalalignment='left',fontsize=10)

        leg=ax.legend(title='Optmization Type',
                    labels=["Continuous","Combinatorial"],
                    mode="expand",ncol=2,
                    loc=(0.01,-0.25),fontsize=9,title_fontsize=9)
        leg=ax.get_legend()
        leg.legendHandles[0].set_color('silver')
        leg.legendHandles[1].set_color('dimgray')

        fig.tight_layout()
        plt.show()
        plt.close()

        print('\tFigure Caption: a) Ranking based on each one-dimensional criterion as well as DOS, \n \t\t\t b) Ranking correlation of one-dimnesional metrics with DOS, \n \t\t\t c) DOS performance on avg. achieved on standardized data.')

    def plot_appendix_ebm_threshold(self):
        sensitivity_frame_reduced=self.ebm_merge[self.ebm_merge["Importance_Types"]!="interaction"]
        sensitivity_frame_reduced=sensitivity_frame_reduced.pivot(index="Simulation_Component",
                                                                columns="CSL_Model",values="Importance_Scores")
        sensitivity_frame_reduced=sensitivity_frame_reduced.iloc[[5],:]

        binary_heatmap=np.where(sensitivity_frame_reduced>=0.01,1.0,0.0)

        colors = ["silver","dimgrey"]
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors),gamma=1.5)

        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(15,3.0))
        plt.figtext(0.01, 0.87, "a)\n\n\n", fontsize=17,weight='bold')
        heatmap(binary_heatmap,linewidth="1.5",linecolor="w",cmap=cmap,
                yticklabels=sensitivity_frame_reduced.index,annot=sensitivity_frame_reduced.values,fmt=".3f",annot_kws={"size":11},
            xticklabels=sensitivity_frame_reduced.columns,cbar=True,cbar_kws={"orientation":"horizontal",
                                                                                "fraction":0.1,
                                                                                "label":"<0.01 >=0.01",
                                                                                "pad":0.49},ax=ax)
        colorbar=ax.collections[0].colorbar
        colorbar.set_label(label="EBM Score <0.01            EBM Score >=0.01",fontsize=13)
        colorbar.set_ticks([])

        ax.set_xticklabels(ax.get_xticklabels(),rotation=25,fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=0)
        ax.set_ylabel("Experimental \nFactor \nScale",fontsize=15)
        ax.set_xlabel("CSL Model",fontsize=15)
        fig.tight_layout()

        plt.show()
        plt.close()

        emb_importance_threshold=0.01
        graph_type_sorted=self.ebm_merge[self.ebm_merge['Simulation_Component']=='Scale'].sort_values(by=['Importance_Scores'],ascending=False)
        graph_type_sorted['Threshold_Importance_Score']=np.where(graph_type_sorted['Importance_Scores'].values>=emb_importance_threshold,'>='+str(emb_importance_threshold),'<'+str(emb_importance_threshold))


        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,5))
        plt.figtext(0.10, 0.91, "b)\n", fontsize=14,weight='bold')
        plt.figtext(0.52, 0.91, "c)\n", fontsize=14,weight='bold')
        criterion=list(graph_type_sorted[graph_type_sorted['Threshold_Importance_Score']=='>='+str(emb_importance_threshold)]['CSL_Model'].values)


        boxplot(x='CSL_Model',y="DOS", hue="Scale",  data=self.all_csl_regrouped[self.all_csl_regrouped['CSL_Model'].isin(criterion)],
                    ax=ax[0],palette=['silver','whitesmoke'])
        ax[0].legend(title='Data Scale',
                    loc=(0.01,-0.42),mode='expand',ncol=2)
        #ax[0].set_title('Significant Variation in DOS w.r.t. Data Scale')
        ax[0].set_xlabel('CSL Model',fontsize=14)
        ax[0].tick_params(axis='x', which='major', labelsize=10,rotation=40)


        criterion=list(graph_type_sorted[graph_type_sorted['Threshold_Importance_Score']=='<'+str(emb_importance_threshold)]['CSL_Model'].values)
        boxplot(x='CSL_Model',y="DOS", hue="Scale",  data=self.all_csl_regrouped[self.all_csl_regrouped['CSL_Model'].isin(criterion)],
                    ax=ax[1],palette=['silver','whitesmoke'])
        ax[1].tick_params(axis='x', which='major', labelsize=10,rotation=40)
        ax[1].set_xlabel('CSL Model',fontsize=14)
        ax[1].set_ylabel('',fontsize=14)

        ax[1].legend().remove()
        #ax[1].set_title('Insignificant Variation in DOS w.r.t. Data Scale')
        plt.show()
        plt.close()

        models_list=[]
        p_vals=[]

        criterion=list(graph_type_sorted[graph_type_sorted['Threshold_Importance_Score']=='>='+str(emb_importance_threshold)]['CSL_Model'].values)
        self.all_csl_regrouped=self.all_csl_regrouped.dropna()
        for criterion_model in criterion:
            
            X=self.all_csl_regrouped[(self.all_csl_regrouped['CSL_Model']==criterion_model)][['Scale','DOS']]
            X['Scale']=X['Scale'].map({'original':1,'standardized':0})
            X = sm.add_constant(X)
            results_ols = sm.OLS(X['DOS'], X.drop(['DOS'],axis=1)).fit()
            p_val_ols=results_ols.pvalues['Scale']
            params_ols=results_ols.params['Scale']
            models_list.append(criterion_model)
            p_vals.append(p_val_ols)
            
        criterion=list(graph_type_sorted[graph_type_sorted['Threshold_Importance_Score']=='<'+str(emb_importance_threshold)]['CSL_Model'].values)
        for criterion_model in criterion:
            X=self.all_csl_regrouped[(self.all_csl_regrouped['CSL_Model']==criterion_model)][['Scale','DOS']]
            X['Scale']=X['Scale'].map({'original':1,'standardized':0})
            X = sm.add_constant(X)
            results_ols = sm.OLS(X['DOS'], X.drop(['DOS'],axis=1)).fit()
            p_val_ols=results_ols.pvalues['Scale']
            params_ols=results_ols.params['Scale']
            models_list.append(criterion_model)
            p_vals.append(p_val_ols)
            
        univariate_regressions=pd.DataFrame({'CSL Model':np.array(models_list),
                    'P-Value':np.around(np.array(p_vals),3)})
        univariate_regressions.index=univariate_regressions['CSL Model']
        univariate_regressions_transposed=univariate_regressions.drop(['CSL Model'],axis=1).T
        univariate_regressions_transposed=univariate_regressions_transposed.reindex(list(sensitivity_frame_reduced.columns),axis=1)

        fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(15,3.0))
        plt.figtext(0.01, 0.87, "d)\n\n", fontsize=14,weight='bold')
        colors = ["silver","dimgrey"]
        cmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors),gamma=1.5)

        heatmap(binary_heatmap,linewidth="1.5",linecolor="w",cmap=cmap,
                yticklabels=univariate_regressions_transposed.index,annot=univariate_regressions_transposed.values,fmt=".3f",annot_kws={"size":13},
            xticklabels=univariate_regressions_transposed.columns,cbar=True,cbar_kws={"orientation":"horizontal",
                                                                                "fraction":0.1,
                                                                                "pad":0.49},ax=ax)
        colorbar=ax.collections[0].colorbar
        colorbar.set_label(label="EBM Score <0.01            EBM Score >=0.01",fontsize=13)
        colorbar.set_ticks([])

        ax.set_xticklabels(ax.get_xticklabels(),rotation=25,fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=0)
        ax.set_ylabel("P-Value",fontsize=15)
        ax.set_xlabel("CSL Model",fontsize=15)
        fig.tight_layout()

        plt.show()
        plt.close()

        print('\tFigure Caption: a) Heatmap of EBM scores for scales colored based on condition >=0.01 or <0.01, \n \t\t\t b) & c) Significant and insignificant variation in DOS w.r.t. data scale, \n \t\t\t d) Heatmap of p-values from T-test for independence colored based on EBM scores from a).')

    def plot_appendix_additional_betas(self):
        with open(self.read_path+'NoTears_Nonlinear_betas_Results.pkl','rb') as f:
            nn_betas=pickle.load(f) 
            
        with open(self.read_path+'DAGMA_betas_Results.pkl','rb') as f:
            dagma_betas_results=pickle.load(f) 
            
        #NoTears Nonlinear:
        nn_betas_results=extract_beta_results(current_betas=nn_betas)

        #H Grouping:
        nn_h_grouping=nn_betas_results.groupby(by=['Beta_Upper_Limit'])[['H_Start','H_10%','H_20%','H_30%',
                                                        'H_40%','H_50%','H_60%','H_70%','H_80%','H_90%','H_End']].mean()
        nn_h_grouping['Beta_Upper_Limit']=nn_h_grouping.index
        nn_h_grouping=prepare_h_grouping(h_grouping=nn_h_grouping)

        #DAGMA:
        dagma_h_grouping=dagma_betas_results.groupby(by=['Beta_Upper_Limit'])[['H_Start','H_10%','H_20%','H_30%',
                                                        'H_40%','H_50%','H_60%','H_70%','H_80%','H_90%','H_End']].mean()
        dagma_h_grouping['Beta_Upper_Limit']=dagma_h_grouping.index
        dagma_h_grouping=prepare_h_grouping(h_grouping=dagma_h_grouping)

        #Lineplot and a barplot:
        colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey']
        fig, ax=plt.subplots(nrows=1,ncols=2,figsize=(15,4))
        plt.figtext(0.10, 0.92, "a)", fontsize=10,weight='bold')
        plt.figtext(0.52,  0.92, "b)", fontsize=10,weight='bold')

        for beta_value in [1,4,6,8]:#range(1,9):
            linestyle=('-' if beta_value==1 else ('--' if beta_value==4 else (':' if beta_value==6 else'-.')))
            lineplot(y='H_Values',
                    data=nn_h_grouping[nn_h_grouping['Beta_Upper_Limit']==beta_value],
                    x='Stage_Training_Process',label=str(beta_value),color='black',linestyle=linestyle,#colors[beta_value-1],
                    ax=ax[0])
            lineplot(y='H_Values',
                    data=dagma_h_grouping[dagma_h_grouping['Beta_Upper_Limit']==beta_value],
                    x='Stage_Training_Process',label=str(beta_value),color='black',linestyle=linestyle,#color=colors[beta_value-1],
                    ax=ax[1])

        #ax[0,0].set_title('NoTears Nonlinear: acyclicity constraint values \nduring different stages of training process')
        ax[0].set_xlabel('Stage of CSL Training Process',fontsize=12)
        ax[0].set_ylabel('H Values (DAG-ness)',fontsize=12)
        ax[0].legend(title='Beta_Upper_Limit',loc=[0.0,-0.33],mode='expand',ncol=4)
        ax[0].tick_params(axis='both', which='major', labelsize=10, width=2.5, length=10,rotation=0)

        #ax[0,1].set_title('DAGMA: acyclicity constraint values \nduring different stages of training process')
        ax[1].set_xlabel('Stage of CSL Training Process',fontsize=12)
        ax[1].set_ylabel('')
        ax[1].legend().remove()
        ax[1].tick_params(axis='both', which='major', labelsize=10, width=2.5, length=10,rotation=0)
        0
        fig, ax=plt.subplots(nrows=1,ncols=2,figsize=(15,5))
        plt.figtext(0.01, 0.98, "c)", fontsize=13,weight='bold')
        plt.figtext(0.52,  0.98, "d)", fontsize=13,weight='bold')

        boxplot(x='Beta_Upper_Limit',y='NumEdges_End',
                data=nn_betas_results[['Beta_Upper_Limit','NumEdges_End']][nn_betas_results['Beta_Upper_Limit'].isin([1,4,6,8])],
            palette=['black','dimgray','silver','whitesmoke'],ax=ax[0])
        #ax[1,0].set_title('NoTears Nonlinear: Number of edges of estimated DAG\nfor different beta upper limits')
        ax[0].set_xlabel('Beta Upper Limit',fontsize=14)
        ax[0].set_ylabel('Edges Number in\n last Training Iteration',fontsize=14)
        ax[0].tick_params(axis='both', which='major', labelsize=12, width=2.5, length=10,rotation=0)

        boxplot(x='Beta_Upper_Limit',y='NumEdges_End',
                data=dagma_betas_results[['Beta_Upper_Limit','NumEdges_End']][dagma_betas_results['Beta_Upper_Limit'].isin([1,4,6,8])],
            palette=['black','dimgray','silver','whitesmoke'],ax=ax[1])
        #ax[1,1].set_title('DAGMA: Number of edges of estimated DAG\nfor different beta upper limits')
        ax[1].set_xlabel('Beta Upper Limit',fontsize=14)
        ax[1].set_ylabel('',fontsize=14)
        ax[1].tick_params(axis='both', which='major', labelsize=12, width=2.5, length=10,rotation=0)


        fig.tight_layout()
        plt.show()
        plt.close()

        print('\tFigure Caption: a) & b) DAG-ness of the estimated adjacency matrices in different stages of the models training \n \t\t\tof NoTears-MLP and DAGMA, respectivelly, \n \t\t\t c) & d) Number of estimated edges at the end of the training process of NoTears-MLP and DAGMA,\n \t\t\t respecitvelly.')

    def plot_appendix_ranking(self):
        #Redundant: to replace:
        conditioned_grouped=self.all_csl_regrouped.copy()#all_csl_regrouped[(all_csl_regrouped['DOS']>=all_csl_regrouped.DOS.quantile(include_quantiles[0])) & ((all_csl_regrouped['DOS']<=all_csl_regrouped.DOS.quantile(include_quantiles[1])))]
        conditioned_grouped=conditioned_grouped[conditioned_grouped["Scale"]!="original"]

        rankings=[conditioned_grouped.groupby(by=['CSL_Model'])[['FScore']].mean().sort_values(by='FScore',
                ascending=False),
                conditioned_grouped.groupby(by=['CSL_Model'])[['TPR']].mean().sort_values(by='TPR',
                ascending=False),
                conditioned_grouped.groupby(by=['CSL_Model'])[['FPR']].mean().sort_values(by='FPR',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['Causal_Order_Divergence']].mean().sort_values(by='Causal_Order_Divergence',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['nSHD']].mean().sort_values(by='nSHD',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['nSID']].mean().sort_values(by='nSID',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['DOS']].mean().sort_values(by='DOS',
                ascending=False)]

        dos_ranking=rankings[-1].copy()
        ####################################################################################################


        conditioned_grouped=self.all_csl_regrouped.copy()#all_csl_regrouped[(all_csl_regrouped['DOS']>=all_csl_regrouped.DOS.quantile(include_quantiles[0])) & ((all_csl_regrouped['DOS']<=all_csl_regrouped.DOS.quantile(include_quantiles[1])))]
        conditioned_grouped=conditioned_grouped[conditioned_grouped["Scale"]=="original"]

        rankings=[conditioned_grouped.groupby(by=['CSL_Model'])[['FScore']].mean().sort_values(by='FScore',
                ascending=False),
                conditioned_grouped.groupby(by=['CSL_Model'])[['TPR']].mean().sort_values(by='TPR',
                ascending=False),
                conditioned_grouped.groupby(by=['CSL_Model'])[['FPR']].mean().sort_values(by='FPR',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['Causal_Order_Divergence']].mean().sort_values(by='Causal_Order_Divergence',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['nSHD']].mean().sort_values(by='nSHD',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['nSID']].mean().sort_values(by='nSID',
                ascending=True),
                conditioned_grouped.groupby(by=['CSL_Model'])[['DOS']].mean().sort_values(by='DOS',
                ascending=False)]

        dos_ranking_original=rankings[-1].copy()

        fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,5.7))
        plt.figtext(0.01, 0.99, "a)", fontsize=10,weight='bold')
        plt.figtext(0.51, 0.99, "b)", fontsize=10,weight='bold')

        barplot(x=dos_ranking["DOS"],y=dos_ranking.index,ax=ax[0],
        orientation="horizontal",edgecolor="0.01",
                palette=['dimgray','silver','dimgray',
                        'dimgray','dimgray','dimgray','silver',
                        'silver','silver','dimgray',
                        'silver','silver','dimgray','silver'])
        ax[0].set_ylabel("CSL Model",fontsize=10)
        ax[0].set_xlabel("DOS",fontsize=10)
        ax[0].tick_params(axis='both', which='major', labelsize=9, width=2.5, length=10,rotation=0)

        ax[0].set(xlim=(0, 0.72))
        for index, row in dos_ranking.iterrows():
            ax[0].text(row.DOS,np.where(dos_ranking.index==row.name)[0][0], round(row.DOS, 3),
                    color='black', horizontalalignment='left',fontsize=10)

        leg=ax[0].legend(title='Optmization Type',
                    labels=["Continuous","Combinatorial"],
                    mode="expand",ncol=2,
                    loc=(0.01,-0.25),fontsize=9,title_fontsize=9)
        leg=ax[0].get_legend()
        leg.legendHandles[0].set_color('silver')
        leg.legendHandles[1].set_color('dimgray')


        barplot(x=dos_ranking_original["DOS"],y=dos_ranking_original.index,ax=ax[1],
        orientation="horizontal",edgecolor="0.01",
                palette=['silver','silver','dimgray',
                        'silver','silver','silver',
                        'silver','dimgray','dimgray',
                        'dimgray','dimgray','dimgray','dimgray','silver'])
        ax[1].set_ylabel("",fontsize=10)
        ax[1].set_xlabel("DOS",fontsize=10)
        ax[1].tick_params(axis='both', which='major', labelsize=9, width=2.5, length=10,rotation=0)

        ax[1].set(xlim=(0, 0.82))
        for index, row in dos_ranking_original.iterrows():
            ax[1].text(row.DOS,np.where(dos_ranking_original.index==row.name)[0][0], round(row.DOS, 3),
                    color='black', horizontalalignment='left',fontsize=10)


        fig.tight_layout()
        plt.show()
        plt.close()

        print('\tFigure Caption: a) & b) Achieved DOS values on average on standardized and original scale, respectively')
