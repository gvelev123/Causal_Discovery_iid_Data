from interpret.glassbox import ExplainableBoostingRegressor
import pickle
import pandas as pd
import numpy as np

class EBM():
    def __init__(self,
                results_dataframe,
                avg_number_edges,
                target_column,
                models_column,
                dataset_characteristics,
                save_path,
                graph_type_column=None,
                connectivity_column=None,
                nodes_column=None):
        
        self.results_dataframe=results_dataframe
        self.avg_number_edges=avg_number_edges

        self.target_column=target_column
        self.models_column=models_column
        self.dataset_characteristics=dataset_characteristics

        #These parameters will only be provided if it is necessary to perform a mapping between ER and SF graph connecitivity:
        self.graph_type_column=graph_type_column
        self.connectivity_column=connectivity_column
        self.nodes_column=nodes_column

        self.save_path=save_path
        self.ebm_explanations=None
        
        super(EBM,self).__init__()
        
    def map_er_sf_connectivity(self):
        
        mapped_connectivity=[]
        for row_idx in range(0,self.results_dataframe.shape[0]):
            current_graph_type=self.results_dataframe.iloc[row_idx,:][self.graph_type_column]
            current_connectivity=self.results_dataframe.iloc[row_idx,:][self.connectivity_column]

            if current_graph_type=='ER':#no overwriting of connectivity as already means edge density in %
                mapped_connectivity.append(current_connectivity)
            else:
                current_number_nodes=self.results_dataframe.iloc[row_idx,:][self.nodes_column]
                nodes_mapping=self.avg_number_edges['Nodes_'+str(current_number_nodes)]
                if nodes_mapping[0.2]['k']==current_connectivity:
                    mapped_connectivity.append(0.2)
                elif nodes_mapping[0.3]['k']==current_connectivity:
                    mapped_connectivity.append(0.3)
                else:
                    mapped_connectivity.append(0.4)

        self.results_dataframe[self.connectivity_column]=np.array(mapped_connectivity)
        return self.results_dataframe
        
    def estimate_EBM_scores(self):
        
        csl_models=list(self.results_dataframe[self.models_column].value_counts().keys())

        ebm_explanations={}

        for csl_model in csl_models:
        
            #Get results for current model:
            current_csl_model=self.results_dataframe[self.results_dataframe[self.models_column]==csl_model]

            #Drop runs, which have resulted into nans:
            current_csl_model=current_csl_model.dropna()

            #Get train features and target:
            current_train_features=current_csl_model[self.dataset_characteristics]
            current_train_target=current_csl_model[self.target_column]

            #Train the EBM model:
            current_ebm = ExplainableBoostingRegressor()
            current_ebm.fit(X=current_train_features,
                            y=current_train_target)
            extracted_importances=current_ebm.explain_global()
            current_explanations_ebm=pd.DataFrame({'Simulation_Component':extracted_importances.data()['names'],
                                                  'Importance_Scores':extracted_importances.data()['scores'],
                                                  'Importance_Types':extracted_importances.feature_types})
            #Save extracted explanations in dictionary:
            ebm_explanations[csl_model]=current_explanations_ebm

        ebm_frames_merge=[]

        for model_key in ebm_explanations.keys():

            model_frame=ebm_explanations[model_key]
            model_frame[self.models_column]=model_key
            ebm_frames_merge.append(model_frame)

        ebm_merge=pd.concat(ebm_frames_merge)

        with open(self.save_path+'EBM_explanations.pkl','wb') as f:
            pickle.dump(ebm_merge,f)
            
        self.ebm_explanations=ebm_merge