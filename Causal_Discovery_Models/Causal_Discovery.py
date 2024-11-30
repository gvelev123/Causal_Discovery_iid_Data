import sys
sys.path.append("..")

from Causal_Discovery_Models.DAS import DAS, make_context
from Causal_Discovery_Models.R2_SortnRegress import r2_sort_regress
from Causal_Discovery_Models.NoTears_Linear import notears_linear,TraceExpm
from Causal_Discovery_Models.NoCurl import BPR
from Causal_Discovery_Models.DAGMA import DagmaNonlinear, DagmaMLP
from castle.algorithms import DirectLiNGAM, GOLEM, NotearsNonlinear, PC
from castle.algorithms.gradient.gran_dag.torch.gran_dag import GraNDAG
import avici
from dask.distributed import Client
import pickle
import torch
import numpy as np
import networkx as nx

class Causal_Discovery():
    def __init__(self,
                #models_to_run,
                frames_list,
                index_frame_description,
                index_true_adjacency,
                index_true_weighted_adjacency,
                index_frame):
        
        #self.models_to_run=models_to_run
        self.frames_list=frames_list
        self.index_frame_description=index_frame_description
        self.index_true_adjacency=index_true_adjacency
        self.index_true_weighted_adjacency=index_true_weighted_adjacency
        self.index_frame=index_frame
        self.causal_discovery_results=None
        
        
        super(Causal_Discovery, self).__init__()
        
    def set_causal_discovery_results(self,current_csl_results):
        self.causal_discovery_results=current_csl_results
    
    def estimate_causal_matrix(self,data,model):
          if model=='DIRECT-LINGAM':
            #improved version of ICA-Lingam in terms of convergence for pre-defined number of steps (2014)
            directln = DirectLiNGAM()
            directln.learn(data)
            causal_matrix=directln.causal_matrix

          elif model=='NoTears':
            #Linear version with first time continuous relaxation of acyclicity constraint (2018)
            trace_expm = TraceExpm.apply
            W_est = notears_linear(data, lambda1=0.1, loss_type='l2')
            causal_matrix=np.where(np.abs(W_est)>0.3,1.0,0.0)

          elif model=='NoTears_Nonlinear':
            #Nonlinear version (2020)
            nts_nonlinear_model=NotearsNonlinear(device_type='cpu')
            nts_nonlinear_model.learn(data=data)
            causal_matrix=nts_nonlinear_model.causal_matrix

          elif model=='GOLEM':
            #CSL model testing also non-equal variances in SEM based on sparse DAGs (2020)
            golem= GOLEM()
            golem.learn(data,lambda_1=2e-2, lambda_2=5.0, equal_variances=True)
            causal_matrix=golem.causal_matrix

          elif model=='NoCurl':
            #CSL model estimating causal DAGs with gradient function in PMLR (2021)
            bpr = BPR(rho_max=1e+16,
                      h_tol=1e-8,
                      lambda1=1000.,
                      lambda2=1000.,
                      train_epochs=1e4,
                      graph_threshold=0.3)
            A, h, alpha, rho = bpr.fit(data, 'nocurl')
            causal_matrix=np.where(A!=0.0,1,0)

          elif model=='DAGMA':
            #CSL Model with new formulation of acyclicity (2022)
            eq_model = DagmaMLP(dims=[data.shape[1], 10, 1], bias=True, dtype=torch.double) # create the model for the structural equations, in this case MLPs
            dagma_model = DagmaNonlinear(eq_model, dtype=torch.double) # create the model for DAG learning
            W_est = dagma_model.fit(data, lambda1=0.02, lambda2=0.005)
            causal_matrix=np.where(W_est!=0.0,1.0,0.0)

          elif model=='PC_Stable':
            pc_model=PC(variant='stable')
            pc_model.learn(data)
            causal_matrix=pc_model.causal_matrix

          elif model=='R2_SortnRegress':
            causal_matrix=1.0*(r2_sort_regress(data)!=0)

          elif model=='Gran_DAG':
            gnd=GraNDAG(input_dim=data.shape[1])
            gnd.learn(data=data)
            causal_matrix=gnd.causal_matrix

          elif model=='DAS':
            #only DAS takes for inputs the pandas dataframe, all other methods take multidimensional np-array. 
            context = make_context().variables(data.columns).build()
            das_model=DAS()
            das_model.learn_graph(data,context)
            causal_matrix=nx.adjacency_matrix(das_model.graph_).todense()

          else:
            print('Selected Model not in available functionalities!')
            return None

          return np.array(causal_matrix)
        
    def test_csl_method(self,data_item,**kwargs):
        frame_description=data_item[self.index_frame_description]#0]
        current_adjacency_matrix=data_item[self.index_true_adjacency]#1]
        current_weighted_adjacency=data_item[self.index_true_weighted_adjacency]#2]
        current_dataframe=data_item[self.index_frame]#3]
        
        estimated_adjacencies_dict={'DAGMA':self.estimate_causal_matrix(data=current_dataframe.values,
                                                       model='DAGMA'),
                                   'NoTears_Nonlinear':self.estimate_causal_matrix(data=current_dataframe.values,
                                                       model='NoTears_Nonlinear'),
                                   'PC_stable':self.estimate_causal_matrix(data=current_dataframe.values,
                                                       model='PC_Stable'),
                                   'DAS':self.estimate_causal_matrix(data=current_dataframe,
                                                       model='DAS'),
                                   'NoCurl':self.estimate_causal_matrix(data=current_dataframe.values,
                                                       model='NoCurl'),
                                   'GOLEM':self.estimate_causal_matrix(data=current_dataframe.values,
                                                       model='GOLEM'),
                                   'NoTears':self.estimate_causal_matrix(data=current_dataframe.values,
                                                       model='NoTears'),
                                   'DIRECT-LINGAM':self.estimate_causal_matrix(data=current_dataframe.values,
                                                       model='DIRECT-LINGAM'),
                                   'Gran-DAG':self.estimate_causal_matrix(data=current_dataframe.values,
                                                       model='Gran_DAG'),
                                   'R2SortnRegress':self.estimate_causal_matrix(data=current_dataframe.values,
                                                       model='R2_SortnRegress')}
                
        return [frame_description,current_adjacency_matrix,current_weighted_adjacency,current_dataframe,estimated_adjacencies_dict]
    
    def extract_causal_graphs(self):
        client=Client()
        print('DASK Client Dashboard Link: ',client.dashboard_link)
        #if self.models_to_run=='all':
        futures = client.map(self.test_csl_method,self.frames_list)
        csl_results = client.gather(futures)

        #Additioanally run AVICI without parallelization: the approach represents a pre-trained transformer
        #loading a pre-trained model each time a parallel run is executed can exghaust the computational ressources
        #thus, since the approach does not require any training power, just simply execute AVICI in a manual loop:
        model_avici=avici.load_pretrained(download="scm-v0")
        for csl_res in csl_results:
            csl_res[-1]['AVICI']=model_avici(x=csl_res[self.index_frame].values,return_probs=False)
        
        self.set_causal_discovery_results(current_csl_results=csl_results)
        
        
    def save_csl_results(self,save_path,save_for_evaluation):
      if save_for_evaluation==False:
        with open(save_path,'wb') as f:
            pickle.dump(self.causal_discovery_results,f)
      else:
        dict_for_evaluation={'Large_Sample_Size':{'10_nodes':[],
                                                  '20_nodes':[],
                                                  '50_nodes':[],
                                                  '100_nodes':[]},
                             'Small_Sample_Size':{'10_nodes':[],
                                                  '20_nodes':[],
                                                  '50_nodes':[],
                                                  '100_nodes':[]}}
        for csl_item in self.causal_discovery_results:
          if csl_item[self.index_frame].shape[0]==2500:
            dict_for_evaluation['Large_Sample_Size'][str(csl_item[self.index_frame_description][1])+'_nodes'].append(csl_item)
          else:
            dict_for_evaluation['Small_Sample_Size'][str(csl_item[self.index_frame_description][1])+'_nodes'].append(csl_item)

      with open(save_path[:-4]+'_Evaluation.pkl','wb') as f:
          pickle.dump(dict_for_evaluation,f)  
