import igraph as ig
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from dask.distributed import Client
from cdt.metrics import SID
import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.1"
from rpy2 import robjects

class DOS():
    def __init__(self,
                dataset_description_index,
                true_binary_adjacency_index,
                estimated_adjacencies_index,
                fpr_metric,
                sample_sizes,
                number_nodes,
                model_names,
                metrics,
                best_scenario,
                worst_scenario,
                #parallelization,
                read_path,
                read_file_names,
                save_path,
                save_file_names):
        
        self.dataset_description_index=dataset_description_index
        self.true_binary_adjacency_index=true_binary_adjacency_index
        self.estimated_adjacencies_index=estimated_adjacencies_index
        self.fpr_metric=fpr_metric
        self.sample_sizes=sample_sizes
        self.number_nodes=number_nodes
        self.model_names=model_names
        self.metrics=metrics
        self.best_scenario=best_scenario
        self.worst_scenario=worst_scenario
        #self.parallelization=parallelization
        self.read_path=read_path
        self.read_file_names=read_file_names
        self.save_path=save_path
        self.save_file_names=save_file_names
        
        #Placeholder results variables, which will be filled while executing evaluate function:
        #self.er_results_evaluation=None
        #self.sf_results_evaluation=None
        self.all_results_evaluations=[]
        self.results_dataframe=None
        
        super(DOS, self).__init__()


    def eliminateCycles_CausalOrder(self,
                                    dag_false_adjacency):
        #Input values: 
        # dag_false_adjacency

        #Output values:
        # adjacency without any cycles
        # number of edges removed
        # estimated topological order with nx

        #Double check if directed graph not a DAG:
        dag_false_G=nx.DiGraph()
        #Add nodes:
        for node in range(0,dag_false_adjacency.shape[0]):
            dag_false_G.add_node(node)

        #Add edges:
        for cause in range(0,dag_false_adjacency.shape[0]):
            for effect in range(0,dag_false_adjacency.shape[1]):
                if dag_false_adjacency[cause,effect]==1:
                    dag_false_G.add_edge(cause,effect)

        #Start removing cycles:
        number_edges_before_cycle_removal=len(list(dag_false_G.edges))
        if nx.is_directed_acyclic_graph(dag_false_G)==False:
            keep_removing_cycles=True
            while keep_removing_cycles:
                resulting_cycle=nx.find_cycle(dag_false_G,orientation='original')
                #Remove cylce by removing the last edge from the cycle:
                dag_false_G.remove_edge(resulting_cycle[-1][0],resulting_cycle[-1][1])
                if nx.is_directed_acyclic_graph(dag_false_G)==True:
                    keep_removing_cycles=False

        #Compute number of removed edges:
        number_edges_after_cycle_removal=len(list(dag_false_G.edges))
        number_edges_removed=number_edges_before_cycle_removal-number_edges_after_cycle_removal
        #print('Number of edges removed to eliminate all Cycles: ',number_edges_removed)

        #Compute estimated binary adjacency matrix without cycles:
        estimated_adjacency_noCycles=np.zeros(shape=(dag_false_adjacency.shape[0],dag_false_adjacency.shape[1]))
        edges_after_cycle_removal=list(dag_false_G.edges())
        for edge_pair in edges_after_cycle_removal:
            estimated_adjacency_noCycles[edge_pair[0],edge_pair[1]]
        #print('Adjacency is DAG after cycle removal: ',is_dag(estimated_adjacency_noCycles))

        #Compute causal (topological) order of DAG after cycle removal:
        estimated_causal_order=list(nx.topological_sort(dag_false_G))
        #all_causal_order_sorts=list(nx.all_topological_sorts(dag_false_G))


        return {'estimated_binary_adjacency_no_cycles':estimated_adjacency_noCycles,
               'number_edges_removed':number_edges_removed,
               'estimated_causal_order':estimated_causal_order}

    def estimate_causal_order(self,estimated_binary_adjacency):
        estimated_dag=nx.DiGraph()
        #Add nodes:
        for node in range(0,estimated_binary_adjacency.shape[0]):
            estimated_dag.add_node(node)

        #Add edges:
        for cause in range(0,estimated_binary_adjacency.shape[0]):
            for effect in range(0,estimated_binary_adjacency.shape[1]):
                if estimated_binary_adjacency[cause,effect]==1:
                    estimated_dag.add_edge(cause,effect)

        #Estimate causal order:
        estimated_causal_order=list(nx.topological_sort(estimated_dag))
        return estimated_causal_order

    def toporder_divergence(self,true_binary_adjacency, order):
        """Compute topological ordering divergence.

        Topological order divergence is used to compute the number of false negatives,
        i.e. missing edges, associated to a topological order of the nodes of a
        graph with respect to the ground truth structure.
        If the topological ordering is compatible with the graph ground truth,
        the divergence is equal to 0. In the worst case of completely reversed
        ordering, toporder_divergence is equals to P, the number of edges (positives)
        in the ground truth graph.
        Note that the divergence defines a lower bound for the Structural Hamming Distance.

        Parameters
        ----------
        true_graph : NetworkxGraph
            Input groundtruth directed acyclic graph.
        order : List[int]
            A topological ordering on the nodes of the graph.

        Returns
        -------
        err : %
            Sum of the number of edges of A not admitted by the given order divided by total number of edges in ture DAG.
        """
        #if not nx.is_directed_acyclic_graph(true_graph):
        if not self.is_dag(true_binary_adjacency):
            raise ValueError("The input graph must be directed and acyclic.")

        # convert graphs to adjacency matrix in numpy array format
        A = true_binary_adjacency #nx.to_numpy_array(true_graph)

        if len(order) != A.shape[0] or A.shape[0] != A.shape[1]:
            raise ValueError("The dimensions of the graph and the order list do not match.")

        false_negatives_from_order = 0
        for i in range(len(order)):
            false_negatives_from_order += A[order[i + 1 :], order[i]].sum()

        #Compute all edges in true DAG:
        number_all_true_edges=len(np.where(A.flatten()==1)[0])

        #Compute percentual causal order divergence:
        percentual_causal_order_divergence=false_negatives_from_order/number_all_true_edges

        return percentual_causal_order_divergence #false_negatives_from_order

    def is_dag(self,W):
        G = ig.Graph.Weighted_Adjacency(W.tolist())
        return G.is_dag()

    def count_accuracy(self,B_true, B_est):#,
                       #SID_function=None):
        """Compute various accuracy metrics for B_est.

        true positive = predicted association exists in condition in correct direction
        reverse = predicted association exists in condition in opposite direction
        false positive = predicted association does not exist in condition

        Args:
            B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
            B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

        Returns:
            fdr: (reverse + false positive) / prediction positive
            tpr: (true positive) / condition positive
            fpr: (reverse + false positive) / condition negative
            shd: undirected extra + undirected missing + reverse
            nnz: prediction positive
        """
        if (B_est == -1).any():  # cpdag
            if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
                raise ValueError('B_est should take value in {0,1,-1}')
            if ((B_est == -1) & (B_est.T == -1)).any():
                raise ValueError('undirected edge should only appear once')
        else:  # dag
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
            #if not is_dag(B_est):
            #    raise ValueError('B_est should be a DAG')


        #1. FDR, FPR, TPR Computation: ######################################################################### 
        d = B_true.shape[0]
        # linear index of nonzeros
        pred_und = np.flatnonzero(B_est == -1)
        pred = np.flatnonzero(B_est == 1)

        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])

        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])

        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        #cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        cond_neg_size = d * (d - 1) - len(cond)

        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        #FPR Condition: ######################################################
        if self.fpr_metric=='NoTears':
            fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        else:
            cmat=confusion_matrix(y_true=B_true.flatten(),
                                  y_pred=B_est.flatten())
            fpr = cmat[0,1] / (cmat[0,1] + cmat[0,0])
        #FDR, FPR, TPR done: #####################################################################################


        #2. Overall causal discovery performance indicators: structural hamming distance and F1 Score ############
        pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
        cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True) 
        shd = len(extra_lower) + len(missing_lower) + len(reverse)

        #compute denominator for normalized shd:
        num_edges_estimation=len(np.where(B_est.flatten()!=0.0)[0])
        num_edges_true=len(np.where(B_true.flatten()!=0.0)[0])
        normalized_shd=shd/(num_edges_estimation+num_edges_true)

        #compute f1_score:
        f1_score_result=f1_score(y_true=B_true.flatten(),y_pred=B_est.flatten())
        #SHD and F1 done: ######################################################################################


        #3. COD Computation: ####################################################################################
        #Perform cycle elimination for COD only after computing all other metrics
        #In this way the condition for cycle-free estimation affects only those metrics, which indeed assume
        #the computed adjacency matric is acyclic.
        dag_check=self.is_dag(B_est)

        if dag_check==True:#if estimated graph is a DAG, then no need to eliminate cycles to compute causal order
            est_causal_order=self.estimate_causal_order(estimated_binary_adjacency=B_est)
            number_eliminated_edges=0
        else:
            causal_order_results=self.eliminateCycles_CausalOrder(dag_false_adjacency=B_est)
            #Overwrite B_est to further compute all metrics on the estimated DAG after cycles removal
            B_est=causal_order_results['estimated_binary_adjacency_no_cycles']
            number_eliminated_edges=causal_order_results['number_edges_removed']
            #Compute percentage eliminated edges from all estimated edges:

            est_causal_order=causal_order_results['estimated_causal_order']


        causal_order_divergence=self.toporder_divergence(true_binary_adjacency=B_true, order=est_causal_order)
        #Compute penalized COD with % eliminated edges:
        #COD done: ##############################################################################################


        #4. SID Computation: #####################################################################################
        #SID can be computed only after checking that there are no cycles remaining in the graph
        #thus, this comes after having computed COD, which also requires the estimation to be a DAG
        #after eliminating all remaining cycles in the estimation:
        #if SID_function==None:
        interventional_distance=SID(target=B_true,
                                       pred=B_est)
        #else:
        #interventional_distance=SID_function(target=B_true,
        #                               pred=B_est)

        denominator_interventional_distance=(d*(d-1))
        normalized_interventional_distance=interventional_distance/denominator_interventional_distance
        #SID done: ################################################################################################

        return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr,
                #Metric quantifying overall causal discovery performance:
                'shd': shd,'normalized_shd':normalized_shd,'f1_score':f1_score_result,
                'nnz': pred_size,
                #Metric, which require a DAG estimation:
                'dag_check':dag_check,'eliminated_edges':number_eliminated_edges,
                'est_causal_order':est_causal_order,
                'causal_order_divergence':causal_order_divergence,
                'sid': interventional_distance,
                'normalized_sid':normalized_interventional_distance}

    def compute_ranking_multiCriteriaMetrics(self,worst_scenario,best_scenario,current_result):
        #TOPSIS 5. step: calculate the euclidean distance from the worst and the best scenario
        dist_worst=np.linalg.norm(current_result-worst_scenario)
        dist_best=np.linalg.norm(current_result-best_scenario)
        #6. step: compute distance to optimal solution
        similarity_score_optimal=dist_worst/(dist_worst+dist_best)

        return similarity_score_optimal


    def DOS_computation(self,current_csl_item,**kwargs):
        #dataset_description_index=kwargs['special_parameters']['dataset_description_index']
        true_binary_adjacency_index=kwargs['special_parameters']['true_binary_adjacency_index']
        estimated_adjacencies_index=kwargs['special_parameters']['estimated_adjacencies_index']
        
        metrics=kwargs['special_parameters']['metrics']
        best_scenario=kwargs['special_parameters']['best_scenario']
        worst_scenario=kwargs['special_parameters']['worst_scenario']
        model_names=kwargs['special_parameters']['model_names']


        #current_dataset_description=current_csl_item[dataset_description_index]
        current_true_binary_adjacency=current_csl_item[true_binary_adjacency_index]
        current_estimated_adjacencies=current_csl_item[estimated_adjacencies_index]

        current_models_results={}

        for model_name in model_names:
            current_models_results[model_name]={}
            #print('-- Curent CSL Model: ',model_name)
            est_model=current_estimated_adjacencies[model_name]

            #Dataset descriptions:
            #0: seed, #1: nodes, #2: connectivity (for ER graphs: edge density, for SF graphs: k)
            #3: number edges, #4: varsortability,#5: noise std
            #6: Transformation_Function,#7: Beta Upper Limit,#8: Scale

            #Save results: ################################################################
            #Save metrics for graph estimations:
            if type(est_model)==np.ndarray:
                current_res=self.count_accuracy(B_true=current_true_binary_adjacency,
                                              B_est=est_model)

                current_models_results[model_name]['TPR']=current_res['tpr']
                current_models_results[model_name]['FPR']=current_res['fpr']
                current_models_results[model_name]['normalized_shd']=current_res['normalized_shd']
                current_models_results[model_name]['f1_score']=current_res['f1_score']
                current_models_results[model_name]['causal_order_divergence']=current_res['causal_order_divergence']
                current_models_results[model_name]['normalized_sid']=current_res['normalized_sid']

                current_models_results[model_name]['dag_check']=current_res['dag_check']
                current_models_results[model_name]['eliminated_edges']=current_res['eliminated_edges']

                current_res=np.array([current_res[metrics[0]],current_res[metrics[1]],
                                         current_res[metrics[2]],current_res[metrics[3]],
                                         current_res[metrics[4]],current_res[metrics[5]] ])

                current_ranking_res=self.compute_ranking_multiCriteriaMetrics(worst_scenario=worst_scenario,
                                                                         best_scenario=best_scenario,
                                                                         current_result=current_res)
                current_models_results[model_name]['DOS']=current_ranking_res
            else:
                current_models_results[model_name]['TPR']=np.nan
                current_models_results[model_name]['FPR']=np.nan
                current_models_results[model_name]['normalized_shd']=np.nan
                current_models_results[model_name]['f1_score']=np.nan
                current_models_results[model_name]['causal_order_divergence']=np.nan
                current_models_results[model_name]['normalized_sid']=np.nan
                current_models_results[model_name]['dag_check']=np.nan
                current_models_results[model_name]['eliminated_edges']=np.nan
                current_models_results[model_name]['DOS']=np.nan

        current_csl_item.append(current_models_results)
        return current_csl_item

    def execute_DOS_computation(self,current_csl_item,**kwargs):
        parallelization=kwargs['special_parameters']['parallelization']
        if parallelization==True:
            not_executed=True
            while not_executed:
                try:
                    from cdt.metrics import SID
                    current_csl_item=self.DOS_computation(current_csl_item,**kwargs)
                    not_executed=False
                except:
                    not_executed=True
        else:
            current_csl_item=self.DOS_computation(current_csl_item,**kwargs)
        return current_csl_item

    def evaluate(self):
        
        for read_file_name_idx in range(0,len(self.read_file_names)):
            read_file_name=self.read_file_names[read_file_name_idx]
            save_file_name=self.save_file_names[read_file_name_idx]

            with open(self.read_path+read_file_name,'rb') as f:
                current_results=pickle.load(f)
            #with open(self.read_path+'SF_CSL_Results.pkl','rb') as f:
            #    sf_results=pickle.load(f)
                
            current_results_evaluation={}
            #sf_results_evaluation={}

            for sample_size in self.sample_sizes:
                current_results_evaluation[sample_size]={}
                #sf_results_evaluation[sample_size]={}

                print('Current Sample Size: ',sample_size,'\n')
                for nr_nodes in self.number_nodes:
                    print('Current Number Nodes: ',nr_nodes,'\n')

                    current_items=current_results[sample_size][nr_nodes]
                    #sf_items=sf_results[sample_size][nr_nodes]

                    #ER Graphs Evaluation:
                    '''
                    if self.parallelization==True:
                        client=Client(n_workers=5,threads_per_worker=10)
                        print('Dashboard Link of initialized DASK Client (ER Graphs): ',client.dashboard_link)
                        futures_items = client.map(self.execute_DOS_computation,current_items,
                                            special_parameters={#'dataset_description_index':dataset_description_index,
                                                                'true_binary_adjacency_index':self.true_binary_adjacency_index,
                                                                'estimated_adjacencies_index':self.estimated_adjacencies_index,
                                                                'fpr_metric':self.fpr_metric,
                                                                'metrics':self.metrics,
                                                                'best_scenario':self.best_scenario,
                                                                'worst_scenario':self.worst_scenario,
                                                                'model_names':self.model_names,
                                                            'parallelization':self.parallelization}) 
                        current_items_evaluation = client.gather(futures_items)
                        client.close()


                        #SF Graphs Evaluation:
                        #client_sf=Client(n_workers=5,threads_per_worker=10)
                        #print('Dashboard Link of initialized DASK Client (SF Graphs): ',client_sf.dashboard_link)
                        #futures_sf = client_sf.map(self.execute_DOS_computation,sf_items,
                        #                    special_parameters={'dataset_description_index':self.dataset_description_index,
                        #                                        'true_binary_adjacency_index':self.true_binary_adjacency_index,
                        #                                        'estimated_adjacencies_index':self.estimated_adjacencies_index,
                        #                                        'fpr_metric':self.fpr_metric,
                        #                                        'metrics':self.metrics,
                        #                                        'best_scenario':self.best_scenario,
                        #                                        'worst_scenario':self.worst_scenario,
                        #                                        'model_names':self.model_names,
                        #                                    'parallelization':self.parallelization})

                        #sf_items_evaluation = client_sf.gather(futures_sf)
                        #client_sf.close()
                    else:
                    '''
                    kwargs={'special_parameters':{'dataset_description_index':self.dataset_description_index,
                                                            'true_binary_adjacency_index':self.true_binary_adjacency_index,
                                                            'estimated_adjacencies_index':self.estimated_adjacencies_index,
                                                            #'fpr_metric':self.fpr_metric,
                                                            'metrics':self.metrics,
                                                            'best_scenario':self.best_scenario,
                                                            'worst_scenario':self.worst_scenario,
                                                            'model_names':self.model_names,
                                                        'parallelization':False}}#self.parallelization}}
                    current_items_evaluation=[]
                    #sf_items_evaluation=[]

                    for item_idx in range(0,len(current_items)):
                        print('- Current CSL Result Index: ',(item_idx+1),'from a total of ',len(current_items))
                        current_current_item=current_items[item_idx]
                        #current_sf_item=sf_items[item_idx]

                        current_items_evaluation.append(self.execute_DOS_computation(current_current_item,**kwargs))
                        #sf_items_evaluation.append(self.execute_DOS_computation(current_sf_item,**kwargs))

                    current_results_evaluation[sample_size][nr_nodes]=current_items_evaluation
                    #sf_results_evaluation[sample_size][nr_nodes]=sf_items_evaluation


            with open(self.save_path+save_file_name,'wb') as f:
                pickle.dump(current_results_evaluation,f)
            self.all_results_evaluations.append(current_results_evaluation)
            #self.er_results_evaluation=er_results_evaluation

            #with open(self.save_path+'SF_CSL_Results_Evaluation.pkl','wb') as f:
            #    pickle.dump(sf_results_evaluation,f)
            #self.sf_results_evaluation=sf_results_evaluation
        
    def create_results_dataframe(self,save_results_frame_name):  

        seeds,nodes,connectivity,edges,tr_function,beta_upper,scale,gr_type,s_size,csl_models=[],[],[],[],[],[],[],[],[],[]
        results_TPR=[]
        results_FPR=[]
        results_nSHD=[]
        results_FScore=[]
        results_COD=[]
        results_nSID=[]
        results_DOS=[]
        dag_checks=[]
        eliminated_edges_per_estimation=[]

        
        for sample_size in self.sample_sizes:
            for nr_nodes in self.number_nodes:
                
                for current_evaluation in self.all_results_evaluations:
                #er_items_evaluation=self.er_results_evaluation[sample_size][nr_nodes]
                #sf_items_evaluation=self.sf_results_evaluation[sample_size][nr_nodes]
                    current_evaluation_items=current_evaluation[sample_size][nr_nodes]

                    for item_idx in range(0,len(current_evaluation_items)):
                        current_evaluation_item=current_evaluation_items[item_idx]
                        #current_sf_item=sf_items_evaluation[item_idx]

                        current_dataset_description=current_evaluation_item[self.dataset_description_index]
                        #current_dataset_description_sf=current_sf_item[self.dataset_description_index]

                        current_csl_results=current_evaluation_item[-1]
                        #current_evaluation_sf=current_sf_item[-1]


                        for model_name in self.model_names:
                            est_model_metrics=current_csl_results[model_name]
                            #sf_est_model_metrics=current_evaluation_sf[model_name]

                            #Dataset descriptions:
                            #0: seed, #1: nodes, #2: connectivity (for ER graphs: edge density, for SF graphs: k)
                            #3: number edges, #4: varsortability,#5: noise std
                            #6: Transformation_Function,#7: Beta Upper Limit,#8: Scale

                            #Save results for ER: ################################################################
                            #Save metrics for ER graph estimations:
                            results_TPR.append(est_model_metrics['TPR'])
                            results_FPR.append(est_model_metrics['FPR'])
                            results_nSHD.append(est_model_metrics['normalized_shd'])
                            results_FScore.append(est_model_metrics['f1_score'])
                            results_COD.append(est_model_metrics['causal_order_divergence'])
                            results_nSID.append(est_model_metrics['normalized_sid'])
                            results_DOS.append(est_model_metrics['DOS'])
                            dag_checks.append(est_model_metrics['dag_check'])
                            eliminated_edges_per_estimation.append(est_model_metrics['eliminated_edges'])

                            seeds.append(current_dataset_description[0])
                            nodes.append(current_dataset_description[1])
                            connectivity.append(current_dataset_description[2])
                            edges.append(current_dataset_description[3])
                            tr_function.append(current_dataset_description[4])
                            beta_upper.append(current_dataset_description[5])
                            scale.append(current_dataset_description[6])
                            gr_type.append(current_dataset_description[7])
                            s_size.append(sample_size)
                            csl_models.append(model_name)
                            #Done with saving results for ER: ######################################################


                            #Save results fro SF: ##################################################################
                            #Save metrics for SF graph estimation:
                            #results_TPR.append(sf_est_model_metrics['TPR'])
                            #results_FPR.append(sf_est_model_metrics['FPR'])
                            #results_nSHD.append(sf_est_model_metrics['normalized_shd'])
                            #results_FScore.append(sf_est_model_metrics['f1_score'])
                            #results_COD.append(sf_est_model_metrics['causal_order_divergence'])
                            #results_nSID.append(sf_est_model_metrics['normalized_sid'])
                            #results_DOS.append(sf_est_model_metrics['DOS'])
                            #dag_checks.append(sf_est_model_metrics['dag_check'])
                            #eliminated_edges_per_estimation.append(sf_est_model_metrics['eliminated_edges'])

                            #Save dataset description for SF graph estimaiton:
                            #seeds.append(current_dataset_description_sf[0])
                            #nodes.append(current_dataset_description_sf[1])
                            #connectivity.append(current_dataset_description_sf[2])
                            #edges.append(current_dataset_description_sf[3])
                            #tr_function.append(current_dataset_description_sf[4])
                            #beta_upper.append(current_dataset_description_sf[5])
                            #scale.append(current_dataset_description_sf[6])
                            #gr_type.append('SF')
                            #s_size.append(sample_size)
                            #csl_models.append(model_name)

        regrouped_frame=pd.DataFrame({'Nr_Run':np.array(seeds),'Nodes':np.array(nodes),'Connectivity':np.array(connectivity),
                                      'Edges':np.array(edges),'Transformation_Function':np.array(tr_function),
                                      'Beta_Upper_Limit':np.array(beta_upper),'Scale':np.array(scale),
                                      'Graph_Type':np.array(gr_type),'Sample_Size':np.array(s_size),
                                      'CSL_Model':np.array(csl_models),'DOS':np.array(results_DOS),
                                      'TPR':np.array(results_TPR),'FPR':np.array(results_FPR),
                                      'nSHD':np.array(results_nSHD),'FScore':np.array(results_FScore),
                                     'DAG_Check':np.array(dag_checks),
                                     'Eliminated_Edges':np.array(eliminated_edges_per_estimation),
                                     'Causal_Order_Divergence':np.array(results_COD),
                                     'nSID':np.array(results_nSID)})#append nSID to the dataframe

        types=[np.int32,np.int32,np.float64,np.int32,object,np.float64,object,object,object,
                   object,np.float64,np.float64,np.float64,np.float64,np.float64,bool,np.int32,np.float64,np.float64]

        for type_idx in range(0,len(types)):
            regrouped_frame[regrouped_frame.columns[type_idx]]=regrouped_frame[regrouped_frame.columns[type_idx]].astype(types[type_idx])
            
        with open(self.save_path+save_results_frame_name,'wb') as f:
            pickle.dump(regrouped_frame,f)

        self.results_dataframe=regrouped_frame
        



    