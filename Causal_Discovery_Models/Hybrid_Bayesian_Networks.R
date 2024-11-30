library('pcalg')
library(reticulate)
library(MLmetrics)
library(caret)
library(bnlearn)
library(data.table)
library(parallel)
library(doParallel)
library(foreach)
library(dplyr)
library(pchc)
library(doSNOW)
library('R.utils')
library(reticulate)


#Function: #####################################################################
fill_in_blacklist_pcstable<-function(x,skeleton,blacklist=NULL){
  #x: current dataframe as matrix
  #a: estimated skeleton object
  nama <- colnames(x)
  if ( is.null(nama) )  nama <-  paste("X", 1:dim(x)[2], sep = "")
  colnames(x) <- nama
  vale <- which(skeleton == 1)
  if ( length(vale) > 0 ) {
    x <- as.data.frame(x)
    mhvale <- as.data.frame( which(skeleton == 0, arr.ind = TRUE) )
    mhvale[, 1] <- nama[ mhvale[, 1] ]
    mhvale[, 2] <- nama[ mhvale[, 2] ]
    colnames(mhvale) <- c("from", "to")
    if ( !is.null(blacklist) ) {
      colnames(blacklist) <- c("from", "to")
      mhvale <- rbind(mhvale, blacklist)
    }
  }
  return(mhvale)}

fill_in_blacklist_fedmmtabu<-function(x,a,blacklist=NULL){
  #x: current dataframe as matrix
  #a: estimated skeleton object
  
  nama <- colnames(x)
  if ( is.null(nama) )  nama <-  paste("X", 1:dim(x)[2], sep = "")
  colnames(x) <- nama
  vale <- which(a$G == 1)
  if ( length(vale) > 0 ) {
    x <- as.data.frame(x)
    mhvale <- as.data.frame( which(a$G == 0, arr.ind = TRUE) )
    mhvale[, 1] <- nama[ mhvale[, 1] ]
    mhvale[, 2] <- nama[ mhvale[, 2] ]
    colnames(mhvale) <- c("from", "to")
    if ( !is.null(blacklist) ) {
      colnames(blacklist) <- c("from", "to")
      mhvale <- rbind(mhvale, blacklist)
    }
  }
  return(mhvale)}

create_pc_skeleton<-function(estimated_adjacency){
  nvar=nrow(as.matrix(estimated_adjacency))
  for (nrow in 1:nvar){
    for (ncol in 1:nvar){
      if (estimated_adjacency[nrow,ncol]==1){
        estimated_adjacency[ncol,nrow]=1
      }
    }
  }
return (estimated_adjacency)}

try_catch_model<-function(current_frame,blacklist,
                          score,tabu_list_length){

  tryCatch(
    expr = {
      dag_tabu<- bnlearn::tabu(x=current_frame,blacklist = blacklist,
                         score = score, tabu = tabu_list_length)
      
      #Fill in binary adjacency matrix:
      estimated_adjacency_tabu=fill_in_estimation(current_frame=current_frame,
                                             nodes_result=dag_tabu$nodes,
                         estimated_causal_matrix=matrix(0,nrow=ncol(current_frame),
                                                        ncol=ncol(current_frame)))
      
      return (estimated_adjacency_tabu)
    },
    error = function(e){
      message('Caught an error with TABU Score',score)
      return ('None')
    } )}

fill_in_estimation<-function(current_frame,nodes_result,estimated_causal_matrix){
  for (col_node in 1:ncol(current_frame)){
    current_node_data=nodes_result[[col_node]]
    
    parents=current_node_data$parents
    children=current_node_data$children
    
    if (length(parents)>0){
      for (parent in parents){
        estimated_causal_matrix[as.integer(parent)+1,col_node]<-1 }}
    
    if (length(children)>0){
      for (child in children){
        estimated_causal_matrix[col_node,as.integer(child)+1]<-1 } }
  }
  return(estimated_causal_matrix)}


#Parallelisation: #############################################################
CSL_res_short<-function(current_frame_description,
                        current_true_binary_adjacency,
                        current_true_weighted_adjacency,
                        current_frame,
                        adjacency_pc_stable,
                        tabu_list_length){
  
  #1. PC-Stable:
  #1.1. Skeleton & Blacklist:
  estimated_skeleton_pcstable=create_pc_skeleton(estimated_adjacency=adjacency_pc_stable)
  blacklist_skeleton_pcstable<-fill_in_blacklist_pcstable(x=as.matrix(current_frame),
                                             skeleton =estimated_skeleton_pcstable)
  #1.2. Orientation of Edges:
  pctabu <- try_catch_model(current_frame=current_frame,
                            blacklist=blacklist_skeleton_pcstable,
                            score='bge',tabu_list_length=tabu_list_length)

  
  #2. MM(TABU):
  #2.1. Skeleton & Blacklist:
  estimated_skeleton_mmpc<-pchc::mmhc.skel(x=as.matrix(current_frame))
  blacklist_skeleton_mmpc<-fill_in_blacklist_fedmmtabu(x=as.matrix(current_frame),
                                             a=estimated_skeleton_mmpc)
  #2.2. Orientation of Edges:
  mmtabu <- try_catch_model(current_frame=current_frame,
                                         blacklist=blacklist_skeleton_mmpc,
                                         score='bge',tabu_list_length=tabu_list_length)

  #3. FED(TABU):
  #3.1. Skeleton & Blacklist:
  estimated_skeleton_fed<-pchc::fedhc.skel(x=as.matrix(current_frame))
  blacklist_skeleton_fed<-fill_in_blacklist_fedmmtabu(x=as.matrix(current_frame),
                                             a=estimated_skeleton_fed)
  #3.2. Orientation of Edges:
  fedtabu <- try_catch_model(current_frame=current_frame,
                                         blacklist=blacklist_skeleton_fed,
                                         score='bge',tabu_list_length=tabu_list_length)

  return (list('dataset'=current_frame,
               'dataset_description'=current_frame_description,
               'true_binary_adjacency'=current_true_binary_adjacency,
               'true_weighted_adjacency'=current_true_weighted_adjacency,
               'fedtabu'=fedtabu,
               'mmtabu'=mmtabu,
               'pctabu'=pctabu))
}
#Done with all functions for running hybrid bayesian networks: #################


#Collect the results from hybrid bayesian networks: ############################
#ADJUST THE FILE NAME IF NECESSARY!!!
results_file_name='Causal_Discovery_Results_Demo.rds'

tabu_list_length=1000
file_path=strsplit(rstudioapi::getSourceEditorContext()$path, split="Causal_Discovery_Models/Hybrid_Bayesian_Networks.R")
file_read_path=paste(file_path,'Performance_Evaluation_Framework/Results/',results_file_name,sep='')
results_list=readRDS(file = file_read_path)

csl_results=list()
for (estimation_index in 1:length(results_list)){
  
  current_dataset_description=results_list[estimation_index][[1]][[1]]
  current_true_adjacency=results_list[estimation_index][[1]][[2]]
  current_true_weighted_adjacency=results_list[estimation_index][[1]][[3]]
  current_frame=results_list[estimation_index][[1]][[4]]
  current_pc_stable_estimation=results_list[estimation_index][[1]][[5]]
  
  bayesian_network_results=CSL_res_short(current_frame_description=current_dataset_description,
                                         current_true_binary_adjacency=current_true_adjacency,
                                         current_true_weighted_adjacency=current_true_weighted_adjacency,
                                         current_frame=current_frame,
                                         adjacency_pc_stable=current_pc_stable_estimation,
                                         tabu_list_length=tabu_list_length)
  
  csl_results[[estimation_index]]=bayesian_network_results
}
#Done with collecting the results: #############################################


#Save the results: #############################################################
file_save_path=paste(file_path,'Performance_Evaluation_Framework/Results/',strsplit(results_file_name,'.rds'),'_Hybrid_Bayesian_Networks.pkl',sep='')
py_save_object(csl_results,file_save_path,pickle='pickle')
#Done with saving the results: #################################################
