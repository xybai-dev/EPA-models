#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
__author__ = "Xinyu Bai"
__email__ = 'xybai@bjmu.edu.cn'
__version__ = '1.0.0'

import math
import os
import numpy as np
import pandas as pd
from scipy import stats, integrate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold,KFold,train_test_split
from sklearn.metrics import recall_score, cohen_kappa_score, roc_auc_score, roc_curve,auc
import pickle
from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
import biovec
import random
import sys
import argparse
import joblib


def pair_prediction(input_file,kinase_format="uniprot_id",output_dir="."):
    df_pair=pd.DataFrame()
    df_input=pd.read_csv(input_file)
    if kinase_format=="uniprot_id":
        kinases=[]
        smiles=[]       
        for ii,row in df_input.iterrows():
            smile=row.smiles.replace("\n","")
            if row.uniprot_id not in supported_uniprot:
                kinases.append(None)
                print("Not support "+row.uniprot_id+"!Please see the supported uniport_id file or input sequence of protein")
            else:
                kinases.append(row.uniprot_id)
                try:
                    smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile)))
                    names.append(row.name)
                except:
                    print(row.smiles+" format is abnormal.")
                    smiles.append(None)
        df_pair["name"]=names
        df_pair["smiles"]=smiles
        df_pair["uniprot_id"]=kinases
        df_pair = df_pair[~df_pair.smiles.isnull()]
        df_pair = df_pair[~df_pair.uniprot_id.isnull()]
        df_pair=df_pair.merge(kinase_seq_df,on="uniprot_id")
    elif kinase_format=="seq":
        kinases=[]
        smiles=[]
        for ii,row in df_input.iterrows():
            smile=row.smiles.replace("\n","")
            kinases.append(row.seq)
            try:
                smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smile)))
                names.append(row.name)
            except:
                print(row.smiles+" format is abnormal.")
                smiles.append(None)
        df_pair["name"]=names 
        df_pair["smiles"]=smiles
        df_pair["seq"]=kinases
        df_pair = df_pair[~df_pair.smiles.isnull()]
    else:
        raise Exception("Please input right format of kinase (uniprot_id or seq)")
    df_mol = pd.DataFrame(list(set(df_pair["smiles"].values)), columns = ["smiles"])
    Chem.PandasTools.AddMoleculeColumnToFrame(df_mol, smilesCol='smiles')
    df_pair= df_pair.merge(df_mol, on="smiles")
    
    df_pair['sentence'] = df_pair.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)
    df_pair['mol2vec'] = [DfVec(x) for x in sentences2vec(df_pair['sentence'], mol_model3, unseen='UNK')]
    df_pair['kinase_vec']= [sum(pv_300.to_vecs(ii)) for ii in df_pair['seq'].values]
    X_pair= np.array([list(np.concatenate((df_pair["mol2vec"].values[i].vec, df_pair["kinase_vec"].values[i]), axis=0)) for i in range(len(df_pair))])
    X_pair_norm=norm_model.transform(X_pair)
    preds=[]
    for pcmaae in os.listdir("models/ensemble_PCM_AAE"):
        if "aae" not in pcmaae:
            continue        
        mlp_norm_gan=joblib.load("./models/ensemble_PCM_AAE/"+pcmaae)
        y_pred = mlp_norm_gan.predict(X_pair_norm)
        preds.append(y_pred)

    X_pair_epa=np.array(preds).T
    df_pair["prediction"]=en_model.predict(X_pair_epa)
    df_pair[["name","smiles","seq","prediction"]].to_csv(output_dir+"/"+input_file.split("/")[-1].split(".")[0]+"_EPA_prediction.csv", index=False)


    
def kinase_profiling_prediction(input_smiles,output_dir="."):
    df_input=pd.read_csv(input_smiles)
    for smile in df_input["smiles"].values:
        name=df_input[df_input["smiles"]==smile]["name"].values[0]
        df_pair=kinase_seq_df[["kinase","uniprot_id","seq"]]

        try:
            smile_deal=Chem.MolToSmiles(Chem.MolFromSmiles(smile.replace("\n","")))
        except:
            raise Exception(smile+" format is abnormal")
        df_pair["smiles"]=[smile_deal for ii in range(len(df_pair))]
        df_mol = pd.DataFrame(list(set(df_pair["smiles"].values)), columns = ["smiles"])
        Chem.PandasTools.AddMoleculeColumnToFrame(df_mol, smilesCol='smiles')
        df_pair= df_pair.merge(df_mol, on="smiles")
        df_pair['sentence'] = df_pair.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)
        df_pair['mol2vec'] = [DfVec(x) for x in sentences2vec(df_pair['sentence'], mol_model3, unseen='UNK')]
        df_pair['kinase_vec']= [sum(pv_300.to_vecs(ii)) for ii in df_pair['seq'].values]
        X_pair= np.array([list(np.concatenate((df_pair["mol2vec"].values[i].vec, df_pair["kinase_vec"].values[i]), axis=0)) for i in range(len(df_pair))])
    
        X_pair_norm=norm_model.transform(X_pair)
        preds=[]
        for pcmaae in os.listdir("models/ensemble_PCM_AAE"):
            if "aae" not in pcmaae:
                continue        
            mlp_norm_gan=joblib.load("./models/ensemble_PCM_AAE/"+pcmaae)
            y_pred = mlp_norm_gan.predict(X_pair_norm)
            preds.append(y_pred)

        X_pair_epa=np.array(preds).T
        df_pair["prediction"]=en_model.predict(X_pair_epa)
        df_pair[["kinase","uniprot_id","prediction"]].to_csv(output_dir+"/"+name+"_kinase_profiling.csv", index=False)


def virtual_screening(kinase_input,input_smiles,kinase_format="uniprot_id",output_dir="."):
    '''
    input_smiles is the molecule database used to screen potential inhibitors.
    kinase_format can be uniprot id of kinase, sequence of kinase(seq) or just kinase name(kinase). Supported kinase name is in data/supported_kinase_name.txt
    '''
    df_pair=pd.DataFrame()
    df_input=pd.read_csv(input_smiles)
    smiles_deal=[]
    names=[]
    for ii,row in df_input.iterrows():
        try:
            smiles_deal.append(Chem.MolToSmiles(Chem.MolFromSmiles(row.smiles.replace("\n",""))))
            names.append(row.name)
        except:
            continue
    df_pair["smiles"]=smiles_deal
    df_pair['name'] = names
    if kinase_format=="uniprot_id":
        if kinase_input in kinase_seq_df["uniprot_id"].values:
            kinase_vec=sum(pv_300.to_vecs(kinase_seq_df[kinase_seq_df["uniprot_id"]==kinase_input]["seq"].values[0]))
        else:
            raise Exception("Not support this uniprot_id, you can see data/supported_UniprotID.txt or input sequence of kinase.")
        uniprot_id=kinase_input
    elif kinase_format=="seq":
        kinase_vec=sum(pv_300.to_vecs(kinase_input))
        df_pair["kinase_vec"]=[kinase_vec for ii in range(len(df_pair))]
        uniprot_id="YourSeq"
    elif kinase_format=="kinase":
        if kinase_input in kinase_seq_df["kinase"].values:
            kinase_vec=sum(pv_300.to_vecs(kinase_seq_df[kinase_seq_df["kinase"]==kinase_input]["seq"].values[0]))
        else:
            raise Exception("Not support this kinase name, you can you can see data/supported_kinase_name.txt or input sequence or uniprot_id of kinase.")
        uniprot_id=kinase_input
    else:
        raise Exception("Please input right format of kinase (uniprot_id or seq or protein name)")    
    df_mol = pd.DataFrame(list(set(df_pair["smiles"].values)), columns = ["smiles"])
    Chem.PandasTools.AddMoleculeColumnToFrame(df_mol, smilesCol='smiles')
    df_pair= df_pair.merge(df_mol, on="smiles")
    df_pair['sentence'] = df_pair.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)
    df_pair['mol2vec'] = [DfVec(x) for x in sentences2vec(df_pair['sentence'], mol_model3, unseen='UNK')]
    df_pair['kinase_vec'] =[kinase_vec for ii in range(len(df_pair))]
    X_pair= np.array([list(np.concatenate((df_pair["mol2vec"].values[i].vec, df_pair["kinase_vec"].values[i]), axis=0)) for i in range(len(df_pair))])
    X_pair_norm=norm_model.transform(X_pair)
    preds=[]
    for pcmaae in os.listdir("models/ensemble_PCM_AAE"):
        if "aae" not in pcmaae:
            continue        
        mlp_norm_gan=joblib.load("./models/ensemble_PCM_AAE/"+pcmaae)
        y_pred = mlp_norm_gan.predict(X_pair_norm)
        preds.append(y_pred)

    X_pair_epa=np.array(preds).T
    df_pair["prediction"]=en_model.predict(X_pair_epa)
    df_pair[["name","smiles","prediction"]].to_csv(output_dir+"/"+uniprot_id+"_VS.csv",index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='program description')
    parser.add_argument('--mode', help='option: pp(pair prediction), kp(kinase profile prediction),vs(virtual screening)')
    parser.add_argument('--fpair', help='The input file name when mode is pp, kformat need input together,for example, examples/PP/pp_input_seq.csv or examples/PP/pp_input_uniprot.csv', default="")
    parser.add_argument('--kinase', help='uniprot id or sequence of kinase, kformat need input together',default="")
    parser.add_argument('--kformat', help='option: uniprot_id, seq, kinase',default="uniprot_id")
    parser.add_argument("--fsmile", help="The input smile file when mode is vs or kp, for example, examples/VS/smiles_input.csv") 
    parser.add_argument("--o", help="output file path",default=".") 

    args = parser.parse_args()
    mode=args.mode
    input_file = args.fpair
    kinase_input=args.kinase
    kinase_format = args.kformat
    output_dir=args.o
    input_smiles=args.fsmile
    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    mol_model3 = word2vec.Word2Vec.load('./models/mol2vec_model/model_300dim.pkl')
    kinase_seq_df=pd.read_csv("./data/496_kinase_info.csv")[["kinase","uniprot_id","seq"]]
    supported_uniprot=kinase_seq_df["uniprot_id"].values
    pv_300 = biovec.models.load_protvec('./models/protvec_model/SwissProt_300.model')
    norm_model=joblib.load("./models/norm_model.sav")
    en_model=joblib.load("./models/ensemble_PCM_AAE/mlp_ensemble.sav")
    if mode=="pp":
        pair_prediction(input_file,kinase_format,output_dir)
    elif mode=="kp":
        kinase_profiling_prediction(input_smiles,output_dir)
    elif mode=="vs":
        virtual_screening(kinase_input,input_smiles,kinase_format,output_dir)
    else:
        print("Wrong mode")
   
