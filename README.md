# About PCM-AAE
## paper "Semi-supervised Proteochemometrics-Adversarial Autoencoder Model (PCM-AAE): Case Study of Large-scale Kinase-inhibitor Mapping"

PCM-AAE is for predicting kinase-inhibitor binding affinity by DNN model with training data augementation. The input can be (1) the pair of a SMILES string of molecule and an amino acid sequence of protein kinase;(2) only SMILES strings of one molecule or a list of multiple molecules to predict the kinase profile; (3) only Uniprot_id or amino acid sequence of one or more protein kinases and compound library to predict the potential inhibitors of the protein from specific compound library. The output is csv files recording possible log binding affinity. In addition, the AAE model architecture for data augmentation, based on TensorFlow, is also shown in "models" directory，which augements data with different types.The overview of PCM-AAE strategy is as follows:

<div align="center">
<p><img src="https://github.com/xybai-dev/PCM-AAE/tree/master/png/architecture.png" width="1400" /></p>
</div>

## Requirements

If you want to predict binding affinity of kinase-inhibitor, you'll need to install following in order to run the codes.

*  Python 3 (Python 2.x is not supported)
*  joblib
*  numpy
*  biovec
*  mol2vec
*  pandas
*  RDKit
*  scikit-learn
*  gensim
*  pickle

If data augmentation need to be applied into the other data types, you‘ll need to install TensorFlow and perform tuning.

## Installation
You can obtain PCM-AAE using git
```
git clone https://github.com/xybai-dev/PCM-AAE.git
```
or you can just go to "https://github.com/xybai-dev/PCM-AAE" and download the repository
```

cd PCM-AAE
```
## Usage

You can input one or more molecules or proteins for diffeerent purposes. Noting input of molecule is SMILES string and input of protein is uniprot_id or sequence will be best matched. Support uniprot_id shown in data/sup_uniprot.txt. if the input is sequence, we recommand sequence of kinase domain input adopted if possible.

### Input: kinase-inhibitor pair
1. one kinase-inhibitor pair
```
python3 

```
2. Multiple kinase-inhitor pairs
the sample input file is in "examples/simple_pred/pairs_pred.txt"

```
python3 

```
### Input: molecules for kinase profile prediction
1. one molecule
```
python3 

```
2. more molecules
the sample input file is in "examples/kinase_profile_pred/molecules.txt"
```
python3 

```

### Input: uniprot_id/sequence of protein for vitrual screening
1. one protein
the sample input file of compound library is in "examples/VS/compound_lib.txt"
```
python3 

```
2. more proteins
the sample input file of protein sequence file is in "examples/VS/protein_seq.txt"
the sample input file of protein uniprot_id file is in "examples/VS/protein_uniprot_id.txt"


```
python3 

```


**For citation:**







