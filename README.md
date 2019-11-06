# About PCM-AAE
## paper "Semi-supervised Proteochemometrics-Adversarial Autoencoder Model (PCM-AAE): Case Study of Large-scale Kinase-inhibitor Mapping"

PCM-AAE is for predicting kinase-inhibitor binding affinity by DNN model with training data augementation. The input can be (1) the pair of a SMILES string of molecule and an amino acid sequence of protein kinase;(2) only SMILES strings of one molecule or a list of multiple molecules to predict the kinase profile; (3) only Uniprot_id or amino acid sequence of one or more protein kinases and compound library to predict the potential inhibitors of the protein from specific compound library. The output is csv files recording possible log binding affinity. In addition, the AAE model architecture for data augmentation, based on TensorFlow, is also shown in "models" directory，which augements data with different types.The overview of PCM-AAE strategy is as follows:

<div align="center">
<p><img src="https://github.com/xybai-dev/PCM-AAE/raw/master/png/EPA.png" width="600" /></p>
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
or you can just go to "https://github.com/xybai-dev/EPA" and download the repository
```

cd PCM-AAE
```
## Usage

You can input one or more molecules or proteins for diffeerent purposes. Noting input of molecule is SMILES string and input of protein is uniprot_id or sequence will be best matched. Support uniprot_id shown in data/sup_uniprot.txt. if the input is sequence, we recommand sequence of kinase domain input adopted if possible.

### Input: kinase-inhibitor pair
If format of protein is uniprot_id,the sample input file is in "examples/PP/pp_input_uniprot.csv"

```
python3 EPA_prediction.py --mode pp --fpair examples/PP/pp_input_uniprot.csv --o examples/PP/results/

```
If format of protein is sequence,the sample input file is in "examples/PP/pp_input_seq.csv". Noting sequence of kinase domain yields reliable prediction.
```
python3 EPA_prediction.py --mode pp --fpair examples/PP/pp_input_seq.csv --kformat seq --o examples/PP/results/

```

### Input: molecules for kinase profile prediction
The sample input file is in "examples/kinase_profile_pred/molecules.txt"
```
python3 EPA_prediction.py --mode kp --fsmile examples/KP/smiles_input.csv --o ./examples/KP/results

```

### Input: uniprot_id/sequence/name of protein for vitrual screening
The sample input file of compound library is in "examples/VS/smiles_input.csv"

If format of kinase is kinase name, supported kinase name is in "data/supported_kinase_name.txt".

```
python3 EPA_prediction.py --mode vs --fsmile examples/VS/smiles_input.csv --kinase EGFR --kformat kinase --o ./examples/KP/results

```
If format of kinase is kinase name, supported kinase name is in "data/supported_UniprotID.txt".
```
python3 EPA_prediction.py --mode vs --fsmile examples/VS/smiles_input.csv --kinase P00533 --o ./examples/VS/results
```

Format of kinase is sequence of kinase. Noting sequence of kinase domain yields reliable prediction.
```
python3 EPA_prediction.py --mode vs --fsmile examples/VS/smiles_input.csv --kinase FKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFS --kformat seq --o ./examples/VS/results

```

## How to cite?







