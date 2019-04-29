# PRO3DCNN
3D Convolutional Neural Networks for Classifying Protein Structures into Folds


## Data

The data can be downloaded from the berkely website. 
Please download multiple pdb archives into the PRO3DCNN directory and use  **tar -xf 'name of tar file'** to unzip the data.
The unzipped files are around 50GB.

### SCOP 1.55 (index file and pdb archive) 1.4GB
https://scop.berkeley.edu/downloads/parse/dir.cla.scop.1.55.txt
https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-1.55.tgz

### SCOP 2.07 (index file and pdb archives) 8.8GB
https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.07-stable.txt
https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-2.07-1.tgz
https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-2.07-2.tgz
https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-2.07-3.tgz
https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-2.07-4.tgz
https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-2.07-5.tgz
https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-2.07-6.tgz
https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-2.07-7.tgz

## Processing Data

The following workflow is used to process the data.

Protein Chain -> Dist M  -> Cropped Dist M

Protein Chain -> Dist M  -> Persistent Barcodes -> Persistence Images 
                         
````
import MAT
import TDA


#This will save the protein chains in batches as chains/0 chains/1 chains/2 ...
MAT.getChains(loadPath='pdbstyle-1.55/', savePath='chains/',SCOPEdir='dir.cla.scop.1.55.txt')
MAT.getChains(loadPath='pdbstyle-2.07/', savePath='chains/',SCOPEdir='dir.cla.scope.2.07-stable.txt')

#rangeTo is the number that the files under 'chains/' that we should process
MAT.getDistMs(loadPath='chains/',savePath='mats/',sparse=False,rangeTo=N)


#########################
# Cropped Dist M
#########################
#upToBatchNum is the number of files under 'mats/' that we should process
MAT.splitMat(loadPath='mats/', savePath='croppedMats/',windowSize=100,upToBatchNum=N)

#########################
# Persistence Homology
#########################
#toRange is the number of files under 'mats/' that we should process
TDA.genHoms(loadPath='mats/', savePath='barcodes/', toRange=N)
#rng is the number of files under 'mats/' that we should process
TDA.getBcodeImgsSeparated(loadPath='barcodes/', savePath='barcodeImgs/',rng=N)


````

## Training and Evaluation
Training and evaluation is done by running the following files.

trainDistM.py

trainBcodeImg.py
