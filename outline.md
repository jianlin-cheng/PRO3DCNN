%==========================================================
# Abstract
%==========================================================

*Motivation* SCOPe 2.07 is a database of 276,231 protein that have been partitioned into varying folds according to their shape and function. Since a protein's fold reveals valuable information about it's shape and function, it is important to find a mapping between protein's and it's fold. There are existing techniques to map a protein's sequence into a fold \cite{deepsf} but none to map a protein's shape into a fold. We focus on the topological features of a protein to map it into a fold. We introduce several new techniques that accomplish this.

*Results* We develop a 2D-convolutional neural network to classify any protein structure into one of 1232 folds.
We extract two classes of input features for each protein's carbon alpha backbone: distance matrix and the persistent homology barcodes. Due to restrictions in our computing resources, we make sample every other point in the carbon alpha chain. We find that it does not lead to significant loss in accuracy.
Using the distance matrix, we achieve an accuracy of 86\% on the entire dataset.

We extract significant topological simplexes of the protein by using persistent homology. We format the persistent homology data into various input features: persistence images \cite{persistenceImages}, simplex distance map, and simplex grouping.  With persistence images of 100x100 resolution, we achieve an accuracy of 62\% on a test dataset consisting of 3000 proteins. With simplex distance maps of 100x100 resolution, we achieve an accuracy of :TODO\%. With simplex groupings, we achieve an accuracy of :TODO\%.


*what is the goal of the paper*
Classify proteins based on their shapes.
*What is important?*

%==========================================================
# Introduction
%==========================================================
- History


%==========================================================
# Materials
%==========================================================

%---------------------
## Datasets
%---------------------

The SCOP database is a database of proteins organized into hierarchical classes based on their shape and function. There are 4 levels of the hierarchy (top down): Class, Folds, Superfamily, Family. We will be primarily concerned with the Fold.

Two different versions of the dataset were used, SCOP 1.55 and SCOPe 2.07. SCOP 1.55 is a smaller dataset which is a subset of SCOPe 2.07, which is larger and more recent. SCOP 1.55 [Ref: Biblio] and SCOPe 2.07 [Ref: Biblio] were downloaded from the Berkeley repository as tar files and unpacked. For each of the datasets, index files [Ref:Biblo] are provided.

A line in the index file provides relevant information for each protein in our database [Ref: index example]. For our example, the PDB (Protein Data Bank) file that describes the shape of 3D protein is found under the directory 'dl' as 'd1dlwa_.ent'.

- Index Line: d1dlwa_ 1dlw    A:  a.1.1.1 14982   cl=46456,cf=46457,sf=46458,fa=46459,dm=46460,sp=46461,px=14982
- Name: d1dlwa_
- Class.Folds.Superfamily.Family: a.1.1.1



### Small Dataset: SCOP 1.55
SCOP 1.55 is a dataset of 31,474 proteins that have been organized into 7 Classes, 605 Folds, 947 Superfamilies, and 1557 Families. The dataset was released and updated till 2001. This dataset is a subset of the SCOPe 2.07 dataset.

[Fig: Freq of data across folds]

We inspect the distribution of the proteins across the different folds. We see that most of the folds do not have a lot of proteins. The median number of classes per fold was 10 and the histogram [Ref: Figure] show that most of our proteins have less than 50 proteins per fold. This would likely impact our learning, since there are not too many examples for the protein to learn.

We split up 70% of the dataset for training, 15% for validation and 15% for testing. We adjust the sampling of the validation and testing so that a wide range of folds are represented.

New methods were tested on the small dataset. This allowed us to quickly prototype and optimize our methods to run on the larger dataset.

### Large Dataset: SCOPe 2.07 :TODO:
SCOPe 2.07 is a database of 276,231 proteins that have been organized into 7 Classes, 1232 Folds, 2026 Superfamilies, and 4919 Families. The dataset was released and updated till 2017. This dataset contains and is about 9 times larger than the SCOP 1.55 dataset.

[Fig: Freq of data across folds]

We inspect the distribution of the proteins across the different folds.



%---------------------
## Third Party Tools
%---------------------

A number of third party tools where used as part of the research. ProDy's (Protein Dynamics & Sequence Analysis) python package was used to extract the backbone structure from each protein's PDB file. NumPy package was used for matrix operations and statistical operations. Matplotlib package was used to generate histograms and 3D plots of the proteins and the barcodes. Pillow was used to generate the 2D images of the distance matrix and the barcode images. Keras and Tensorflow was used to setup and train the convolutional neural network.

%---------------------
## Computing Resources :TODO:
%---------------------

Initially the research started on a personal laptop without a GPU and very limited storage space [Ref: Table]. Since the unpacked data of SCOPe 2.07 took up around 40GB, the data was stored on a flash drive. 

Even relatively simple methods on the smaller, SCOP 1.55, dataset took around 20 hours because of the lack of a CPU. 
This significantly slowed

- CPU: Intel Core i7-4650U 
- GPU (CUDA-enabled): None
- RAM: 8GB
- Memory: 128GB SSD
- OS: OSx

Due to the limitations of the personal laptop, a personal workstation was purchased at $500 on Winter of 2018 [Ref: Table]. Although it has moderate computing power, relative to industry standards, the new machine led to significant increase in speed and efficiency. Most of the optimized methods took around 16 hours at most to complete on the SCOPe 2.07 database. 

- CPU: E5-2630 6-Core
- GPU (CUDA-enabled): Nvidia GTX 1060 6GB
- RAM: 16GB
- Memory: 500GB SSD
- OS: Ubuntu 16.02


%==========================================================
# Methods
%==========================================================

%---------------------
## Background Information: Proteins
%---------------------
A protein is a sequence of amino acids, 22 organic compounds which link into a protein chain. The interaction between the amino acids and the surrounding environment determine the how the protein folds into its structure.

For a protein that we are tasked to classify, we are provided with many information: Protein sequence and the sequential coordinates of every atom on our protein. Since we are primarily interested in the topological features of our data, we characterize the protein's shape with the protein backbone. The protein's backbone is constructed with a sequence of points, where each point represents each amino acid in a 3D space. The point representation of each amino acid is determined by the algorithm 'parsePDB' in the ProDy (Protein Dynamics & Sequence Analysis) package. 

Since we will be dealing primarily with the protein's backbone, we introduce the following notation.

*Definition*
A protein, `$P$`, with sequence of N amino acids will be a called a protein with length N. It's backbone will be denoted as a sequence of 3D coordinate points `$ \{P_i\}^{N}_{i=1} $` where `$ P_{i} \e mathbb{R}^3 $`

For each coordinate point `$ P_{i} $`, the x-coordinate is referred as `$ P_{i}(1) $`, y-coordinate as `$ P_{i}(2) $`, and z-coordinate as `$ P_{i}(3) $`.

We extract two distinguishing types of input features from the protein's backbone chain: Distance matrix and persistence homology. These two features are very robust. They are both *rotation and translation invariant*, meaning that the features remains the same even if the protein is rotated or moved. They are also very stable. Minor changes to the data does not create a significant variation in the feature.


%---------------------
## Distance Matrix
%---------------------

With the points in the protein backbone, we construct a distance matrix of the distances between the points. The unit of the distances Ångströms, the units of provided in the protein PDB files. [Protein Data Bank]. We used the Euclidean Distance for the distances between the points.

*Definition*
For a protein,`$P$`, with length N, it's distance matrix denoted as `$M_{P}$`.
We construct it as follows.
`$ M_{P} := [M_{ij}] $` where `$M_{ij} = Euclidean Distance(P_i, P_j) = \sqrt((P_i(1)-P_j(1))^2+(P_i(2)-P_j(2))^2+(P_i(3)-P_j(3))^2)$`

*Remark*
We note the following for any distance matrix `$M_{P}$`.
- `$M_{ii} = 0$`
- `$M_{ij} = M_{ji}$`
- The intersection of the ith row and the jth column corresponds to `$M_{ij}$`, the distance between the ith and the jth point.

We inspect the topological structure of our protein '1ux8'. It is a protein of length 118. This protein has some spiral structures (alpha helix). These spiral structures sometimes lie in close proximity in parallel or anti parallel direction (beta sheet). These types of structures (secondary structures) are known in molecular biology to be important features of a protein's shape.

[Fig: Figure of Protein]

We analyze the distance matrix to see if important structural features are represented in the matrix. To help with visualization, the distance matrix is mapped to an image of equal size, where closest distances appear in white and furthest distances appear in black. Distances in between take a gray hue with the intensity based on its value.

[Fig: Distance Matrix]

Feature Protein Backbone:
Along the diagonal line of the image, the distance matrix is completely white. This is because the distance between a point to itself is zero [Ref: Remark]. We note that this diagonal white line uniquely identifies the protein backbone chain (Since the distance between. Having a clear representation of the protein backbone is important because  it is a central structure that other features can be spatially oriented around.

Feature Alpha Helix:
We note thick white regions running parallel along the diagonal line of the image [Ref: Image]. These regions indicate that at a given point in the chain, it is in close proximity to the nearby neighbors [Ref Diagram]. We also note that in for a point in an Alpha Helix, it is also in close proximity to it's nearby neighbors. In our example, these four thick white regions correspond the the four helix structures on our protein.

[Fig: Analysis of the Thick Region] [Fig: Diagram of Thick Regions]


*Defintion*
A helix is composed of points in the protein backbone. Suppose `$H$` is the set of indices of these points along the length of the protein. We call the row belonging to this helix as the collection of rows, `$R_i$`, of the distance matrix such that `$i \in H$`. Similarly we define the column belonging to the helix. `$\{C_i | i \in H \}$`

Feature Beta Sheet:
We note patches of thick white regions in the intersection of the rows belonging to a helix and the columns belonging to another helix [Ref Diagram]. This indicates that the points of the two helices, and hence the two helices, are in close proximity. In particular, the regions are close sequentially: the ith point in helix A is close to the jth point in helix B and the i+1 th point in helix A is close to the j-1th point in helix B. This sequential relationship describes a Anti-parallel Beta Sheets. For Parallel Beta Sheets, the i+1th point would be close to the j+1 th point. In our example, the 3 pairs of Beta-sheets formed the 4 helices are represented in our distance matrix.

### Cropped Distance Matrix

Due to limitations in computing resources, we sample distance matrices in smaller window sizes.

Because of the limitations set by the windowed distance matrix, a point in the backbone can only see information about, on average, half of the window size forwards and backwards. In our example, the cropped distance matrix would not see information about it's relationship with the third alpha helix. This limitation affects the cropped matrix's ability to detect longer range contact information, which can be critical in determining the protein's overall shape.

### Sampled Protein Backbone

[Figure: Graph of Sampled Protein Backbone]

We sampled every other point in our protein backbone and graphed the coordinates in $\mathbb{R}^3$. [Ref ] When compared with the original structure [Ref: Original Image], we see that the general structure of the protein and its features diminished but preserved. The sparse distance matrix is constructed in the same fashion as a regular distance matrix but using the sequence of points in the sparse protein backbone. We analyze the distance matrix of the Sparse Protein Backbone and see if the backbone chain, alpha helix and beta sheet features are preserved in the distance matrix.

[Figure of the distance matrix of sparse protein]


%---------------------
## Persistence Homology
%---------------------

### Mathematical Background
### Sparse Persistence Homology
### Backbone Aware Persistence Homology

%==========================================================
# Bibliography
%==========================================================
Protein Data Bank 
https://cdn.rcsb.org/wwpdb/docs/documentation/file-format/changesv3.20.pdf

SCOPe2.06
https://scop.berkeley.edu/astral/pdbstyle/ver=2.06
https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.06-stable.txt

SCOP1.55
https://scop.berkeley.edu/astral/pdbstyle/ver=1.55
https://scop.berkeley.edu/downloads/parse/dir.cla.scop.1.55.txt

PRODY ParsePDB Ref
http://prody.csb.pitt.edu/manual/reference/proteins/pdbfile.html#prody.proteins.pdbfile.parsePDB