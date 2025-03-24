# Contact matrices stratified by age, SES and SEP

This repository includes the code used in the following manuscript:
"Individual-based and neighbourhood-based socio-economic factors relevant for contact behaviour and epidemic control"
by Laura Di Domenico, Martina L. Reichmuth, Christian L. Althaus, available soon on medrxiv

This work uses social contact data collected in Switzerland through the CoMix survey. Check out the [CoMix](https://github.com/ISPMBern/comix) repository for other publications related to these data.

The folder "code" contains the list of scripts used for the analyses (in python and R). 

The scripts numbered as 00, 01, 02, 03 and 04 use as input individual-based social contact data, which are not contained in this repository (see the Data Availability Statement in the manuscript above).

The outputs of script 04 are the estimated age-stratified contact matrix and intermediate contact matrix. They are stored in the "output/matrices" folder and are used as input for the rest of the scripts.

All the other scripts can be executed to reproduce the results and the figures of the manuscript. 

In particular:
- script 05 reproduces Fig. 3a,b
- script 06 and the scripts contained in the subfolder "parameter_space_exploration" define the assortativity parameters for each block of the expanded contact matrix
- script 07 reproduces Fig. S10, S11 and S12
- script 08 generates a set of 10,000 expanded contact matrices
- script 09 reproduces Fig. 3d,e,f,g,h
- scripts 10, 11 and 12 allow to reproduce the results of Fig. 4 (epidemic spreading in absence of control strategies)
- scripts 13 and 14 allow to reproduce the results of Fig. 5 (effectiveness of targeted control strategies)
