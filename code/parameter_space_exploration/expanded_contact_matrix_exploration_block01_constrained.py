#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import cm
import math
import random


# # read population data

data_pop_grouped = pd.read_csv("./input/pop_size_by_age_SEP_edu.csv")

dict_pop_full = dict(zip(data_pop_grouped[['age_group',
                           'sep_level', 'edu_level']].apply(lambda x:
                                               x[0]+", "+x[1]+", "+x[2], axis=1).values,
                   data_pop_grouped['population']))
                   
pop_age = data_pop_grouped.groupby('age_group')['population'].sum()
pop_age = dict(zip(pop_age.index, pop_age.values))

distrib_pop = data_pop_grouped.groupby('age_group')['population'].sum()/data_pop_grouped['population'].sum()
distrib_pop = dict(zip(distrib_pop.index, distrib_pop.values))
                   
# # read intermediate contact matrix

intermediate_matrix_rec = pd.read_csv("./input/intermediate_matrix_rec.csv", index_col = 0)

intermediate_matrix_rec = intermediate_matrix_rec.fillna(0)

# # expanded contact matrix

# ## functions with linear systems

# ### diagonal blocks (16 elements)

# In[28]:


#x111_211, x111_212, x111_221, x111_222
#x112_211, x112_212, x112_221, x112_222
#x121_211, x121_212, x121_221, x121_222
#x122_211, x122_212, x122_221, x122_222

#x211_111, x211_112, x211_121, x211_122
#x212_111, x212_112, x212_121, x212_122
#x221_111, x221_112, x221_121, x221_122
#x222_111, x222_112, x222_121, x222_122


def offdiag_block(q11, q21, q12_sep):

    age_i = '0-14'
    age_j = '15-24'

    tag_11 = ', low SEP, low edu'
    tag_12 = ', low SEP, high edu'
    tag_21 = ', high SEP, low edu'
    tag_22 = ', high SEP, high edu'

    a = np.array([[1., 1., 1., 1.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## sum
                  #[0., 0., 0., 0.,  1., 1., 1., 1.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                  # 0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## sum
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  1., 1., 1., 1.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## sum
                  #[0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  1., 1., 1., 1.,
                  # 0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## sum
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   1., 1., 1., 1.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## sum
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  1., 1., 1., 1.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## sum
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  1., 1., 1., 1.,  0., 0., 0., 0.], ## sum
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  1., 1., 1., 1.], ## sum

                  [1., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## assortatitivy subdiagonal
                  #[0., 0., 0., 0.,  0., 1., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                  # 0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## assortatitivy subdiagonal
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 1., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## assortatitivy subdiagonal
                  #[0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 1.,
                  # 0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## assortatitivy subdiagonal

                  #[0., 1., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                  # 0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## subassortatitivy
                  #[0., 0., 0., 0.,  0., 0., 1., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                  # 0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## subassortatitivy
                  #[0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 1.,  0., 0., 0., 0.,
                  # 0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## subassortatitivy


                [1., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   -1., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## reciprocity
                  [0., 1., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  -1., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## reciprocity
                  [0., 0., 1., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  -1., 0., 0., 0.,  0., 0., 0., 0.], ## reciprocity
                  [0., 0., 0., 1.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  -1., 0., 0., 0.], ## reciprocity

                #[0., 0., 0., 0.,  dict_pop_full[age_i+tag_12], 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                #   0., -dict_pop_full[age_j+tag_11], 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## reciprocity
                #  [0., 0., 0., 0.,  0., dict_pop_full[age_i+tag_12], 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                #   0., 0., 0., 0.,  0., -dict_pop_full[age_j+tag_12], 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## reciprocity
                #  [0., 0., 0., 0.,  0., 0., dict_pop_full[age_i+tag_12], 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                #   0., 0., 0., 0.,  0., 0., 0., 0.,  0., -dict_pop_full[age_j+tag_21], 0., 0.,  0., 0., 0., 0.], ## reciprocity
                #  [0., 0., 0., 0.,  0., 0., 0., dict_pop_full[age_i+tag_12],  0., 0., 0., 0.,  0., 0., 0., 0.,
                #   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., -dict_pop_full[age_j+tag_22], 0., 0.], ## reciprocity

                [0., 0., 0., 0.,  0., 0., 0., 0., 1., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., -1., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## reciprocity
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 1., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., -1.,  0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## reciprocity
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 1., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., -1., 0.,  0., 0., 0., 0.], ## reciprocity
                #  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., dict_pop_full[age_i+tag_21],  0., 0., 0., 0.,
                #   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., -dict_pop_full[age_j+tag_22], 0.], ## reciprocity

                #  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  dict_pop_full[age_i+tag_22], 0., 0., 0.,
                #   0., 0., 0., -dict_pop_full[age_j+tag_11],  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## reciprocity
                #  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., dict_pop_full[age_i+tag_22], 0., 0.,
                #   0., 0., 0., 0.,  0., 0., 0., -dict_pop_full[age_j+tag_12],  0., 0., 0., 0.,  0., 0., 0., 0.], ## reciprocity
                #  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., dict_pop_full[age_i+tag_22], 0.,
                #   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., -dict_pop_full[age_j+tag_21],  0., 0., 0., 0.], ## reciprocity
                  #[0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., dict_pop_full[age_i+tag_22],
                  # 0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., -dict_pop_full[age_j+tag_22]], ## reciprocity

                 [1., 1., 0., 0.,
                   1., 1., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## assortativity sep dimension
                 # [dict_pop_full[age_i+tag_11], 0., dict_pop_full[age_i+tag_11], 0.,
                 #  0., 0., 0., 0.,
                 #  dict_pop_full[age_i+tag_21], 0., dict_pop_full[age_i+tag_21], 0.,
                 #  0., 0., 0., 0.,
                 #  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## assortativity edu dimension
                 ### null values
                  [0., 0., 0., 0.,  1., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 1., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 1., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 1.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  1., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 1., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 1., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 1.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 1., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 1.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 1., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 1.,  0., 0., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 1., 0., 0.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 1.,  0., 0., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 1., 0., 0.],
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,
                   0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 1.],
               ],
                )

    b = np.array([intermediate_matrix_rec.loc[age_i+tag_11][age_j]*dict_pop_full[age_i+tag_11],
                  #intermediate_matrix_rec.iloc[0:4][age_j].iloc[1],
                  intermediate_matrix_rec.loc[age_i+tag_21][age_j]*dict_pop_full[age_i+tag_21],
                  #intermediate_matrix_rec.iloc[0:4][age_j].iloc[3],
                  intermediate_matrix_rec.loc[age_j+tag_11][age_i]*dict_pop_full[age_j+tag_11],
                  intermediate_matrix_rec.loc[age_j+tag_12][age_i]*dict_pop_full[age_j+tag_12],
                  intermediate_matrix_rec.loc[age_j+tag_21][age_i]*dict_pop_full[age_j+tag_21],
                  intermediate_matrix_rec.loc[age_j+tag_22][age_i]*dict_pop_full[age_j+tag_22],
                  q11*intermediate_matrix_rec.loc[age_i+tag_11][age_j]*dict_pop_full[age_i+tag_11],
                  #q12*intermediate_matrix_rec.iloc[0:4][age_j].iloc[1],
                  q21*intermediate_matrix_rec.loc[age_i+tag_21][age_j]*dict_pop_full[age_i+tag_21],
                  #q22*intermediate_matrix_rec.iloc[0:4][age_j].iloc[3],
                  #r1*(1.-q11)*intermediate_matrix_rec.iloc[0:4][age_j].iloc[0],
                  #r2*(1.-q12)*intermediate_matrix_rec.iloc[0:4][age_j].iloc[1],
                  #r3*(1.-q21)*intermediate_matrix_rec.iloc[0:4][age_j].iloc[2],
                  0., 0., 0., 0.,
                  #0., 0., 0., 0.,
                  0., 0., 0., #0.,
                  #0., 0., 0., #0.,
                  q12_sep*(dict_pop_full[age_i+tag_11]*intermediate_matrix_rec.loc[age_i+tag_11][age_j]+dict_pop_full[age_i+tag_12]*intermediate_matrix_rec.loc[age_i+tag_12][age_j]),
                  #q12_edu*(dict_pop_full[age_i+tag_11]*intermediate_matrix_rec.iloc[0:4][age_j].iloc[0]+dict_pop_full[age_i+tag_21]*intermediate_matrix_rec.iloc[0:4][age_j].iloc[2]),
                  ### null values (16)
                  0., 0., 0., 0.,
                  0., 0., 0., 0.,
                  0., 0.,
                  0., 0.,
                  0., 0.,
                  0., 0.,
                 ])

    x = np.linalg.solve(a, b)
    
    return x
    
# ## solve systems with a given combination of parameters

# In[36]:

found = 0

#random.seed(42)
#random.seed(23)

rnd_seed = 29
random.seed(rnd_seed)

num_samples = 2000000

name_file = 'df_qs_01_seed{}_{}.csv'.format(rnd_seed, num_samples)

f = open(name_file, 'w')

f.write('q11,q21,q_sep\n')

for sample in range(num_samples):
    q11 = np.random.uniform(0., 1.)
    #q12 = np.random.uniform(0., 1.)
    q21 = np.random.uniform(0., 1.)
    #q22 = np.random.uniform(0., 1.)
    #r1 = np.random.uniform(0., 0.8)
    #r2 = np.random.uniform(0., 1.)
    #r3 = np.random.uniform(0.2, 1.)
    #q_edu = np.random.uniform(0.47, 0.60)
    q_sep = np.random.uniform(0., 1.)

    x = offdiag_block(q11, q21, q_sep)
    x = np.around(x)
    
    if sample%1000 == 0:
        print(sample)

    if np.all(x>=0.):
    
        found+=1
        
        if found%10 == 0:
            f.close()
            f = open(name_file, 'a')
        
        f.write('{},{},{}\n'.format(q11, q21, q_sep))
