#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

# # expanded contact matrix

# ## functions with linear systems

# ### diagonal blocks (16 elements)

#x111_111, x111_112, x111_121, x111_122
#x112_111, x112_112, x112_121, x112_122
#x121_111, x121_112, x121_121, x121_122
#x122_111, x122_112, x122_121, x122_122

def diag_block(q11, q12, q21, q22, q_sep, q_edu):

    age = '65+'

    tag_11 = ', low SEP, low edu'
    tag_12 = ', low SEP, high edu'
    tag_21 = ', high SEP, low edu'
    tag_22 = ', high SEP, high edu'

    # 16 unknowns
    # 4 conditions on the sum per row
    # 6 conditions on the reciprocity
    # 4 conditions on the assortativity on the diagonal
    # 1 condition on the assortativity in the edu dimension
    # 1 condition on the assortativity in the sep dimension

    a = np.array([[1., 1., 1., 1.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## sum
                  [0., 0., 0., 0.,  1., 1., 1., 1.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## sum
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  1., 1., 1., 1.,  0., 0., 0., 0.], ## sum
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  1., 1., 1., 1.], ## sum
                  [0., 1., 0., 0.,
                   -1., 0., 0., 0.,
                   0., 0., 0., 0.,
                   0., 0., 0., 0.], ## reciprocity
                  [0., 0., 1., 0.,
                   0., 0., 0., 0.,
                   -1., 0., 0., 0.,
                   0., 0., 0., 0.], ## reciprocity
                  [0., 0., 0., 1.,
                   0., 0., 0., 0.,
                   0., 0., 0., 0.,
                   -1., 0., 0., 0.], ## reciprocity
                  [0., 0., 0., 0.,
                   0., 0., 1., 0.,
                   0., -1., 0., 0.,
                   0., 0., 0., 0.], ## reciprocity
                  [0., 0., 0., 0.,
                   0., 0., 0., 1.,
                   0., 0., 0., 0.,
                   0., -1., 0., 0.], ## reciprocity
                  [0., 0., 0., 0.,
                   0., 0., 0., 0.,
                   0., 0., 0., 1.,
                   0., 0., -1., 0.], ## reciprocity
                  [1., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## assortativity 1st row
                  [0., 0., 0., 0.,  0., 1., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.], ## assortativity 2nd row
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 1., 0.,  0., 0., 0., 0.], ## assortativity 3rd row
                  [0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 0.,  0., 0., 0., 1.], ## assortativity 4th row
                  [1., 1., 0., 0.,
                   1., 1., 0., 0.,
                   0., 0., 0., 0.,
                   0., 0., 0., 0.], ## assortativity sep dimension
                  [1., 0., 1., 0.,
                   0., 0., 0., 0.,
                   1., 0., 1., 0.,
                   0., 0., 0., 0.], ## assortativity edu dimension


                 ])## assortativity
    b = np.array([intermediate_matrix_rec.loc[age+tag_11][age]*dict_pop_full[age+tag_11],
                  intermediate_matrix_rec.loc[age+tag_12][age]*dict_pop_full[age+tag_12],
                  intermediate_matrix_rec.loc[age+tag_21][age]*dict_pop_full[age+tag_21],
                  intermediate_matrix_rec.loc[age+tag_22][age]*dict_pop_full[age+tag_22],
                  0., 0., 0., 0., 0., 0.,
                  q11*intermediate_matrix_rec.loc[age+tag_11][age]*dict_pop_full[age+tag_11],
                  q12*intermediate_matrix_rec.loc[age+tag_12][age]*dict_pop_full[age+tag_12],
                  q21*intermediate_matrix_rec.loc[age+tag_21][age]*dict_pop_full[age+tag_21],
                  q22*intermediate_matrix_rec.loc[age+tag_22][age]*dict_pop_full[age+tag_22],
                  q_sep*(dict_pop_full[age+tag_11]*intermediate_matrix_rec.loc[age+tag_11][age]+dict_pop_full[age+tag_12]*intermediate_matrix_rec.loc[age+tag_12][age]),
                  q_edu*(dict_pop_full[age+tag_11]*intermediate_matrix_rec.loc[age+tag_11][age]+dict_pop_full[age+tag_21]*intermediate_matrix_rec.loc[age+tag_21][age])])

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

name_file = 'df_qs_3_seed{}_{}_constrained.csv'.format(rnd_seed, num_samples)

f = open(name_file, 'w')

f.write('q11,q12,q21,q22,q_sep,q_edu\n')

for sample in range(num_samples):
    q11 = np.random.uniform(0., 1.)
    q12 = np.random.uniform(0., 1.)
    q21 = np.random.uniform(0., 1.)
    q22 = np.random.uniform(0., 1.)
    q_sep = np.random.uniform(0., 1.0)
    q_edu = np.random.uniform(0.55, 1.0)

    x = diag_block(q11, q12, q21, q22, q_sep, q_edu)
    x = np.around(x)
    
    if sample%1000 == 0:
        print(sample)

    if np.all(x>=0.):
    
        found+=1
        
        if found%10 == 0:
            f.close()
            f = open(name_file, 'a')
        
        f.write('{},{},{},{},{},{}\n'.format(q11, q12, q21, q22, q_sep, q_edu))
