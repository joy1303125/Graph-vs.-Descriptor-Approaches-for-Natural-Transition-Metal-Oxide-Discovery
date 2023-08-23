#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pymatgen
from pymatgen.ext.matproj import MPRester


# # transitional_oxide_bi

# In[ ]:


with MPRester("mv0hJgupKqxOTd9c") as mpr:
    transitional_metals = ['Ti','Nb','W','Cu','Fe','V','Ni','Mn','Co','Cr','Zn','Sc','Pd','Ru','Pt','Au','Hf','Ir','Ag','W','Ta','Y','Zr','Hg','Cd']
    criteria={"elements":{"$in":transitional_metals, "$all": ["O"]}, "nelements":2,'icsd_ids': {'$gte': 0}
            }
    properties=['material_id', 'pretty_formula','energy_per_atom','formation_energy_per_atom','structure','e_above_hull','band_gap','crystal_system','energy']
    df_transitional_oxide_bi=(mpr.query(criteria, properties))


# In[ ]:


df_transitional_oxide_bi=pd.DataFrame(df_transitional_oxide_bi)


# In[ ]:


df_transitional_oxide_bi["Category_label"] = np.nan
df_transitional_oxide_bi['Category_label'] = df_transitional_oxide_bi['Category_label'].replace(np.nan, 1)
df_transitional_oxide_bi["Class_name"] = "Binary_oxide"


# In[ ]:


# from matminer.featurizers.structure import DensityFeatures
density=DensityFeatures()
DensityFeatures_bi=density.featurize_dataframe(df_transitional_oxide_bi,"structure")
DensityFeatures_bi.head(3)


# # transitional_oxide_tri

# In[ ]:


with MPRester("mv0hJgupKqxOTd9c") as mpr:
    transitional_metals = ['Ti','Nb','W','Cu','Fe','V','Ni','Mn','Co','Cr','Zn','Sc','Pd','Ru','Pt','Au','Hf','Ir','Ag','W','Ta','Y','Zr','Hg','Cd']
    criteria={"elements":{"$in":transitional_metals, "$all": ["O"]}, "nelements":3,'icsd_ids': {'$gte': 0}}
    properties=['material_id', 'pretty_formula','energy_per_atom','formation_energy_per_atom','structure','e_above_hull','band_gap','crystal_system','energy']
    df_transitional_oxide_tri=(mpr.query(criteria, properties))


# In[ ]:


df_transitional_oxide_tri=pd.DataFrame(df_transitional_oxide_tri)


# In[ ]:


df_transitional_oxide_tri["Category_label"] = np.nan
df_transitional_oxide_tri['Category_label'] = df_transitional_oxide_tri['Category_label'].replace(np.nan, 2)
df_transitional_oxide_tri["Class_name"] = "Ternary_oxide"


# In[ ]:


from matminer.featurizers.structure import DensityFeatures
density=DensityFeatures()
DensityFeatures_tri=density.featurize_dataframe(df_transitional_oxide_tri,"structure")
DensityFeatures_tri.head(3)


# # transitional_oxide_tetra

# In[ ]:


with MPRester("mv0hJgupKqxOTd9c") as mpr:
    transitional_metals = ['Ti','Nb','W','Cu','Fe','V','Ni','Mn','Co','Cr','Zn','Sc','Pd','Ru','Pt','Au','Hf','Ir','Ag','W','Ta','Y','Zr','Hg','Cd']
    criteria={"elements":{"$in":transitional_metals, "$all": ["O"]}, "nelements":4,'icsd_ids': {'$gte': 0}}
    properties=['material_id', 'pretty_formula','energy_per_atom','formation_energy_per_atom','structure','e_above_hull','band_gap','crystal_system','energy']
    df_transitional_oxide_tetra=(mpr.query(criteria, properties))


# In[ ]:


df_transitional_oxide_tetra=pd.DataFrame(df_transitional_oxide_tetra)


# In[ ]:


df_transitional_oxide_tetra["Category_label"] = np.nan
df_transitional_oxide_tetra['Category_label'] = df_transitional_oxide_tetra['Category_label'].replace(np.nan, 3)
df_transitional_oxide_tetra["Class_name"] = "Tetra_oxide"


# In[ ]:


from matminer.featurizers.structure import DensityFeatures
density=DensityFeatures()
DensityFeatures_tetra=density.featurize_dataframe(df_transitional_oxide_tetra,"structure",ignore_errors=True)
DensityFeatures_tetra.head(3)


# # transitional_oxide_quinary

# In[ ]:


with MPRester("mv0hJgupKqxOTd9c") as mpr:
    transitional_metals = ['Ti','Nb','W','Cu','Fe','V','Ni','Mn','Co','Cr','Zn','Sc','Pd','Ru','Pt','Au','Hf','Ir','Ag','W','Ta','Y','Zr','Hg','Cd']
    criteria={"elements":{"$in":transitional_metals, "$all": ["O"]}, "nelements":5,'icsd_ids': {'$gte': 0}}
    properties=['material_id', 'pretty_formula','energy_per_atom','formation_energy_per_atom','structure','e_above_hull','band_gap','crystal_system','energy']
    df_transitional_oxide_quinary=(mpr.query(criteria, properties))


# In[ ]:


df_transitional_oxide_quinary=pd.DataFrame(df_transitional_oxide_quinary)
df_transitional_oxide_quinary["Category_label"] = np.nan
df_transitional_oxide_quinary['Category_label'] = df_transitional_oxide_quinary['Category_label'].replace(np.nan, 4)
df_transitional_oxide_quinary["Class_name"] = "quinary_oxide"


# In[ ]:


from matminer.featurizers.structure import DensityFeatures
density=DensityFeatures()
DensityFeatures_quinary=density.featurize_dataframe(df_transitional_oxide_quinary,"structure",ignore_errors=True)
DensityFeatures_quinary


# # transitional_oxide_seanary

# In[3]:


with MPRester("mv0hJgupKqxOTd9c") as mpr:
    transitional_metals = ['Ti','Nb','W','Cu','Fe','V','Ni','Mn','Co','Cr','Zn','Sc','Pd','Ru','Pt','Au','Hf','Ir','Ag','W','Ta','Y','Zr','Hg','Cd']
    criteria={"elements":{"$in":transitional_metals, "$all": ["O"]}, "nelements":6,'icsd_ids': {'$gte': 0}}
    properties=['material_id', 'pretty_formula','energy_per_atom','formation_energy_per_atom','structure','e_above_hull','band_gap','crystal_system','energy']
    df_transitional_oxide_seanary=(mpr.query(criteria, properties))


# In[4]:


df_transitional_oxide_seanary=pd.DataFrame(df_transitional_oxide_seanary)
df_transitional_oxide_seanary["Category_label"] = np.nan
df_transitional_oxide_seanary['Category_label'] = df_transitional_oxide_seanary['Category_label'].replace(np.nan, 6)
df_transitional_oxide_seanary["Class_name"] = "sexanary_oxide"


# In[5]:


from matminer.featurizers.structure import DensityFeatures
density=DensityFeatures()
DensityFeatures_seanary=density.featurize_dataframe(df_transitional_oxide_seanary,"structure",ignore_errors=True)
DensityFeatures_seanary


# In[ ]:


df_orbital=[df_transitional_oxide_bi, df_transitional_oxide_tri, df_transitional_oxide_tetra,df_transitional_oxide_quinary,df_transitional_oxide_seanary]


# In[ ]:



result = pd.concat(df_orbital)


# In[ ]:


result.reset_index(drop=True, inplace=True)

result


# # XRDPowderPattern

# In[ ]:


from matminer.featurizers.structure import XRDPowderPattern
XRDPowder= XRDPowderPattern()
XRDPowderPattern= XRDPowder.featurize_dataframe(result,"structure")
XRDPowderPattern.head(3) 


# # Magpie features

# In[7]:


from pymatgen.core.structure import Structure  # Corrected import


# In[13]:


from matminer.featurizers.composition import ElementProperty


# In[ ]:


import os
from pymatgen.ext.matproj import MPRester
from pymatgen.core.structure import Structure
from matminer.featurizers.composition import ElementProperty
import pandas as pd

# Replace 'YOUR_API_KEY' with your actual Materials Project API key
api_key = 'mv0hJgupKqxOTd9c'
mpr = MPRester(api_key)

# Assuming you have a DataFrame named 'results' with material_id and formation_energy_per_atom columns
material_ids = result['material_id'].tolist()
  # Take only the first 3 material IDs
magpie_features_list = []
cif_files = []

# Initialize the ElementProperty featurizer with the Magpie preset
ep_feat = ElementProperty.from_preset("magpie")

for material_id in material_ids:
    structure = mpr.get_structure_by_material_id(material_id)
    cif_file = os.path.join("./", f"{material_id}.cif")
    structure.to(fmt="cif", filename=cif_file)
    cif_files.append(cif_file)
    comp_features = ep_feat.featurize(structure.composition.fractional_composition)
    magpie_features_list.append(comp_features)

formation_energies =result['formation_energy_per_atom'].tolist()  # Take only the first 3 formation energies

# Create a new DataFrame with separate columns for each Magpie feature
new_columns = [str(i) for i in range(len(magpie_features_list[0]))]
new_df = pd.DataFrame(magpie_features_list, columns=new_columns)
new_df.insert(0, 'file_name', cif_files)
new_df['formation_energy'] = formation_energies

new_df


# In[ ]:





# In[ ]:




