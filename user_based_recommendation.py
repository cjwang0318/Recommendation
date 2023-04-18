#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from src.utils.data_utils import read_purchase_data, read_csv, save_temp_data
from src.similarity import compute_cosine_similarity, compute_jaccard_similarity
from src.ops import get_topk_index, get_recommend_with_topk_user, get_recommend_with_topk_user_score, get_recommend_matrix


# ### hyperparameter

# In[2]:


purchase_path = './郭元益問卷建模資料_0916/purchase_data.csv'
new_user_purchase_path = './郭元益問卷建模資料_0916/new_user_purchase_data.csv'

item_path = './郭元益問卷建模資料_0916/product_data.csv'
user_path = './郭元益問卷建模資料_0916/member_data.csv'

output_path = './郭元益問卷建模資料_0916/output_data.csv'
new_user_output_path = './郭元益問卷建模資料_0916/new_user_output_data.csv'

temp_data_path = './郭元益問卷建模資料_0916/temp_data.npz'

user_sim_output_path = './郭元益問卷建模資料_0916/user_similarity.csv'
item_sim_output_path = './郭元益問卷建模資料_0916/item_similarity.csv'

index_col = 'MEMBER_ID'
encoding = 'Big5'
top_k_user = 20 # 使用相似度前幾名的user
top_k_item = 15 # 使用前幾名的推薦商品

id2name_purchase = {0:'預測不喜歡', 1:'真實喜歡', 2:'預測喜歡'}


# # recommend item to additional user with similarity
# ### read file and save temporary data

#使用者(ID)購買過哪些Item
purchase_df, purchase_matrix = read_purchase_data(purchase_path, index_col=index_col)

#取出使用者ID
member_id = purchase_df.index
#讀取產品屬性：產品ID, 屬性1, 屬性2, 屬性3...
item_df = read_csv(item_path, encoding=encoding)
#讀取客戶屬性：客戶ID, 屬性1, 屬性2, 屬性3...
user_df = read_csv(user_path, encoding=encoding)


# save temporary data 使用numpy.savez保存字典
data_dict = {'purchase_matrix' : purchase_matrix}
save_temp_data(temp_data_path, data_dict)


# ### recommend item with similarity

# In[5]:


user_similarity = compute_cosine_similarity(purchase_matrix)
item_similarity = compute_jaccard_similarity(purchase_matrix.T, purchase_matrix.T)


# In[6]:


topk_index = get_topk_index(user_similarity, top_k_user)


# In[7]:


ranked_item_index, _ = get_recommend_with_topk_user(purchase_matrix, topk_index)
# item_index_with_score, _ = get_recommend_with_topk_user_score(purchase_matrix, topk_index, user_similarity)


# In[8]:


recommend_item_index = ranked_item_index[:, :top_k_item].tolist()
recommend_matrix = get_recommend_matrix(purchase_matrix, recommend_item_index)


# In[9]:


recommend_matrix = recommend_matrix.tolist()
for c_idx, col in enumerate(recommend_matrix):
    for r_idx, element in enumerate(col):
        recommend_matrix[c_idx][r_idx] = id2name_purchase[element]


# ### output recomendation result

# In[11]:


output_df = pd.DataFrame(recommend_matrix, index=member_id)

user_similarity_df = pd.DataFrame(user_similarity)
item_similarity_df = pd.DataFrame(item_similarity)


# In[12]:


output_df.to_csv(output_path, encoding=encoding)
user_similarity_df.to_csv(user_sim_output_path)
item_similarity_df.to_csv(item_sim_output_path)


# ---

# # recommend item to additional user with similarity
# ### read new file and load temporary data

# In[13]:


added_purchase_df, added_purchase_matrix = read_purchase_data(new_user_purchase_path, index_col=index_col)
added_member_id = added_purchase_df.index


# In[14]:


temp_data = np.load(temp_data_path)
purchase_matrix = temp_data['purchase_matrix']
purchase_matrix = np.concatenate([purchase_matrix, added_purchase_matrix])


# In[15]:


# save temporary data
data_dict = {'purchase_matrix' : purchase_matrix}
save_temp_data(temp_data_path, data_dict)


# ### recommend item with similarity

# In[16]:


user_similarity = compute_cosine_similarity(purchase_matrix, added_purchase_matrix).T
item_similarity = compute_jaccard_similarity(purchase_matrix.T, purchase_matrix.T)


# In[17]:


topk_index = get_topk_index(user_similarity, top_k_user)


# In[18]:


ranked_item_index, _ = get_recommend_with_topk_user(purchase_matrix, topk_index, target_purchase_matrix=added_purchase_matrix)
# item_index_with_score, _ = get_recommend_with_topk_user_score(purchase_matrix, topk_index, user_similarity)


# In[19]:


recommend_item_index = ranked_item_index[:, :top_k_item].tolist()
recommend_matrix = get_recommend_matrix(added_purchase_matrix, recommend_item_index)


# In[20]:


recommend_matrix = recommend_matrix.tolist()
for c_idx, col in enumerate(recommend_matrix):
    for r_idx, element in enumerate(col):
        recommend_matrix[c_idx][r_idx] = id2name_purchase[element]


# ### output recomendation result

# In[21]:


output_df = pd.DataFrame(recommend_matrix, index=added_member_id)


# In[22]:


output_df.to_csv(new_user_output_path, encoding=encoding)

