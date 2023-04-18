import numpy as np
import pandas as pd
from src.utils.data_utils import read_purchase_data, read_csv, save_temp_data
from src.similarity import compute_cosine_similarity, compute_jaccard_similarity
from src.ops import get_topk_index, get_recommend_with_topk_user, get_recommend_matrix

purchase_path = './Expo_data/booth_data.csv'
temp_data_path = './Expo_data/temp_data.npz'

output_path = './Expo_data/output_data.csv'
user_sim_output_path = './Expo_data/user_similarity.csv'

index_col = 'BOOTH_ID'
encoding = 'Big5'
top_k_user = 20  # 使用相似度前幾名的攤位booth

# 攤位ID(column), 攤位關鍵字(row)
purchase_df, purchase_matrix = read_purchase_data(purchase_path, index_col=index_col)
# print(purchase_df)
# 取出使用者ID
member_id = purchase_df.index

# save temporary data 使用numpy.savez保存字典
data_dict = {'purchase_matrix': purchase_matrix}
save_temp_data(temp_data_path, data_dict)

# recommend item with similarity
user_similarity = compute_cosine_similarity(purchase_matrix)

# 挑選相似度top k個使用者當作推薦依據
topk_index = get_topk_index(user_similarity, top_k_user)

# output recommendation result

output_df = pd.DataFrame(topk_index, index=member_id)
user_similarity_df = pd.DataFrame(user_similarity)

# 輸出csv檔案
output_df.to_csv(output_path, encoding=encoding)
user_similarity_df.to_csv(user_sim_output_path)
