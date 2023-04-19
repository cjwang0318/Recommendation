import numpy as np
import pandas as pd
import json
from src.utils.data_utils import read_purchase_data, read_csv, save_temp_data
from src.similarity import compute_cosine_similarity, compute_jaccard_similarity
from src.ops import get_topk_index, get_recommend_with_topk_user, get_recommend_matrix

booth_path = './Expo_data/booth_data.csv'
temp_data_path = './Expo_data/temp_data.npz'

output_path = './Expo_data/expo_recom_dict.json'
booth_sim_output_path = './Expo_data/booth_similarity.csv'

index_col = 'BOOTH_ID'
encoding = 'utf-8'
top_k_booth = 20  # 使用相似度前幾名的攤位booth


def recommendation_training():
    # 攤位ID(column), 攤位關鍵字(row)
    booth_df, booth_matrix = read_purchase_data(booth_path, index_col=index_col)
    # print(purchase_df)
    # 取出Booth ID
    booth_id = booth_df.index

    # save temporary data 使用numpy.savez保存字典
    data_dict = {'purchase_matrix': booth_matrix}
    save_temp_data(temp_data_path, data_dict)

    # recommend item with similarity
    user_similarity = compute_cosine_similarity(booth_matrix)

    # 挑選相似度top k個使用者當作推薦依據
    topk_index = get_topk_index(user_similarity, top_k_booth)

    # output recommendation result
    output_df = pd.DataFrame(topk_index, index=booth_id)
    user_similarity_df = pd.DataFrame(user_similarity)

    # 建立推薦名單字典
    expo_recom_dict = output_df.T.to_dict('list')

    # 輸出csv檔案 and 給推薦系統使用的Json檔
    output_df.to_csv(output_path, encoding=encoding)
    user_similarity_df.to_csv(booth_sim_output_path)
    with open(output_path, "w", encoding=encoding) as fp:
        json.dump(expo_recom_dict, fp, indent=2)  # encode dict into JSON


if __name__ == '__main__':
    recommendation_training()
