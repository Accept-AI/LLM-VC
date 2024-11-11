#a = {"20221107-Cleveland Cavaliers-Los Angeles Clippers": {"img_size": [1280, 720, 3], "img_num": 750, "st_time": 5, "ed_time": 8.8, "Caption": "Defensive rebound by E.Mobley", "bbox": {"E.Mobley": [[692, 409, 799, 619], [708, 390, 798, 617], [705, 255, 791, 570], [700, 250, 801, 538], [730, 304, 800, 547], [724, 341, 832, 554], [737, 334, 853, 541], [745, 291, 846, 534], [730, 289, 846, 532], [712, 312, 880, 531], [701, 307, 852, 520], [682, 279, 840, 518], [685, 280, 808, 512], [658, 271, 824, 503], [670, 278, 779, 501], [652, 261, 840, 489], [0, 0, 0, 0], [692, 251, 855, 487], [737, 263, 854, 497], [757, 244, 916, 488], [0, 0, 0, 0]]}}}
import os
import numpy as np
#a = [8.6529,7.0089,8.6539, 6.5835,8.6793,8.2157,5.0619,8.9334, device='cuda:0'), tensor(10.3737, device='cuda:0'), tensor(9.2813, device='cuda:0'), tensor(12.4820, device='cuda:0'), tensor(4.7754, device='cuda:0'), tensor(5.6964, device='cuda:0'), tensor(4.9173, device='cuda:0')]
import pickle
with open("/home/xzy/xzy_nba/LLM_VC/Player_identify/code/A_data_preprocessing/E_videoid_top2.pkl", "rb") as f:
    loaded_data = pickle.load(f)
#print(loaded_data)
for k, v in loaded_data.items():
    print("----------------")
    print(k)
    hh = list(v.keys())
    print(hh)
    print(hh[0])
    print(hh[1])
    for ki, vi in v.items():
        print(ki)

        print(vi.shape)

# store_dir = os.listdir("/home/xzy/xzy_nba/VG_NBA_Timesformer-features")
# #print(store_dir)
# len_list = []
# for video_id in store_dir:
#     print(video_id)
#     video_path = os.path.join("/home/xzy/xzy_nba/VG_NBA_Timesformer-features", video_id, "out.npy")
#     video_feature = np.load(video_path)
#     print(f"{video_id}: ", video_feature.shape)
#     len_list.append(video_feature.shape[0])
# print(np.max(len_list))   # 200
# print(np.min(len_list))
# print(np.mean(len_list))

# video_mask = np.zeros((1, 10), dtype=np.int_)
# print("video_mask: ", video_mask)