import os
import json

"""
Give VideoID to each sub_file
"""
start = 100000
end = 300000
VideoID = [f"Video{i}" for i in range(start, end + 1)]
file_dict = {}
# print(VideoID)
i = 0
j_list = []
m_list = []
d_list = []
save_path = '/home/xzy/xzy/caption/LLM_VC/Player_identify/Save/'
# 创建json
new_file = open(save_path + 'A_VideoID_path.json', mode='a', encoding='utf-8')

# 遍历当前文件夹中的所有子文件夹
for dirpath, dirnames, filenames in os.walk('/media/xzy/58bff591-e818-4a4b-8c3e-e857128495ed/xzy/VG_NBA_2024/'):
    # print("dirpath: ", dirpath)
    # print("dirnames: ", dirnames)
    # print("filenames: ", filenames)
    # file_path = os.path.join(dirpath, dirnames)
    #print("-------")
    #print(dirpath)
    if len(os.listdir(dirpath)) > 9:
        pass
    else:
        #print(dirpath)
        #print(os.listdir(dirpath))
        a = dirpath
        num = a.split("/")[-1]
        #print(num)
        str_path = os.listdir(dirpath)
        if (num + ".json") not in os.listdir(dirpath):
            #print("not json: ", dirpath)
            j_list.append(dirpath)
        if (num + ".mp4") not in os.listdir(dirpath):
            #print("not mp4: ", dirpath)
            m_list.append(dirpath)
        if (num + ".json") not in os.listdir(dirpath) and (num + ".mp4") not in os.listdir(dirpath):
            d_list.append(dirpath)
        if (num + ".json") in os.listdir(dirpath) and (num + ".mp4") in os.listdir(dirpath):
            file_dict[VideoID[i]] = dirpath
            i = i + 1
print("not json: ", len(j_list))
print("not mp4: ", len(m_list))
print("not double: ", len(d_list))
print("json: ", j_list)
print("mp4: ", m_list)
print("d: ", d_list)

json_save = json.dumps(file_dict, ensure_ascii=False)
new_file.write(json_save)
new_file.close()
print("保存！")
    # if ".json" not in os.listdir(dirpath):
    #     print("not json: ", dirpath)
    # if ".mp4" not in os.listdir(dirpath):
    #     print("not mp4: ", dirpath)
    # for file in os.listdir(dirpath):
    #     print(file)
    #     if ".json" not in file and :
    #         j_list.append("*")
    #     if ".mp4" in file:
    #         m_list.append("*")
# print("json: ", len(j_list))   # json:  9943
# print("mp4: ", len(m_list))   # mp4:  9961


    # for f_i in filenames:
    #     files = dirpath + "/" + f_i
    #     print(files)
