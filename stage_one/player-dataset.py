with open('train_video_ids', 'r') as input_file:
   # 读取文件中的所有行
   lines = input_file.readlines()
# 遍历每行，提取数字
numbers = []
for line in lines:
   # 使用 split() 方法将每行文本中的数字和逗号分割开
   parts = line.split(',')
   # 提取所有的数字
   numbers.extend(parts[:len(parts)//2])
# 打开新文件，将数字写入
with open('output.txt', 'w') as output_file:
   # 将数字连接成字符串，使用逗号分隔
   output_str = ','.join(map(str, numbers))
   # 将字符串写入文件
   output_file.write(output_str)