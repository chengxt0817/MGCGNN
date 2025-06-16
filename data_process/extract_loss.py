import re

# 定义日志文件路径
log_file_path = "/root/cxt/construct_graph_20241125/graph_walk_80_15/embeddings_0215141251/block_9/log.txt"

# 初始化一个空列表来存储提取的数值
validation_losses = []

try:
    # 打开并读取日志文件
    with open(log_file_path, 'r') as file:
        log_content = file.readlines()  # 逐行读取日志内容

    # 遍历每一行日志内容
    for line in log_content:
        # 使用正则表达式匹配 "Average validation loss:" 后的数字
        match = re.search(r'Average validation loss:\s*([\d.]+)', line)
        if match:
            # 将匹配到的数字转换为浮点数并添加到列表中
            validation_loss = float(match.group(1))
            validation_losses.append(validation_loss)

    # 打印提取的数值数组
    print(validation_losses)

except FileNotFoundError:
    print(f"Error: The file at {log_file_path} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")