# 数据格式说明
对于任一数据集，其中的一条数据（即一个pair）需要被处理为一个npz文件，存储一个dict，包含：
- instruction字段：一个str
- images字段：存储为(num_frames, H, W, C)的图片序列 uint8 array