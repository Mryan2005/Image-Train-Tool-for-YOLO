import os
import random
import shutil
import time


def split(root_dir: str, train_ratio: float, valid_ratio: float, test_ratio: float, split_dir: str, workdir: str):
    # 设置随机种子
    random.seed(time.time())

    os.makedirs(os.path.join(split_dir, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'valid/images'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'valid/labels'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'test/images'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'test/labels'), exist_ok=True)

    # 获取图片文件列表
    image_files = os.listdir(os.path.join(root_dir, 'images'))
    label_files = os.listdir(os.path.join(root_dir, 'labels'))

    # 随机打乱文件列表
    combined_files = list(zip(image_files, label_files))
    random.shuffle(combined_files)
    image_files_shuffled, label_files_shuffled = zip(*combined_files)

    # 根据比例计算划分的边界索引
    train_bound = int(train_ratio * len(image_files_shuffled))
    valid_bound = int((train_ratio + valid_ratio) * len(image_files_shuffled))

    train_txt = open(split_dir + '/train/' + 'train.txt', 'w')
    valid_txt = open(split_dir + '/valid/' + 'valid.txt', 'w')

    # 将图片和标签文件移动到相应的目录
    for i, (image_file, label_file) in enumerate(zip(image_files_shuffled, label_files_shuffled)):
        if i <= train_bound:
            shutil.move(os.path.join(root_dir, 'images', image_file),
                        os.path.join(split_dir, 'train/images', image_file))
            shutil.move(os.path.join(root_dir, 'labels', label_file),
                        os.path.join(split_dir, 'train/labels', label_file))
            train_txt.write(workdir + '/train/images/' + image_file + '\n')
        elif i < valid_bound:
            shutil.move(os.path.join(root_dir, 'images', image_file),
                        os.path.join(split_dir, 'valid/images', image_file))
            shutil.move(os.path.join(root_dir, 'labels', label_file),
                        os.path.join(split_dir, 'valid/labels', label_file))
            valid_txt.write(workdir + '/valid/images/' + image_file + '\n')
        # else:
        #     shutil.move(os.path.join(root_dir, 'images', image_file),
        #                 os.path.join(split_dir, 'test/images', image_file))
        #     shutil.move(os.path.join(root_dir, 'labels', label_file),
        #                 os.path.join(split_dir, 'test/labels', label_file))
