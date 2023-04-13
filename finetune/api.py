import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from log.log import MyLogger
import configparser
import json
import os
import base64

import toml
from fastapi import FastAPI, HTTPException
from starlette.staticfiles import StaticFiles



app = FastAPI()
# app.mount('/static', StaticFiles(directory='static'),
#           name='static')
LOG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../log', 'log.txt')

# API端点，用于查询日志信息
@app.get("/logs")
async def read_logs():
    logs = []
    # 读取日志文件
    with open(LOG_FILE, 'r') as f:
        for line in f.readlines():
            try:
                # 将每一行日志信息解析为JSON格式，并添加到列表中
                logs.append(line)
            except Exception as e:
                print(f"Error parsing log line: {e}")
    # 返回日志信息列表
    return logs

def decode_base64_to_file(base64_string, filename):
    with open(filename, 'wb') as f:
        f.write(base64.b64decode(base64_string))


def aimalltrainlora(train_data: dict, reg_data: dict):
    train_folder_name = '../dataset/lora'
    reg_folder_name = '../dataset/lora_reg'
    trainout_folder_name = '../dataset/lora_out'
    train_class = train_data.get("train_class")
    train_tag = train_data.get("tags")
    train_images = train_data.get("images")
    reg_images = reg_data.get("images")
    reg_class = reg_data.get("reg_class")
    train_num = train_data.get("num")
    reg_num = reg_data.get("num")
    if not os.path.exists(trainout_folder_name):
        os.mkdir(trainout_folder_name)
    # remove existing train dataset folder
    train_folder_path = f"{train_folder_name}"
    if os.path.exists(train_folder_path):
        for folder in os.listdir(train_folder_path):
            folder_path = os.path.join(train_folder_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
                os.rmdir(folder_path)
    else:
        os.mkdir(train_folder_path)

    # remove existing reg dataset folder
    reg_folder_path = f"{reg_folder_name}"
    if os.path.exists(reg_folder_path):
        for folder in os.listdir(reg_folder_path):
            folder_path = os.path.join(reg_folder_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
                os.rmdir(folder_path)
    else:
        os.mkdir(reg_folder_path)

    for i in range(len(train_class)):
        folder_name = f"{train_num}_{train_tag[i]} {train_class[i]}"
        folder_path = os.path.join(train_folder_path, folder_name)
        os.mkdir(folder_path)

        for j in range(len(train_images[i])):
            image_name = f"{j}.jpg"
            image_path = os.path.join(folder_path, image_name)
            decode_base64_to_file(train_images[i][j], image_path)

    for i in range(len(reg_class)):
        folder_name = f"{reg_num}_{reg_class[i]}"
        folder_path = os.path.join(reg_folder_path, folder_name)
        os.mkdir(folder_path)

        for j in range(len(reg_images[i])):
            image_name = f"{j}.jpg"
            image_path = os.path.join(folder_path, image_name)
            decode_base64_to_file(reg_images[i][j], image_path)

    return {"message": "Images saved successfully."}


def aimalltraindeambooth(train_data: dict, reg_data: dict):
    train_folder_name = '../dataset/dreambooth'
    reg_folder_name = '../dataset/dreambooth_reg'
    trainout_folder_name = '../dataset/dreambooth_out'
    train_class = train_data.get("train_class")
    train_tag = train_data.get("tags")
    train_images = train_data.get("images")
    reg_images = reg_data.get("images")
    reg_class = reg_data.get("reg_class")
    train_num = train_data.get("num")
    reg_num = reg_data.get("num")
    if not os.path.exists(trainout_folder_name):
        os.mkdir(trainout_folder_name)
    # remove existing train dataset folder
    train_folder_path = f"{train_folder_name}"
    if os.path.exists(train_folder_path):
        for folder in os.listdir(train_folder_path):
            folder_path = os.path.join(train_folder_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
                os.rmdir(folder_path)
    else:
        os.mkdir(train_folder_path)

    # remove existing reg dataset folder
    reg_folder_path = f"{reg_folder_name}"
    if os.path.exists(reg_folder_path):
        for folder in os.listdir(reg_folder_path):
            folder_path = os.path.join(reg_folder_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
                os.rmdir(folder_path)
    else:
        os.mkdir(reg_folder_path)

    for i in range(len(train_class)):
        folder_name = f"{train_num}_{train_tag[i]} {train_class[i]}"
        folder_path = os.path.join(train_folder_path, folder_name)
        os.mkdir(folder_path)

        for j in range(len(train_images[i])):
            image_name = f"{j}.jpg"
            image_path = os.path.join(folder_path, image_name)
            decode_base64_to_file(train_images[i][j], image_path)

    for i in range(len(reg_class)):
        folder_name = f"{reg_num}_{reg_class[i]}"
        folder_path = os.path.join(reg_folder_path, folder_name)
        os.mkdir(folder_path)

        for j in range(len(reg_images[i])):
            image_name = f"{j}.jpg"
            image_path = os.path.join(folder_path, image_name)
            decode_base64_to_file(reg_images[i][j], image_path)

    return {"message": "Images saved successfully."}


@app.post("/aimall/v1/deamboothorlora")
def aimalltraindeamboothorlora(base_model, isdeambooth: bool, train_data: dict, reg_data: dict):
    import subprocess
    logger = MyLogger(log_file='../log/log.txt')
    if isdeambooth:
        st="deambooth"
    else:
        st='lora'
    logger.info("开始训练模型,"+"method:"+st)

    try:
        if isdeambooth:
            aimalltraindeambooth(train_data, reg_data)
            train_data_dir = '../dataset/dreambooth/'
            reg_data_dir = '../dataset/dreambooth_reg/'
            out='../dataset/dreambooth_out'
            enable_bucket = True
            resolution = 512
            batch_size = 1

            caption_extension = '.caption'

            is_reg = True
            train_num = train_data.get("num")
            reg_num = reg_data.get("num")
            train_class = train_data.get("train_class")
            train_tag = train_data.get("tags")
            reg_class = reg_data.get("reg_class")
            config_file='./dreamboothconfig.toml'
            with open(config_file, "r") as f:
                config = toml.load(f)
            # 修改配置
            # 修改配置文件中的值
            # 更新配置
            config['general']['enable_bucket'] = enable_bucket
            config['datasets'][0]['resolution'] = resolution
            config['datasets'][0]['batch_size'] = batch_size
            config['datasets'][0]['subsets'].clear()
            for i in range(len(train_class)):
                folder_name = f"{train_num}_{train_tag[i]} {train_class[i]}"
                new_subset = {
                    'image_dir':train_data_dir+folder_name,
                    'caption_extension': caption_extension,
                    'class_tokens': train_class[i],
                    'num_repeats': train_num
                }
                config['datasets'][0]['subsets'].append(new_subset)
            for i in range(len(reg_class)):
                folder_name = f"{reg_num}_{reg_class[i]}"
                new_subset = {
                    'is_reg': is_reg,
                    'image_dir': reg_data_dir+folder_name,
                    'class_tokens': reg_class[i],
                    'num_repeats': reg_num
                }
                config['datasets'][0]['subsets'].append(new_subset)

            # 将更新后的配置写回到.toml文件
            with open("dreamboothconfig.toml", "w") as f:
                toml.dump(config, f)
                logger.info("配置dreamboothconfig.toml文件")
            for i in range(len(train_class)):
                folder_name = f"{train_num}_{train_tag[i]} {train_class[i]}"
                cmd = ["python", "./tag_images_by_wd14_tagger.py",
                   "--batch_size", "4","--caption_extension",".caption", train_data_dir+folder_name]
                subprocess.run(cmd)
                logger.info("图片批量打标签:" +  train_data_dir+folder_name)
            for i in range(len(reg_class)):
                folder_name = f"{reg_num}_{reg_class[i]}"
                cmd = ["python", "./tag_images_by_wd14_tagger.py",
                   "--batch_size", "4","--caption_extension",".caption", reg_data_dir+folder_name]
                subprocess.run(cmd)
                logger.info("图片批量打标签:"+reg_data_dir+folder_name)
            cmd = [
                "python", "../train_db.py",
                "--pretrained_model_name_or_path="+base_model,
                "--dataset_config=./dreamboothconfig.toml",
                "--output_dir="+out,
                "--output_name=dreamboothmodel"
                "--save_model_as=safetensors"
                "--prior_loss_weight=1.0",
                "--learning_rate=1e-6",
                "--max_train_steps=1600",
                "--use_8bit_adam",
                "--mixed_precision=fp16",
                "--xformers",
                "--cache_latents",
                "--gradient_checkpointing"
            ]
            logger.info("开始训练......")
            subprocess.run(cmd, check=True)
            logger.info("训练结束：输出模型dreamboothmodel")
        else:
            aimalltrainlora(train_data,reg_data)
            train_data_dir = '../dataset/lora/'
            reg_data_dir = '../dataset/lora_reg/'
            out = '../dataset/lora_out'
            enable_bucket = True
            resolution = 512
            batch_size = 1

            caption_extension = '.caption'

            is_reg = True
            train_num = train_data.get("num")
            reg_num = reg_data.get("num")
            train_class = train_data.get("train_class")
            train_tag = train_data.get("tags")
            reg_class = reg_data.get("reg_class")
            config_file = './loraconfig.toml'
            with open(config_file, "r") as f:
                config = toml.load(f)
            # 修改配置
            # 修改配置文件中的值
            # 更新配置
            config['general']['enable_bucket'] = enable_bucket
            config['datasets'][0]['resolution'] = resolution
            config['datasets'][0]['batch_size'] = batch_size
            config['datasets'][0]['subsets'].clear()
            for i in range(len(train_class)):
                folder_name = f"{train_num}_{train_tag[i]} {train_class[i]}"
                new_subset = {
                    'image_dir': train_data_dir + folder_name,
                    'caption_extension': caption_extension,
                    'class_tokens': train_class[i],
                    'num_repeats': train_num
                }
                config['datasets'][0]['subsets'].append(new_subset)
            for i in range(len(reg_class)):
                folder_name = f"{reg_num}_{reg_class[i]}"
                new_subset = {
                    'is_reg': is_reg,
                    'image_dir': reg_data_dir + folder_name,
                    'class_tokens': reg_class[i],
                    'num_repeats': reg_num
                }
                config['datasets'][0]['subsets'].append(new_subset)

            # 将更新后的配置写回到.toml文件
            with open("loraconfig.toml", "w") as f:
                toml.dump(config, f)
                logger.info("配置loraconfig.toml文件")
            for i in range(len(train_class)):
                folder_name = f"{train_num}_{train_tag[i]} {train_class[i]}"
                cmd = ["python", "./tag_images_by_wd14_tagger.py",
                   "--batch_size", "4","--caption_extension",".caption", train_data_dir+folder_name]
                subprocess.run(cmd)
                logger.info("图片批量打标签:" + train_data_dir + folder_name)
            for i in range(len(reg_class)):
                folder_name = f"{reg_num}_{reg_class[i]}"
                cmd = ["python", "./tag_images_by_wd14_tagger.py",
                   "--batch_size", "4","--caption_extension",".caption", reg_data_dir+folder_name]
                subprocess.run(cmd)
                logger.info("图片批量打标签:" + reg_data_dir + folder_name)
            cmd = ['python', '../train_network.py',
                   '--pretrained_model_name_or_path='+base_model,
                   '--dataset_config=./loraconfig.toml',
                   '--output_dir='+out,
                   '--output_name=loramodel',
                   '--save_model_as=safetensors',
                   '--prior_loss_weight=1.0',
                   '--learning_rate=1e-4',
                   '--max_train_steps=400',
                   '--optimizer_type=AdamW8bit',
                   '--xformers',
                   '--mixed_precision=fp16',
                   '--cache_latents',
                   '--gradient_checkpointing',
                   '--save_every_n_epochs=1',
                   '--network_module=networks.lora']
            logger.info("开始训练......")
            subprocess.run(cmd)
            logger.info("训练结束：输出模型loramodel")
    except Exception as e:
        logger.error({"message": e})



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)