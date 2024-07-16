#DATASET:
The dataset is downloaded from [HERE](https://drive.google.com/drive/folders/1znU8wBDiqsMe0lRW5z9gqcsxKPhz1eIq?usp=sharing)

Convert the dataset into this format

# Dataset Folder Structure

Dataset
├───train
│   ├────images
│   │    ├───Clip_1_00001.jpg
│   │    ├───...
│   │    └───Clip_65_00320.jpg
│   ├────labels
│   │    ├───Clip_1_00001.txt
│   │    ├───...
│   │    └───Clip_65_00320.txt
│   └────videos
│        └───video_dict_length.pkl
├───test
│   ├────images
│   │    ├───Clip_1_00001.jpg
│   │    ├───...
│   │    └───Clip_20_00320.jpg
│   ├────labels
│   │    ├───Clip_1_00001.txt
│   │    ├───...
│   │    └───Clip_20_00320.txt
│   └────videos
│        └───video_dict_length.pkl
└───val
    ├────images
    │    ├───Clip_1_00001.jpg
    │    ├───...
    │    └───Clip_10_00326.jpg
    ├────labels
    │    ├───Clip_1_00001.txt
    │    ├───...
    │    └───Clip_10_00326.txt
    └────videos
         └───video_dict_length.pkl

The video_dict_length.pkl should be created using python dictionary format as {"int(video_id)" such as 1,2 : int(num_frames)}
# Train
Create environment using pytorch-ampere.yml file : conda env create --file pytorch-ampere.yml

pip install -r requirements.txt

Go to the root Directory and run this command to train on custom dataset
change the paths in viso.yaml accordingly

python train.py --img 1024 --adam --batch 1 --epochs 80 --data ./data/viso.yaml --weights ./pretrained/yolo5l.pt --hyp ./data/hyps/hyp.VisDrone.yaml --cfg ./models/yolov5l.yaml --device 0 --project ./runs/train/Viso --name Results --exist-ok
# Running pre-trained checkpoints
Best runs are saved in runs/train/Viso/Results

The code is highly borrowed from [TransVisDrone](https://github.com/tusharsangam/TransVisDrone/tree/main)


# References
* [yolov5-tph](https://github.com/cv516Buaa/tph-yolov5)
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)
* [ICPR 2024](https://satvideodt2024.github.io/)
