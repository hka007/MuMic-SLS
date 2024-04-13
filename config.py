import torch


class CFG:
    debug = False

    device = "mps"

    dataset = "Advision"

    # '''''''''''''''Advision Dataset 100 Labels'''''''''''''''
    # captions_path = "AdvisionDataset100Labels/Advision_ML.csv"
    # captions_path_train = "AdvisionDataset100Labels/train_dataset.csv"
    # captions_path_eval = "AdvisionDataset100Labels/test_dataset.csv"
    # all_label = "AdvisionDataset100Labels/captions.csv"
    #
    # train_path = "/Users/hka-private/PycharmProjects/Advision/all_transformed_images/"
    # eval_path = "/Users/hka-private/PycharmProjects/Advision/all_transformed_images/"

    # '''''''''''''''Advision Dataset All Labels'''''''''''''''
    # captions_path = "AdvisionDatasetAllData/Advision_ML.csv"
    # captions_path_train = "AdvisionDatasetAllData/train_dataset.csv"
    # captions_path_eval = "AdvisionDatasetAllData/test_dataset.csv"
    # all_label = "AdvisionDatasetAllData/captions.csv"
    #
    # train_path = "/Users/hka-private/PycharmProjects/Advision/all_transformed_images/"
    # eval_path = "/Users/hka-private/PycharmProjects/Advision/all_transformed_images/"
    #
    '''''''''''''''COCO Dataset All Labels'''''''''''''''

    captions_path = "COCO/coco_multilabel_classification.csv"

    train_path = "/Users/hka-private/PycharmProjects/Dataset/train2017/"
    eval_path = "/Users/hka-private/PycharmProjects/Dataset/val2017/"

    image_path = "/Users/hka-private/PycharmProjects/Dataset/train2017/"

    captions_path_train = "COCO/coco_multilabel_classification.csv"
    captions_path_eval = "COCO/eval_coco_multilabel_classification.csv"

    all_label = "COCO/coco_all_labels.csv"

    batch_size = 32  # 64
    num_workers = 0
    head_lr = 1e-3  # 1e-3
    image_encoder_lr = 1e-4  # 1e-4
    text_encoder_lr = 1e-5  # 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 40

    image_encoder_model = 'resnet50'  # 'inception_v3'  # 'vgg16.tv_in1k'  # 'vit_base_patch16_224' # 'resnet50'  #  'resnet101'
    image_embedding = 2048  # 768  384 4096 2048
    text_encoder_model = "distilbert-base-uncased" #"distilbert-base-uncased"  # "dbmdz/distilbert-base-german-europeana-cased" "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True  # for both image encoder and text encoder
    trainable = True  # for both image encoder and text encoder

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256

    temperature = 0.07  # try it with 0.07, 0.05, 0.01, 0.009, 0.005

    alpha = 0.5
    gamma = 3  # 0, 0.5, 1, 2, 5,
    beta = 2

    dropout = 0.5

    number_of_classes = 373

# /Users/hka-private/miniconda3/envs/conda-env/bin/python /Users/hka-private/PycharmProjects/mumic/mumic-multilabel-image-classification/MuMic-advision/Evaluation_v2.py
# 100%|██████████| 155/155 [00:55<00:00,  2.78it/s]
# MuMic_coco
# 0.6446803643446353
# Precision@5: 0.46
# Recall@5: 0.9927302
# F1@5: 0.60
# 80
# mAP@5: 0.6743639707565308
# avg_precision@5: 0.4573909531502503avg_recall@5: 0.9927302100161551avg_f1@5: 0.5980964881913984
# macro_mAP: 0.7290896773338318weighted_mAP: 0.7757543325424194
# ap_metric: tensor([0.9839, 0.6598, 0.8214, 0.8557, 0.9711, 0.8514, 0.9400, 0.6516, 0.8512,
#         0.7935, 0.7360, 0.6847, 0.6294, 0.5733, 0.7171, 0.9293, 0.7947, 0.8680,
#         0.8745, 0.8256, 0.9340, 0.9699, 0.9852, 0.9804, 0.4132, 0.7481, 0.4851,
#         0.7489, 0.5688, 0.8264, 0.9363, 0.6228, 0.7196, 0.9181, 0.7527, 0.8807,
#         0.9166, 0.9198, 0.9763, 0.5951, 0.6497, 0.6196, 0.6189, 0.5219, 0.4396,
#         0.6602, 0.7988, 0.5161, 0.5870, 0.7004, 0.8646, 0.6686, 0.5850, 0.8564,
#         0.7218, 0.6924, 0.7183, 0.7831, 0.4879, 0.8241, 0.7190, 0.9204, 0.8157,
#         0.7599, 0.7984, 0.6950, 0.8366, 0.5347, 0.5789, 0.7815, 0.2561, 0.8654,
#         0.7702, 0.6106, 0.7557, 0.6770, 0.4945, 0.7865, 0.1431, 0.3034],
#        device='mps:0')
#
