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
