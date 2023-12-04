
import pickle as pkl
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from dassl.utils import setup_logger, set_random_seed, collect_env_info

from PIL import Image
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, ColorJitter,
    RandomApply, GaussianBlur, RandomGrayscale, RandomResizedCrop,
    RandomHorizontalFlip
)
from torchvision.transforms.functional import InterpolationMode
INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

def load_clip_to_cpu():
    # backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone_name = "RN50"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class CustomCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        raise NotImplementedError()
    
    def tokenize(self, text):
        return clip.tokenize(f"{text}")
    
    def encode_text(self, text):
        text_features = self.clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True) # To compute cosine sim: first norm, then dot

        return text_features
    
    def encode_image(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features

def build_textual_graph(datapath, clipmodel):

    device = "cuda:6"
    model = CustomCLIP(clipmodel).to(device)
    dtype = model.dtype
    print(f"clip_model.dtype: {model.dtype}")

    with open(datapath, "rb") as f:
        data = pkl.load(f)
        train = data["train"]
    
    visited_labels = set()
    model_inputs = []
    labels = []

    dtype = clipmodel.dtype

    with torch.no_grad():
        for train_sample in train:
            label = train_sample.label
            if not label in visited_labels:
                visited_labels.add(label)

                classname = train_sample.classname
                text = f"a photo of a {classname}" # Caltech101 Dataset
                tokenized_text = model.tokenize(text)
                model_inputs.append(tokenized_text)

                labels.append(label)
            
        model_inputs = torch.cat(model_inputs).to(device)
        text_features = model.encode_text(model_inputs)

        # logit_scale = model.logit_scale.exp()
        # adj_full = logit_scale * graph_nodes @ graph_nodes.t()
        adj_full = text_features @ text_features.t()

        # Back to cpu before return 
        adj_full = adj_full.to("cpu")
        text_features = text_features.to("cpu")
        labels = torch.LongTensor(labels)
    
    return text_features, adj_full, labels, model_inputs

# /home/yliumh/github/Dassl.pytorch/dassl/data/transforms/transforms.py
def _build_transform_test(interpolation, input_size, pixel_mean, pixel_std, choices, target_size, normalize):
    print("Building transform_test")
    tfm_test = []

    interp_mode = INTERPOLATION_MODES[interpolation]
    input_size = input_size

    print(f"+ resize the smaller edge to {max(input_size)}")
    tfm_test += [Resize(max(input_size), interpolation=interp_mode)]

    print(f"+ {target_size} center crop")
    tfm_test += [CenterCrop(input_size)]

    print("+ to torch tensor of range [0, 1]")
    tfm_test += [ToTensor()]

    if "normalize" in choices:
        print(
            f"+ normalization (mean={pixel_mean}, std={pixel_std})"
        )
        tfm_test += [normalize]

    if "instance_norm" in choices:
        raise NotImplementedError
        # print("+ instance normalization")
        # tfm_test += [InstanceNormalization()]

    tfm_test = Compose(tfm_test)

    return tfm_test

def build_vision_graph(datapath, clipmodel, text_labels=None):
    device = "cuda:6"
    model = CustomCLIP(clipmodel).to(device)

    with open(datapath, "rb") as f:
        data = pkl.load(f)
        train = data["train"]
        print(f"length of train: {len(train)}")
    
    visited_labels = dict()
    model_inputs = []
    N = 5 # Maximum num of images in each class, to calc the average embedding
    image_labels = []
    image_features_box = dict()

    # Align with text_labels
    if text_labels is not None:
        for label in text_labels:
            image_features_box[int(label)] = 0
        print(f"Total classes: {len(image_features_box)}")

    # Build image transformations
    input_size = (224, 224)
    interpolation = "bicubic"
    pixel_mean = [0.48145466, 0.4578275, 0.40821073]
    pixel_std = [0.26862954, 0.26130258, 0.27577711]
    choices = ["random_resized_crop", "random_flip", "normalize"]

    target_size = f"{input_size[0]}x{input_size[1]}"

    normalize = Normalize(mean=pixel_mean, std=pixel_std)
    transforms = _build_transform_test(interpolation, input_size, pixel_mean, pixel_std, choices, target_size, normalize)

    with torch.no_grad():
        for train_sample in train:
            impath = train_sample.impath
            label = train_sample.label

            if not label in visited_labels.keys():
                visited_labels[label] = 0
            if visited_labels[label] >= N:
                continue
            visited_labels[label] += 1

            img0 = Image.open(impath).convert("RGB")
            img = transforms(img0)
            
            model_inputs.append(img.unsqueeze(0))
            image_labels.append(label)

        model_inputs = torch.cat(model_inputs).to(device)
        print(model_inputs.shape)
        image_features = model.encode_image(model_inputs)
        print(f"image_features.shape: {image_features.shape}")

        for i in range(image_features.shape[0]):
            image_feature = image_features[i]
            label = image_labels[i]
            n_images = visited_labels[label]
            
            try:
                image_features_box[int(label)] += image_feature.unsqueeze(0) / n_images
            except:
                print(f"Error: {label} not in text_labels !!")
        
        print(f"image_features_box[0].shape: {image_features_box[0].shape}")
        image_features_avg = torch.cat(list(image_features_box.values()))
        print(f"image_features_avg.shape: {image_features_avg.shape}")
        adj_full = image_features_avg @ image_features_avg.t()

        # Back to cpu before return 
        adj_full = adj_full.to("cpu")
        image_features = image_features_avg.to("cpu")
        labels = torch.LongTensor(list(image_features_box.keys()))

    return image_features, adj_full, labels


class myDataset(Dataset):
    def __init__(self, x, edge_index, edge_weight):
        super().__init__()

        self.x = x
        self.edge_index = edge_index
        self.edge_weight = edge_weight

if __name__ == "__main__":

    seed = 1
    print("Setting fixed seed: {}".format(seed))
    set_random_seed(seed)

    clipmodel = load_clip_to_cpu()
    datapath = f"/data/yliumh/caltech-101/split_fewshot/shot_16-seed_1.pkl"


    # Build text graph
    print("Building text graph")
    text_features, text_adj, text_labels, tokenized_texts = build_textual_graph(datapath, clipmodel)

    print(text_features.shape, text_adj.shape)
    print(text_adj[:20, :20])

    # build torch_geometric graph: data=[x, edge_index, edge_weight]
    n, h = text_features.shape
    edge_weight = text_adj.reshape(-1)
    edge_index = []
    for i in range(n):
        for j in range(n):
            edge_index.append([i,j])
    edge_index = torch.tensor(edge_index)

    print(f"{text_features.dtype}")
    print(f"{edge_index.dtype}")
    print(f"{edge_weight.dtype}")
    print(f"{text_labels.dtype}")
    print(f"{tokenized_texts.dtype}")

    text_graph = {
        "x": text_features,
        "edge_index": edge_index.transpose(1,0),
        "edge_weight": edge_weight,
        "y": text_labels,
        "texts": tokenized_texts,
    }

    with open("text_graph.pkl", "wb") as f:
        pkl.dump(text_graph, f)
    

    # Build image graph
    print("Building vision graph")
    image_features, image_adj, image_labels = build_vision_graph(datapath, clipmodel, text_labels)
    assert torch.sum(image_labels - text_labels) == 0 # Two modals have aligned

    print(image_features.shape, image_adj.shape)
    print(image_adj[:20, :20])

    # build torch_geometric graph: data=[x, edge_index, edge_weight]
    n, h = image_features.shape
    edge_weight = image_adj.reshape(-1)
    edge_index = []
    for i in range(n):
        for j in range(n):
            edge_index.append([i,j])
    edge_index = torch.tensor(edge_index)

    print(f"{image_features.dtype}")
    print(f"{edge_index.dtype}")
    print(f"{edge_weight.dtype}")
    print(f"{image_labels.dtype}")

    image_graph = {
        "x": image_features,
        "edge_index": edge_index.transpose(1,0),
        "edge_weight": edge_weight,
        "y": image_labels,
    }

    with open("image_graph.pkl", "wb") as f:
        pkl.dump(image_graph, f)

