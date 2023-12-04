import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

import pickle as pkl

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
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

# class GCN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels,
#                              normalize=True)
#         self.conv2 = GCNConv(hidden_channels, out_channels,
#                              normalize=True)

#     def forward(self, x, edge_index, edge_weight=None):
#         print(f"GCN shape: {x.shape} {edge_index.shape} {edge_weight.shape}")

#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv1(x, edge_index, edge_weight).relu()
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         return x
    
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, dtype, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False).type(dtype)
        self.act = nn.PReLU(dtype=dtype) if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.dtype = dtype

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward_(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)
    
    # Shape of seq: (nodes, features)
    # Shape of adj: (nodes, nodes)
    def forward(self, seq, adj, sparse=False):
        # print(f"seq.dtype: {seq.dtype}")
        # print(f"fc.dtype: {self.fc.weight.dtype}")
        # print(f"dtype: {self.dtype}")
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.spmm(adj, seq_fts)
        else:
            out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class GNNClip(nn.Module):
    def __init__(self, clip_model, text, text_x, text_adj, image_x, image_adj, alpha=0.6, beta=0.7):
        super().__init__()
        # self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # self.text_encoder = TextEncoder(clip_model)
        # self.image_encoder = clip_model.visual

        ctx_dim = clip_model.ln_final.weight.shape[0]*2 # CLIP的输出表征维度到底是多少？

        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        print(f"clip_model.dtype: {self.dtype}")

        self.text_gnn = GCN(ctx_dim, ctx_dim, "prelu", dtype=self.dtype)
        self.visual_gnn = GCN(ctx_dim, ctx_dim, "prelu", dtype=self.dtype)

        print(f"ctx_dim= {ctx_dim}")

        self.text = text
        self.text_x = text_x
        self.text_adj = text_adj
        self.image_x = image_x
        self.image_adj = image_adj

        self.alpha = alpha
        self.beta = beta
    
    def forward_(self, image, text, x, edge_index, edge_weight):
        image_features = self.clip_model.encode_image(image.type(self.dtype))
        text_features = self.clip_model.encode_text(text)

        # GCN
        text_features_gnn = self.text_gnn(x, edge_index, edge_weight)
        text_features = 0.6*text_features + 0.4*text_features_gnn # residual connection

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    
    def forward(self, image):
        image_features = self.clip_model.encode_image(image.type(self.dtype))
        text_features = self.clip_model.encode_text(self.text)

        # GCN
        text_features_gnn = self.text_gnn(self.text_x, self.text_adj, sparse=False) # x表示的类的顺序与text的顺序一致
        visual_features_gnn = self.visual_gnn(self.image_x, self.image_adj, sparse=False) 
        text_features_gnn = self.beta*text_features_gnn + (1-self.beta)*visual_features_gnn # multi modal aggregation

        text_features = self.alpha*text_features + (1-self.alpha)*text_features_gnn # residual connection


        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image


@TRAINER_REGISTRY.register()
class GraphOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    # def __init__(self, cfg):
    #     super(GraphOp, self).__init__(cfg)
    #     # Load constructed graph from disk

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

        # 读取构建好的文本图
        # dassl 重写了从磁盘读取的函数，导致单GPU报错
        with open(f"/home/yliumh/github/CoOp/graph/text_graph.pkl", "rb") as f:
            data = pkl.load(f)
            text_x, text_edge_index, text_edge_weight, text_labels, texts = data["x"], data["edge_index"], data["edge_weight"], data["y"], data["texts"]
        
        self.text_x = text_x.to(self.device)
        self.text_edge_index = text_edge_index.to(self.device)
        self.text_edge_weight = text_edge_weight.to(self.device)
        self.text_labels = text_labels.to(self.device)
        self.texts = texts.to(self.device)
        
        # construct adj from edge_index and edge_weight
        n,d = self.text_x.shape
        text_adj = torch.zeros((n,n))
        for idx, (u,v) in enumerate(text_edge_index.T):
            text_weight = text_edge_weight[idx]
            text_adj[u,v] = text_adj[v,u] = text_weight
        self.text_adj = text_adj.type(self.text_x.dtype).to(self.device)

        # 读取构建好的视觉图
        with open(f"/home/yliumh/github/CoOp/graph/image_graph.pkl", "rb") as f:
            data = pkl.load(f)
            image_x, image_edge_index, image_edge_weight, image_labels = data["x"], data["edge_index"], data["edge_weight"], data["y"]
        
        self.image_x = image_x.to(self.device)
        self.image_edge_index = image_edge_index.to(self.device)
        self.image_edge_weight = image_edge_weight.to(self.device)
        self.image_labels = image_labels.to(self.device)
        
        # construct adj from edge_index and edge_weight
        n,d = self.image_x.shape
        image_adj = torch.zeros((n,n))
        for idx, (u,v) in enumerate(image_edge_index.T):
            image_weight = image_edge_weight[idx]
            image_adj[u,v] = image_adj[v,u] = image_weight
        self.image_adj = image_adj.type(self.image_x.dtype).to(self.device)


    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = GNNClip(clip_model, self.texts, self.text_x, self.text_adj, self.image_x, self.image_adj)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "text_gnn" not in name and "visual_gnn" not in name:
                param.requires_grad_(False) # 冻结CLIP的预训练权重

        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("text_gnn", self.model.text_gnn, self.optim, self.sched)
        self.register_model("visual_gnn", self.model.visual_gnn, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            # print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model)
            print(f"I will not use multiple GPUs")


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        # print(f"image shape: {image.shape} {label.shape}")
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # output = self.model(image, self.texts, self.x, self.edge_index, self.edge_weight)
            output = self.model(image) # should be self.model(image) to perform evaluate after training
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
