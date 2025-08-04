import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import trunc_normal_
from torch.nn import functional as F
from model.neck import get_neck
from model.utile.decoder import Transformer_decoder
from model.utile.share_decoder import Share_decoder
# from model.utile.new_encoder_l import Transformer_encoder_l
# from model.utile.new_encoder_u import Transformer_encoder_u
from model.utile.aggregation_encoder import Transformer_aggregation_encoder
import copy
from model.clip.model import AttentionPool2d

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER

        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes_T = 256
            self.in_planes = 2048
            self.in_planes_proj = [1024, 1024, 1024]
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner_w = PromptLearner_w(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.prompt_learner_u = PromptLearner_u(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.prompt_learner_l = PromptLearner_l(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)

        self.text_encoder = TextEncoder(clip_model)
#################################################################################################################################3

        self.image_encoder_u = copy.deepcopy(self.image_encoder)
        self.image_encoder_l = copy.deepcopy(self.image_encoder)

        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE, **cfg.ADJUST.KWARGS)

        self.upsampling = nn.Conv2d(self.in_planes_T, self.in_planes, kernel_size=1, bias=False)
        self.upsampling.apply(weights_init_kaiming)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.downsapling = nn.ConvTranspose2d(self.in_planes_T * 2, self.in_planes_T, kernel_size=1, bias=False)
        self.downsapling.apply(weights_init_kaiming)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_projs = nn.ModuleList([
            nn.BatchNorm1d(self.in_planes_proj[i]) for i in range(len(self.in_planes_proj))
        ])

        for layer in self.bottleneck_projs:
            layer.bias.requires_grad_(False)
            layer.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifier_proj = nn.ModuleList([
            nn.Linear(self.in_planes_proj[i], self.num_classes, bias=False) for i in range(len(self.in_planes_proj))
        ])

        for layer in self.classifier_proj:
            layer.apply(weights_init_classifier)

        channel = 256

        # self.encoder_l = Transformer_encoder_l(channel, 8, 1)
        # self.encoder_u = Transformer_encoder_u(channel, 8, 1)
        self.encoder_l = Transformer_aggregation_encoder(channel, 8, 1)
        self.encoder_u = Transformer_aggregation_encoder(channel, 8, 1)
        self.decoder = Transformer_decoder(channel, 8, 1)

    def forward(self, x = None, x_u=None, x_l=None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text == True:
            prompts_w = self.prompt_learner_w(label)
            text_features_w = self.text_encoder(prompts_w, self.prompt_learner_w.tokenized_prompts)
            prompts_u = self.prompt_learner_u(label)
            text_features_u = self.text_encoder(prompts_u, self.prompt_learner_u.tokenized_prompts)
            prompts_l = self.prompt_learner_l(label)
            text_features_l = self.text_encoder(prompts_l, self.prompt_learner_l.tokenized_prompts)
            return text_features_w, text_features_u, text_features_l

        if get_image == True:
            _, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]

        if self.model_name == 'RN50':
            image_features_w, image_features_proj_w = self.image_encoder(x)
            image_features_u, image_features_proj_u = self.image_encoder_u(x_u)
            image_features_l, image_features_proj_l = self.image_encoder_l(x_l)

            img_feat_w = nn.functional.avg_pool2d(image_features_w, image_features_w.shape[2:4]).view(image_features_w.shape[0], -1)
            img_feat_u = nn.functional.avg_pool2d(image_features_u, image_features_u.shape[2:4]).view(image_features_u.shape[0], -1)
            img_feat_l = nn.functional.avg_pool2d(image_features_l, image_features_l.shape[2:4]).view(image_features_l.shape[0], -1)

            img_feat = [img_feat_w, img_feat_u, img_feat_l]

            ad_image_features = self.neck([image_features_w, image_features_u, image_features_l])

            b, c, w, h = ad_image_features[0].size() #128,256,16,8

            encoder_out_u = self.encoder_u(ad_image_features[0].view(b, c, -1).permute(2, 0, 1),\
                                       ad_image_features[1].view(b, c, -1).permute(2, 0, 1),)


            encoder_out_l = self.encoder_l(ad_image_features[0].view(b, c, -1).permute(2, 0, 1),\
                                       ad_image_features[2].view(b, c, -1).permute(2, 0, 1),)

            memory = torch.cat([encoder_out_u.permute(1, 2, 0).view(b, c, w, h), encoder_out_l.permute(1, 2, 0).view(b, c, w, h)], dim = 1)
            memory = self.downsapling(memory)
            res = self.decoder((ad_image_features[0]).view(b, c, -1).permute(2, 0, 1), memory.view(b, c, -1).permute(2, 0, 1))

            res = res.permute(1, 2, 0).view(b, c, w, h)
            res = self.upsampling(res)
            res = self.bn(res)

            res = nn.functional.avg_pool2d(res, res.shape[2:4]).view(res.shape[0], -1)

            image_feature_proj_w = image_features_proj_w[0]
            image_feature_proj_u = image_features_proj_u[0]
            image_feature_proj_l = image_features_proj_l[0]

            img_feature_proj = [image_feature_proj_w, image_feature_proj_u, image_feature_proj_l]

        feat = self.bottleneck(res)
        feat_proj = []
        for i, (bottleneck, proj) in enumerate(zip(self.bottleneck_projs, img_feature_proj)):
            feat_proj.append(bottleneck(proj))

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj_w = self.classifier_proj[0](feat_proj[0])
            cls_score_proj_u = self.classifier_proj[1](feat_proj[1])
            cls_score_proj_l = self.classifier_proj[2](feat_proj[2])
            return [cls_score, cls_score_proj_w, cls_score_proj_u, cls_score_proj_l], res, img_feat, image_feature_proj_w, image_feature_proj_u, image_feature_proj_l
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj[0]], dim=1)
            else:
                return torch.cat([res, image_feature_proj_w], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner_w(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of the whole body of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 8
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts


class PromptLearner_u(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of the upper body of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 8

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class PromptLearner_l(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of the lower body of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 8

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts



