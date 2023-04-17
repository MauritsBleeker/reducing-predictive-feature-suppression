"""
Reference code:
"""
import torch
import torch.nn as nn
from utils.tern import PositionalEncodingImageBoxes


class ImageEncoder(nn.Module):

    def __init__(self, num_transformer_layers, feat_dim, embed_size, n_head=4, aggr='mean', pos_encoding=None, dropout=0.1, order_embeddings=False):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=n_head,
                                                       dim_feedforward=2048,
                                                       dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer,
                                                         num_layers=num_transformer_layers)
        if pos_encoding is not None:
            self.pos_encoding_image = PositionalEncodingImageBoxes(feat_dim, pos_encoding)
        self.fc = nn.Linear(feat_dim, embed_size)
        self.aggr = aggr
        self.order_embeddings = order_embeddings
        if aggr == 'gated':
            self.gate_fn = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, 1)
            )
            self.node_fn = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim)
            )
        self.pos_encoding = pos_encoding

    def forward(self, visual_feats, visual_feats_len=None, boxes=None):
        """
        Takes an variable len batch of visual features and preprocess them through a transformer. Output a tensor
        with the same shape as visual_feats passed in input.
        :param visual_feats:
        :param visual_feats_len:
        :return: a tensor with the same shape as visual_feats passed in input.
        """

        visual_feats = visual_feats.permute(1, 0, 2)

        if self.pos_encoding is not None:
            visual_feats = self.pos_encoding_image(visual_feats, boxes)

        if visual_feats_len is not None:
            bs = visual_feats.shape[1]
            # construct the attention mask
            max_len = max(visual_feats_len)
            mask = torch.zeros(bs, max_len).bool()
            for e, l in zip(mask, visual_feats_len):
                e[l:] = True
            mask = mask.to(visual_feats.device)
        else:
            mask = None

        visual_feats = self.transformer_encoder(visual_feats, src_key_padding_mask=mask)

        if self.aggr == 'mean':
            out = visual_feats.mean(dim=0)
        elif self.aggr == 'gated':
            out = visual_feats.permute(1, 0, 2)
            m = torch.sigmoid(self.gate_fn(out))   # B x S x 1
            v = self.node_fn(out)   # B x S x dim
            out = torch.bmm(m.permute(0, 2, 1), v)      # B x 1 x dim
            out = out.squeeze(1)    # B x dim
        else:
            out = visual_feats[0]

        out = self.fc(out)
        if self.order_embeddings:
            out = torch.abs(out)

        return out, visual_feats.permute(1, 0, 2)
