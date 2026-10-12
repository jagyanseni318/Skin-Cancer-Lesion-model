"""Shared definitions for the revised TrustXAI-Derm pipeline (reproduces the original Kaggle notebook's
model / loss / preprocessing exactly; only the training schedule, checkpoint-selection split and seeds change)."""
import os, json, random
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = Path(__file__).resolve().parent / 'out'
CACHE = Path(__file__).resolve().parent / 'cache'
OUT.mkdir(exist_ok=True, parents=True); CACHE.mkdir(exist_ok=True, parents=True)
CLASSES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
C2I = {c: i for i, c in enumerate(CLASSES)}
IMG = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
SEEDS = [42, 43, 44, 45, 46]
SPLIT_SEED = 42            # split seed fixed across model seeds (lesion-level split identical to original)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_MAP = {'NEV': 'NV', 'MEL': 'MEL', 'BCC': 'BCC', 'ACK': 'AKIEC', 'SEK': 'BKL'}


def dull_razor(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, cand = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(cand, cv2.MORPH_OPEN, disk)
    hair = cv2.subtract(cand, opened)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hair, connectivity=8)
    cleaned = np.zeros_like(hair)
    for l in range(1, n):
        if stats[l, cv2.CC_STAT_AREA] >= 8:
            cleaned[labels == l] = 255
    hair = cleaned
    if hair.sum() == 0:
        return image_bgr
    hair = cv2.dilate(hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return cv2.inpaint(image_bgr, hair, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def load_ham():
    df = pd.read_csv(DATA / 'HAM10000' / 'HAM10000_metadata.csv')
    M = {'mel': 'MEL', 'nv': 'NV', 'bcc': 'BCC', 'akiec': 'AKIEC', 'bkl': 'BKL', 'df': 'DF', 'vasc': 'VASC'}
    df['label'] = df.dx.map(M); df['label_idx'] = df.label.map(C2I)
    paths = {}
    for d in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        for f in (DATA / 'HAM10000' / d).glob('*.jpg'):
            paths[f.stem] = str(f)
    df['image_path'] = df.image_id.map(paths)
    u = df.drop_duplicates('lesion_id')[['lesion_id', 'label']]
    tr, te = train_test_split(u, test_size=0.20, random_state=SPLIT_SEED, stratify=u['label'])
    va, ts = train_test_split(te, test_size=0.50, random_state=SPLIT_SEED, stratify=te['label'])
    df['split'] = 'train'
    df.loc[df.lesion_id.isin(set(va.lesion_id)), 'split'] = 'val'
    df.loc[df.lesion_id.isin(set(ts.lesion_id)), 'split'] = 'test'
    df['cache_idx'] = np.arange(len(df))
    # val halves (as in original CELL 9): val_sel -> checkpoint selection, val_cal -> temperature scaling
    v = df[df.split == 'val']
    vsel, vcal = train_test_split(v, test_size=0.5, random_state=SPLIT_SEED, stratify=v['label'])
    df['val_role'] = ''
    df.loc[vsel.index, 'val_role'] = 'sel'; df.loc[vcal.index, 'val_role'] = 'cal'
    return df


def load_pad():
    m = pd.read_csv(DATA / 'PAD-UFES-20' / 'metadata.csv')
    m['label'] = m.diagnostic.map(PAD_MAP)
    m['path'] = m.img_id.map(lambda x: str(DATA / 'PAD-UFES-20' / 'images' / x))
    m['cache_idx'] = np.arange(len(m))
    m['label_idx'] = m.label.map(C2I)
    return m   # includes SCC rows (label NaN) so cache index covers all 2,298


class DenseNet121_TrustXAI(nn.Module):
    def __init__(self, n_classes=7, dropout_rate=0.3, use_mc_dropout=False):
        super().__init__()
        self.use_mc_dropout = use_mc_dropout
        self.backbone = models.densenet121(weights='DEFAULT')
        inf = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(inf, 512); self.bn1 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=0.2); self.classifier = nn.Linear(512, n_classes)

    def head(self, f):
        if self.use_mc_dropout:
            f = self.dropout1(f)
        x = F.relu(self.bn1(self.fc1(f))); x = self.dropout2(x)
        return self.classifier(x)

    def forward(self, x):
        return self.head(self.backbone(x))

    def cam_layers(self):
        return [self.backbone.features.denseblock4.denselayer16.conv2]

    def param_groups(self, lr=3e-4):
        f = self.backbone.features
        stem = (list(f.conv0.parameters()) + list(f.norm0.parameters()) + list(f.transition1.parameters()) +
                list(f.transition2.parameters()) + list(f.transition3.parameters()) + list(f.norm5.parameters()))
        return [{'params': stem, 'lr': 1e-5}, {'params': f.denseblock1.parameters(), 'lr': 1e-5},
                {'params': f.denseblock2.parameters(), 'lr': 1e-5}, {'params': f.denseblock3.parameters(), 'lr': 5e-5},
                {'params': f.denseblock4.parameters(), 'lr': 5e-5},
                {'params': list(self.fc1.parameters()) + list(self.bn1.parameters()) + list(self.classifier.parameters()), 'lr': lr}]


class ResNet50_TrustXAI(nn.Module):
    """Second architecture used only for the architecture-robustness check (Section 4.12): same head design,
    training recipe and Grad-CAM++ target-layer convention (last conv block before global pooling) as
    DenseNet121_TrustXAI, so results are comparable and not an artefact of one backbone's inductive bias."""
    def __init__(self, n_classes=7, dropout_rate=0.3, use_mc_dropout=False):
        super().__init__()
        self.use_mc_dropout = use_mc_dropout
        self.backbone = models.resnet50(weights='DEFAULT')
        inf = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(inf, 512); self.bn1 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=0.2); self.classifier = nn.Linear(512, n_classes)

    def head(self, f):
        if self.use_mc_dropout:
            f = self.dropout1(f)
        x = F.relu(self.bn1(self.fc1(f))); x = self.dropout2(x)
        return self.classifier(x)

    def forward(self, x):
        return self.head(self.backbone(x))

    def cam_layers(self):
        return [self.backbone.layer4[-1].conv3]

    def param_groups(self, lr=3e-4):
        b = self.backbone
        stem = list(b.conv1.parameters()) + list(b.bn1.parameters())
        return [{'params': stem, 'lr': 1e-5}, {'params': b.layer1.parameters(), 'lr': 1e-5},
                {'params': b.layer2.parameters(), 'lr': 1e-5}, {'params': b.layer3.parameters(), 'lr': 5e-5},
                {'params': b.layer4.parameters(), 'lr': 5e-5},
                {'params': list(self.fc1.parameters()) + list(self.bn1.parameters()) + list(self.classifier.parameters()), 'lr': lr}]


ARCHES = {'densenet121': DenseNet121_TrustXAI, 'resnet50': ResNet50_TrustXAI}


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__(); self.register_buffer('alpha', alpha); self.gamma = gamma

    def forward(self, x, t):
        lp = F.log_softmax(x, 1); pt = torch.exp(lp.gather(1, t[:, None]).squeeze(1)).clamp(min=1e-8)
        return (-self.alpha[t] * (1 - pt).pow(self.gamma) * torch.log(pt)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__(); self.focal = FocalLoss(alpha, gamma); self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x, t):
        return self.focal(x, t) + 0.1 * self.ce(x, t)


def make_alpha(train_df):
    n = len(train_df); cnt = train_df.label_idx.value_counts().sort_index()
    cw = torch.tensor([n / (7 * cnt[i]) for i in range(7)], dtype=torch.float32)
    man = {'DF': 3.0, 'VASC': 2.5, 'AKIEC': 2.0, 'BCC': 1.5, 'MEL': 1.2, 'BKL': 1.2, 'NV': 1.0}
    a = torch.tensor([man[c] * cw[i].item() for i, c in enumerate(CLASSES)], dtype=torch.float32)
    return a / a.sum() * 7


def norm_batch(x_uint8):
    """uint8 NHWC array -> normalized NCHW float tensor"""
    x = torch.as_tensor(np.asarray(x_uint8)).float() / 255.0
    x = (x - torch.tensor(MEAN)) / torch.tensor(STD)
    return x.permute(0, 3, 1, 2).contiguous()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
