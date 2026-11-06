"""CTFusionNet: lightweight hybrid CNN-Transformer with bidirectional cross-task fusion
for simultaneous low-dose CT denoising and lesion localization."""
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from skimage.transform import radon, iradon

HU_MIN, HU_MAX = -1024, 3071      # HU range used for normalisation
STRIDE  = 4                       # detection output stride (512 -> 128)
MAX_OBJ = 8                       # max lesions per slice (DeepLesion has 1-3)
WINDOWS = {'Soft tissue': (-160, 240), 'Liver': (-45, 205), 'Lung': (-1350, 150), 'Bone': (-450, 1050)}

# ------------------------------------------------------------------ I/O
def hu_from_png_bytes(b):
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None: raise ValueError('Could not decode image')
    if img.ndim == 3: img = img[..., 0]
    if img.dtype == np.uint16:                       # DeepLesion format: HU = pixel - 32768
        return img.astype(np.float32) - 32768, True
    return img.astype(np.float32) / 255 * 400 - 160, False   # 8-bit: assume soft-tissue window

def load_hu(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    return img.astype(np.float32) - 32768

def save_hu_png(hu, path):
    cv2.imwrite(path, np.clip(np.round(hu) + 32768, 0, 65535).astype(np.uint16))

def norm(hu):  return (np.clip(hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
def to_hu(x):  return x * (HU_MAX - HU_MIN) + HU_MIN
def window(hu, win=(-160, 240)):
    return (np.clip((hu - win[0]) / (win[1] - win[0]), 0, 1) * 255).astype(np.uint8)

# ------------------------------------------------- low-dose simulation
def project(hu, spacing, n_angles=360, mu_water=0.02):
    theta = np.linspace(0., 180., n_angles, endpoint=False)
    mu = np.clip(mu_water * (1 + hu / 1000.0), 0, None)
    return radon(mu, theta=theta, circle=False) * spacing, theta

def fbp(sino, theta, spacing, n, mu_water=0.02):
    mu = iradon(sino / spacing, theta=theta, filter_name='ramp', circle=False, output_size=n)
    return (mu / mu_water - 1) * 1000.0

def add_poisson(sino, I0, seed):
    rng = np.random.default_rng(seed)
    cnt = np.maximum(rng.poisson(I0 * np.exp(-sino)), 1)
    return -np.log(cnt / I0)

def simulate_low_dose(hu, spacing, I0, seed=0, n_angles=360, mu_water=0.02):
    s, th = project(hu, spacing, n_angles, mu_water)
    return fbp(add_poisson(s, I0, seed), th, spacing, hu.shape[0], mu_water)

class TorchCT:
    """Same physics on the GPU: parallel-beam Radon (bilinear line sampling, 1-px steps),
    Ram-Lak ramp filter (skimage construction), linear-interpolated back projection, pi/(2A) scaling.
    Validated against the skimage version in the notebook before use."""
    def __init__(self, n=512, n_angles=360, device='cuda', chunk=30):
        self.n, self.A, self.dev, self.chunk = n, n_angles, device, chunk
        self.D = int(np.ceil(np.sqrt(2) * n))
        th = torch.deg2rad(torch.linspace(0, 180, n_angles + 1, device=device)[:-1])
        self.cos, self.sin = torch.cos(th), torch.sin(th)
        self.t = torch.arange(self.D, device=device, dtype=torch.float32) - (self.D - 1) / 2
        g = torch.arange(n, device=device, dtype=torch.float32) - (n - 1) / 2
        self.yy, self.xx = torch.meshgrid(g, g, indexing='ij')
        self.size = max(64, int(2 ** np.ceil(np.log2(2 * self.D))))
        k = np.concatenate((np.arange(1, self.size / 2 + 1, 2, dtype=int), np.arange(self.size / 2 - 1, 0, -2, dtype=int)))
        f = np.zeros(self.size); f[0] = 0.25; f[1::2] = -1 / (np.pi * k) ** 2
        self.filt = torch.tensor(2 * np.real(np.fft.fft(f)), dtype=torch.float32, device=device)

    def radon(self, img):                                   # img (N,N) -> (A,D) line integrals in pixel units
        x, half, out = img[None, None], (self.n - 1) / 2, []
        for a0 in range(0, self.A, self.chunk):
            cs, sn = self.cos[a0:a0 + self.chunk, None, None], self.sin[a0:a0 + self.chunk, None, None]
            t, s = self.t[None, :, None], self.t[None, None, :]
            px, py = t * cs - s * sn, t * sn + s * cs        # points on each ray
            grid = torch.stack([px / half, py / half], -1).reshape(1, -1, self.D, 2)
            v = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            out.append(v.view(-1, self.D, self.D).sum(-1))   # integrate along each ray
        return torch.cat(out, 0)

    def fbp(self, sino):                                    # sino (A,D) -> (N,N)
        q = torch.fft.ifft(torch.fft.fft(sino, n=self.size, dim=-1) * self.filt, dim=-1).real[:, :self.D]
        half_d, rec = (self.D - 1) / 2, torch.zeros(self.n * self.n, device=self.dev)
        for a0 in range(0, self.A, self.chunk):
            cs, sn = self.cos[a0:a0 + self.chunk, None, None], self.sin[a0:a0 + self.chunk, None, None]
            pos = (self.xx[None] * cs + self.yy[None] * sn + half_d).reshape(cs.shape[0], -1)
            i0 = pos.floor().long(); w = pos - i0
            valid = (i0 >= 0) & (i0 < self.D - 1); i0 = i0.clamp(0, self.D - 2)
            qa = q[a0:a0 + self.chunk]
            v = qa.gather(1, i0) * (1 - w) + qa.gather(1, i0 + 1) * w
            rec += (v * valid).sum(0)
        return rec.view(self.n, self.n) * np.pi / (2 * self.A)

    @torch.no_grad()
    def simulate(self, hu, spacing, I0s, seed, mu_water=0.02, return_clean=False):
        """hu: (N,N) numpy HU. Returns list of low-dose HU images (numpy), one per I0."""
        h = torch.as_tensor(hu, dtype=torch.float32, device=self.dev)
        p = self.radon((mu_water * (1 + h / 1000)).clamp(min=0)) * spacing
        to_hu_ = lambda s: ((self.fbp(s / spacing) / mu_water - 1) * 1000).cpu().numpy()
        outs = []
        for k, I0 in enumerate(I0s):
            g = torch.Generator(device=self.dev); g.manual_seed(int(seed) + k)
            cnt = torch.poisson(I0 * torch.exp(-p), generator=g).clamp(min=1)
            outs.append(to_hu_(-torch.log(cnt / I0)))
        return (outs, to_hu_(p)) if return_clean else outs

# ------------------------------------------------------ detection targets
def make_targets(boxes, size=512):
    H = size // STRIDE
    hm = np.zeros((H, H), np.float32)
    wh, off = np.zeros((2, MAX_OBJ), np.float32), np.zeros((2, MAX_OBJ), np.float32)
    ind, mask = np.zeros(MAX_OBJ, np.int64), np.zeros(MAX_OBJ, np.float32)
    yy, xx = np.mgrid[0:H, 0:H]
    for k, (x1, y1, x2, y2) in enumerate(list(boxes)[:MAX_OBJ]):
        cx, cy = (x1 + x2) / 2 / STRIDE, (y1 + y2) / 2 / STRIDE
        ix, iy = int(np.clip(cx, 0, H - 1)), int(np.clip(cy, 0, H - 1))
        w, h = (x2 - x1) / STRIDE, (y2 - y1) / STRIDE
        sigma = max(w, h) / 6 + 1
        hm = np.maximum(hm, np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sigma ** 2)))
        hm[iy, ix] = 1.0
        wh[:, k], off[:, k] = (w, h), (cx - ix, cy - iy)
        ind[k], mask[k] = iy * H + ix, 1
    return hm, wh, off, ind, mask

# ---------------------------------------------------------------- blocks
class LocalBlock(nn.Module):
    """Local texture: depthwise 7x7 conv + pointwise MLP (ConvNeXt-lite), residual."""
    def __init__(self, c, exp=2):
        super().__init__()
        self.dw, self.bn = nn.Conv2d(c, c, 7, padding=3, groups=c), nn.BatchNorm2d(c)
        self.pw1, self.act, self.pw2 = nn.Conv2d(c, c * exp, 1), nn.GELU(), nn.Conv2d(c * exp, c, 1)
    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.bn(self.dw(x)))))

class GlobalBlock(nn.Module):
    """Global anatomical context: efficient multi-head self-attention
    (spatial-reduction keys/values, conditional positional encoding) + MLP."""
    def __init__(self, c, heads=4, sr=1, mlp=2):
        super().__init__()
        self.h = heads
        self.pos = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.n1, self.n2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.q, self.kv, self.proj = nn.Linear(c, c), nn.Linear(c, 2 * c), nn.Linear(c, c)
        self.sr = nn.Sequential(nn.Conv2d(c, c, sr, stride=sr)) if sr > 1 else None
        self.srn = nn.LayerNorm(c) if sr > 1 else None
        self.mlp = nn.Sequential(nn.Linear(c, c * mlp), nn.GELU(), nn.Linear(c * mlp, c))
    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.pos(x)
        t = x.flatten(2).transpose(1, 2)                         # B, N, C
        h = self.n1(t)
        q = self.q(h).view(B, -1, self.h, C // self.h).transpose(1, 2)
        if self.sr is not None:
            kin = self.srn(self.sr(h.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2))
        else:
            kin = h
        k, v = self.kv(kin).view(B, -1, 2, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)              # B, heads, N, d
        t = t + self.proj(a.transpose(1, 2).reshape(B, -1, C))
        t = t + self.mlp(self.n2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)

class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.reduce, self.bn, self.block = nn.Conv2d(cin + cskip, cout, 1), nn.BatchNorm2d(cout), LocalBlock(cout)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(F.gelu(self.bn(self.reduce(torch.cat([x, skip], 1)))))

class CrossTaskFusion(nn.Module):
    """Bidirectional cross-task fusion (CTF).
    det -> den : lesion-aware spatial attention tells the denoiser WHERE detail must be preserved.
    den -> det : channel attention from denoising features passes noise-suppressed evidence to detection.
    Learnable gates start at 0, so training starts from the no-fusion model."""
    def __init__(self, c):
        super().__init__()
        self.lesion_att = nn.Sequential(nn.Conv2d(c, c // 4, 1), nn.GELU(), nn.Conv2d(c // 4, 1, 3, padding=1))
        self.det2den = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.BatchNorm2d(c), nn.GELU())
        self.ch_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), nn.GELU(), nn.Conv2d(c // 4, c, 1))
        self.den2det = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.BatchNorm2d(c), nn.GELU())
        self.g_den, self.g_det = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    def forward(self, f_den, f_det):
        a  = torch.sigmoid(self.lesion_att(f_det))                # B,1,H,W lesion attention
        cw = torch.sigmoid(self.ch_att(f_den))                    # B,C,1,1 channel weights
        den = f_den + self.g_den * self.det2den(torch.cat([f_den, f_det], 1)) * a
        det = f_det + self.g_det * self.den2det(torch.cat([f_det, f_den], 1)) * cw
        return den, det, a

def _head(cin, cout, bias=0.0):
    h = nn.Sequential(nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, cout, 1))
    nn.init.constant_(h[-1].bias, bias)
    return h

class CTFusionNet(nn.Module):
    """Shared hybrid encoder -> (denoising decoder, localization decoder) with CTF at 1/8 and 1/4 scale."""
    def __init__(self, base=32, use_transformer=True, use_fusion=True):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 6, base * 8]       # 512, 256, 128, 64, 32 px
        self.stem = nn.Sequential(nn.Conv2d(1, ch[0], 3, padding=1), nn.BatchNorm2d(ch[0]), nn.GELU(), LocalBlock(ch[0]))
        self.down = nn.ModuleList([nn.Sequential(nn.BatchNorm2d(ch[i]), nn.Conv2d(ch[i], ch[i + 1], 2, stride=2))
                                   for i in range(4)])
        g3 = GlobalBlock(ch[3], sr=2) if use_transformer else LocalBlock(ch[3])
        g4 = [GlobalBlock(ch[4]), GlobalBlock(ch[4])] if use_transformer else [LocalBlock(ch[4]), LocalBlock(ch[4])]
        self.stages = nn.ModuleList([
            nn.Sequential(LocalBlock(ch[1]), LocalBlock(ch[1])),
            nn.Sequential(LocalBlock(ch[2]), LocalBlock(ch[2])),
            nn.Sequential(LocalBlock(ch[3]), g3),
            nn.Sequential(*g4)])
        self.den_up = nn.ModuleList([Up(ch[4], ch[3], ch[3]), Up(ch[3], ch[2], ch[2]),
                                     Up(ch[2], ch[1], ch[1]), Up(ch[1], ch[0], ch[0])])
        self.det_up = nn.ModuleList([Up(ch[4], ch[3], ch[3]), Up(ch[3], ch[2], ch[2])])
        self.fuse = nn.ModuleList([CrossTaskFusion(ch[3]), CrossTaskFusion(ch[2])]) if use_fusion else None
        self.den_head = nn.Conv2d(ch[0], 1, 3, padding=1)
        self.hm, self.wh, self.off = _head(ch[2], 1, -2.19), _head(ch[2], 2), _head(ch[2], 2)

    def forward(self, x):
        f = [self.stem(x)]
        for d, s in zip(self.down, self.stages):
            f.append(s(d(f[-1])))
        den = det = f[4]
        atts = []
        for i in range(2):                                        # 1/8 and 1/4 scale: both decoders + fusion
            den, det = self.den_up[i](den, f[3 - i]), self.det_up[i](det, f[3 - i])
            if self.fuse is not None:
                den, det, a = self.fuse[i](den, det); atts.append(a)
        for i in (2, 3):                                          # denoising continues to full resolution
            den = self.den_up[i](den, f[3 - i])
        return {'den': x + self.den_head(den), 'hm': self.hm(det), 'wh': self.wh(det),
                'off': self.off(det), 'att': atts}

# ---------------------------------------------------------------- losses
def gather_feat(feat, ind):                     # feat B,C,H,W ; ind B,M  ->  B,C,M
    B, C = feat.shape[:2]
    return feat.reshape(B, C, -1).gather(2, ind.unsqueeze(1).expand(B, C, ind.shape[1]))

def focal_loss(logits, gt):
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = (gt >= 0.999).float()
    loss = -(pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * (1 - gt) ** 4 * p ** 2 * torch.log(1 - p)).sum()
    return loss / pos.sum().clamp(min=1)

def detection_loss(out, hm, wh, off, ind, mask):
    m = mask.unsqueeze(1)
    n = mask.sum().clamp(min=1)
    l_hm  = focal_loss(out['hm'], hm)
    l_wh  = (torch.abs(gather_feat(out['wh'].float(), ind) - wh) * m).sum() / n
    l_off = (torch.abs(gather_feat(out['off'].float(), ind) - off) * m).sum() / n
    return l_hm + 0.1 * l_wh + l_off, (l_hm.item(), l_wh.item(), l_off.item())

# ---------------------------------------------------------------- decoding
@torch.no_grad()
def decode(out, K=20, thr=0.01):
    """Returns, per image, a list of (score, [x1, y1, x2, y2]) sorted by score."""
    p = torch.sigmoid(out['hm'].float())
    p = p * (p == F.max_pool2d(p, 3, 1, 1))
    B, _, H, W = p.shape
    scores, inds = p.view(B, -1).topk(K)
    ys, xs = (inds // W).float(), (inds % W).float()
    wh, off = gather_feat(out['wh'].float(), inds), gather_feat(out['off'].float(), inds)
    cx, cy = (xs + off[:, 0]) * STRIDE, (ys + off[:, 1]) * STRIDE
    w, h = wh[:, 0].clamp(min=1) * STRIDE, wh[:, 1].clamp(min=1) * STRIDE
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).cpu().numpy()
    scores = scores.cpu().numpy()
    return [[(float(s), b.tolist()) for s, b in zip(scores[i], boxes[i]) if s >= thr] for i in range(B)]

def iou(box, boxes):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1]); b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-6)

def match_detections(dets, gts, iou_thr=0.5):
    """Greedy matching. Returns (records [(score, is_tp)], lesion_scores per GT (-1 = missed))."""
    recs, les = [], []
    for d, g in zip(dets, gts):
        g = np.asarray(g, np.float32).reshape(-1, 4); used = {}
        for s, b in d:
            ov = iou(b, g) if len(g) else np.zeros(0)
            for j in used: ov[j] = 0
            j = int(ov.argmax()) if len(ov) else -1
            if j >= 0 and ov[j] >= iou_thr:
                used[j] = s; recs.append((s, True))
            else:
                recs.append((s, False))
        les.append([used.get(j, -1.0) for j in range(len(g))])
    return recs, les

def froc(recs, n_images, n_lesions, fps=(0.5, 1, 2, 4, 8)):
    recs = sorted(recs, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([not r[1] for r in recs])
    res, thr = {}, {}
    for f in fps:
        ok = np.where(fp <= f * n_images)[0]
        res[f'sens@{f}FP'] = float(tp[ok[-1]] / n_lesions) if len(ok) else 0.0
        thr[f] = recs[ok[-1]][0] if len(ok) else 1.0
    return res, thr

def froc_curve(recs, n_images, n_lesions, max_fp=10, n=200):
    recs = sorted(recs, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([not r[1] for r in recs])
    xs = np.linspace(0.125, max_fp, n); ys = []
    for f in xs:
        ok = np.where(fp <= f * n_images)[0]
        ys.append(tp[ok[-1]] / n_lesions if len(ok) else 0.0)
    return xs, np.array(ys)

class REDCNN(nn.Module):
    """RED-CNN baseline (Chen et al., IEEE TMI 2017): 5 conv + 5 deconv with residual shortcuts."""
    def __init__(self, c=96):
        super().__init__()
        self.conv = nn.ModuleList([nn.Conv2d(1, c, 5)] + [nn.Conv2d(c, c, 5) for _ in range(4)])
        self.tconv = nn.ModuleList([nn.ConvTranspose2d(c, c, 5) for _ in range(4)] + [nn.ConvTranspose2d(c, 1, 5)])
    def forward(self, x):
        r1 = x
        o = F.relu(self.conv[0](x)); o = F.relu(self.conv[1](o)); r2 = o
        o = F.relu(self.conv[2](o)); o = F.relu(self.conv[3](o)); r3 = o
        o = F.relu(self.conv[4](o))
        o = self.tconv[0](o) + r3
        o = self.tconv[2](F.relu(self.tconv[1](F.relu(o)))) + r2
        o = self.tconv[4](F.relu(self.tconv[3](F.relu(o)))) + r1
        return {'den': F.relu(o), 'att': []}

def build_model(cfg):
    if cfg.get('MODEL', 'ctfusion') == 'redcnn':
        return REDCNN()
    return CTFusionNet(cfg['BASE_CH'], cfg['USE_TRANSFORMER'], cfg['USE_FUSION'])
