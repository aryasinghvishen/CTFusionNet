import os, json, time
import numpy as np, pandas as pd, cv2, torch, streamlit as st
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from ctfusion import *

PROJECT = os.environ.get('CT_PROJECT', '/content/drive/MyDrive/CT_Multitask_Research')
RUN     = os.environ.get('CT_RUN', 'ctfusion_full')
LESION_TYPES = {1: 'Bone', 2: 'Abdomen', 3: 'Mediastinum', 4: 'Liver', 5: 'Lung', 6: 'Kidney', 7: 'Soft tissue', 8: 'Pelvis'}
st.set_page_config(page_title='CTFusionNet demo', layout='wide')

@st.cache_resource
def load_model():
    cfg = json.load(open(f'{PROJECT}/results/{RUN}/config.json'))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = build_model(cfg)
    m.load_state_dict(torch.load(f'{PROJECT}/models/{RUN}/best.pt', map_location=dev))
    return m.to(dev).eval(), cfg, dev

@st.cache_data
def load_test():
    df = pd.read_csv(f'{PROJECT}/data/splits/manifest_paired.csv')
    return df[df['Train_Val_Test'] == 3]

@st.cache_data(show_spinner='Simulating low-dose scan (Radon → Poisson → FBP)...')
def simulate(key, _hu, spacing, I0, n_angles, mu_water):
    return simulate_low_dose(_hu, spacing, I0, seed=0, n_angles=n_angles, mu_water=mu_water)

model, cfg, dev = load_model()
test = load_test()
doses = cfg['DOSE_LEVELS']

st.title('CTFusionNet — low-dose CT denoising + lesion localization')
st.caption(f"Run `{RUN}` · model: {cfg.get('MODEL', 'ctfusion')} · transformer: {cfg['USE_TRANSFORMER']} · cross-task fusion: {cfg['USE_FUSION']} · "
           f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters · device: {dev}")

with st.sidebar:
    st.header('Input')
    src = st.radio('Image source', ['DeepLesion test set', 'Upload PNG'])
    gt, types, ld = [], [], None
    if src == 'DeepLesion test set':
        name = st.selectbox('Test slice', sorted(test['File_name'].unique()))
        rows = test[test['File_name'] == name]
        clean = np.clip(load_hu(rows.iloc[0]['Image_Path']), HU_MIN, HU_MAX)
        spacing = float(str(rows.iloc[0]['Spacing_mm_px_']).split(',')[0])
        gt = [[float(v) for v in b.split(',')] for b in rows['Bounding_boxes']]
        types = rows['Coarse_lesion_type'].tolist()
        options = [f'Pre-generated {d} dose (I0={v:.0e})' for d, v in doses.items()] + ['Custom simulation']
    else:
        up = st.file_uploader('DeepLesion-style 16-bit PNG (HU + 32768)', type=['png'])
        if up is None:
            st.info('Upload a slice to start.'); st.stop()
        name = up.name
        clean, is16 = hu_from_png_bytes(up.read())
        if not is16: st.warning('8-bit image: HU approximated from a soft-tissue window.')
        if clean.shape != (512, 512): clean = cv2.resize(clean, (512, 512), interpolation=cv2.INTER_AREA)
        clean = np.clip(clean, HU_MIN, HU_MAX)
        spacing = st.number_input('Pixel spacing (mm)', 0.3, 1.5, 0.7, 0.01)
        options = ['Custom simulation', 'Image is already low-dose (no simulation)']
    choice = st.selectbox('Low-dose input', options)
    if choice == 'Custom simulation':
        I0 = 10 ** st.slider('Dose level: log10(I0)  (lower = noisier)', 4.5, 7.0, 6.0, 0.25)
    st.header('Display')
    thr = st.slider('Detection score threshold', 0.05, 0.95, 0.30, 0.05)
    win = WINDOWS[st.selectbox('Window', list(WINDOWS))]
    show_att = st.checkbox('Show cross-task lesion attention', value=cfg['USE_FUSION'])

has_ref = True
if choice.startswith('Pre-generated'):
    d = choice.split()[1]
    ld = np.clip(load_hu(rows.iloc[0]['LowDose_Path' if d == 'low' else 'ModDose_Path']), HU_MIN, HU_MAX)
elif choice == 'Custom simulation':
    ld = np.clip(simulate(f'{name}-{I0}', clean, spacing, I0, cfg['N_ANGLES'], cfg['MU_WATER']), HU_MIN, HU_MAX)
else:
    ld, has_ref = clean.copy(), False

x = torch.from_numpy(norm(ld).astype(np.float32))[None, None].to(dev)
t0 = time.time()
with torch.no_grad():
    out = model(x)
if dev == 'cuda': torch.cuda.synchronize()
ms = (time.time() - t0) * 1000
den = to_hu(out['den'][0, 0].float().clamp(0, 1).cpu().numpy())
dets = decode(out, K=10, thr=thr)[0] if 'hm' in out else []

def overlay(hu_img, boxes_pred, boxes_gt):
    rgb = cv2.cvtColor(window(hu_img, win), cv2.COLOR_GRAY2RGB)
    for b in boxes_gt:
        x1, y1, x2, y2 = map(int, b); cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for s, b in boxes_pred:
        x1, y1, x2, y2 = map(int, b); cv2.rectangle(rgb, (x1, y1), (x2, y2), (255, 40, 40), 2)
        cv2.putText(rgb, f'{s:.2f}', (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 40, 40), 1)
    return rgb

c = st.columns(4)
c[0].image(window(ld, win), caption='Low-dose input')
c[1].image(window(den, win), caption=f'Denoised output ({ms:.0f} ms)')
c[2].image(window(clean, win), caption='Reference (original scan)' if has_ref else 'Input')
c[3].image(overlay(den, dets, gt), caption='Detections (red) vs ground truth (green)')

if has_ref:
    y, xin, xo = norm(clean), norm(ld), norm(den)
    body = clean > -500
    m = st.columns(4)
    p_in, p_out = psnr(y, xin, data_range=1), psnr(y, xo, data_range=1)
    s_in, s_out = ssim(y, xin, data_range=1), ssim(y, xo, data_range=1)
    m[0].metric('PSNR (dB)', f'{p_out:.2f}', f'{p_out - p_in:+.2f} vs input')
    m[1].metric('SSIM', f'{s_out:.4f}', f'{s_out - s_in:+.4f} vs input')
    m[2].metric('Noise std in body (HU)', f'{(den - clean)[body].std():.1f}', f'{(den - clean)[body].std() - (ld - clean)[body].std():+.1f}', delta_color='inverse')
    m[3].metric('Lesions detected', f'{sum(any(iou(b, [g])[0] >= 0.5 for _, b in dets) for g in gt)} / {len(gt)}' if gt else '—')

if show_att and out['att']:
    st.subheader('Cross-task fusion: lesion attention passed from the localization branch to the denoiser')
    a = st.columns(len(out['att']))
    for i, att in enumerate(out['att']):
        amap = cv2.resize(att[0, 0].float().cpu().numpy(), (512, 512))
        heat = cv2.applyColorMap((amap / max(amap.max(), 1e-6) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        blend = cv2.addWeighted(cv2.cvtColor(window(den, win), cv2.COLOR_GRAY2RGB), 0.6, heat, 0.4, 0)
        a[i].image(blend, caption=f'Fusion stage {i + 1} ({att.shape[-1]}×{att.shape[-1]})')

st.subheader('Detections')
if dets:
    st.dataframe(pd.DataFrame([{'score': round(s, 3), 'x1': round(b[0]), 'y1': round(b[1]), 'x2': round(b[2]), 'y2': round(b[3]),
                                'width (mm)': round((b[2] - b[0]) * spacing, 1), 'height (mm)': round((b[3] - b[1]) * spacing, 1),
                                'best IoU with GT': round(float(iou(b, gt).max()), 2) if gt else None} for s, b in dets]))
else:
    st.write('No detections above the threshold.')
if types:
    st.caption('Ground-truth lesion types: ' + ', '.join(LESION_TYPES.get(int(t), 'unknown') for t in types))
st.caption('Low-dose images are simulated from DeepLesion (Poisson noise in the projection domain + FBP). Research prototype, not for clinical use.')
