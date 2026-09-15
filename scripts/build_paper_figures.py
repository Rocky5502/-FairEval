from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, FancyArrowPatch, Rectangle
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

COL = {
    'ink':'#1f2937','muted':'#64748b','line':'#cbd5e1','panel':'#f8fafc',
    'blue':'#2563eb','blue2':'#dbeafe','red':'#b91c1c','red2':'#fee2e2',
    'green':'#15803d','green2':'#dcfce7','gold':'#a16207','gold2':'#fef3c7',
    'violet':'#6d28d9','violet2':'#ede9fe','white':'#ffffff'
}


def rr(ax, x, y, w, h, fc, ec=None, lw=1.0, r=0.02):
    p = FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.008,rounding_size={r}',
                       facecolor=fc, edgecolor=ec or COL['line'], linewidth=lw)
    ax.add_patch(p); return p


def arrow(ax, x1,y1,x2,y2, c=None, lw=1.2, ms=11, style='-|>'):
    a=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=ms,
                      linewidth=lw,color=c or COL['muted'],shrinkA=2,shrinkB=2)
    ax.add_patch(a); return a


def txt(ax,x,y,s,fs=9,weight='normal',c=None,ha='center',va='center'):
    return ax.text(x,y,s,fontsize=fs,fontweight=weight,color=c or COL['ink'],ha=ha,va=va)


def icon_user(ax, cx, cy, s=0.018, c=None):
    c=c or COL['ink']; ax.add_patch(Circle((cx,cy+s*0.7),s*0.42,fill=False,ec=c,lw=1.1))
    ax.add_patch(FancyBboxPatch((cx-s*0.6,cy-s*0.7),s*1.2,s*0.85,
        boxstyle='round,pad=0.001,rounding_size=0.006',fill=False,ec=c,lw=1.1))

def icon_model(ax,cx,cy,s=0.018,c=None):
    c=c or COL['ink']; ax.add_patch(Rectangle((cx-s*0.7,cy-s*0.55),s*1.4,s*1.1,fill=False,ec=c,lw=1.1))
    for dx in (-0.35,0,0.35):
        for dy in (-0.25,0.25): ax.add_patch(Circle((cx+dx*s,cy+dy*s),s*0.09,fc=c,ec='none'))

def icon_scale(ax,cx,cy,s=0.018,c=None):
    c=c or COL['ink']; ax.plot([cx,cx],[cy-s*0.7,cy+s*0.7],c=c,lw=1.1)
    ax.plot([cx-s*0.7,cx+s*0.7],[cy+s*0.35,cy+s*0.35],c=c,lw=1.1)
    for dx in (-0.55,0.55):
        ax.plot([cx+dx*s,cx+dx*s-s*0.2,cx+dx*s+s*0.2,cx+dx*s],[cy+s*0.35,cy-s*0.2,cy-s*0.2,cy+s*0.35],c=c,lw=0.9)

def icon_shield(ax,cx,cy,s=0.018,c=None):
    c=c or COL['ink']; pts=[(cx,cy+s*0.8),(cx-s*0.7,cy+s*0.35),(cx-s*0.5,cy-s*0.5),(cx,cy-s*0.85),(cx+s*0.5,cy-s*0.5),(cx+s*0.7,cy+s*0.35)]
    ax.add_patch(Polygon(pts,closed=True,fill=False,ec=c,lw=1.1))


def save(fig, name):
    fig.savefig(OUT/name, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

# Figure 1: overall framework
fig, ax = plt.subplots(figsize=(12.2,5.7))
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
txt(ax,0.02,0.955,'FairEval: preference-conditioned fairness evaluation',fs=12,weight='bold',ha='left')
txt(ax,0.02,0.915,'Sensitivity is observed; harm is established only through utility/exposure consequences.',fs=8.5,c=COL['muted'],ha='left')

rr(ax,0.02,0.14,0.18,0.71,COL['panel'])
txt(ax,0.04,0.82,'1  Evidence',fs=10,weight='bold',ha='left')
icon_user(ax,0.052,0.76,0.025,COL['blue'])
for i,(a,b) in enumerate([
    ('Personality 2018','Big Five + movies'),('Music Master','BFI-2 + songs'),('REASONER','Big Five + video'),
    ('MovieLens-1M','gender / age'),('LastFM-1K','gender / age'),('MIND','logged news clicks')]):
    y=0.72-i*0.095
    txt(ax,0.078,y,a,fs=8.2,weight='bold',ha='left')
    txt(ax,0.078,y-0.028,b,fs=7.2,c=COL['muted'],ha='left')

rr(ax,0.235,0.14,0.22,0.71,COL['white'])
txt(ax,0.255,0.82,'2  Counterfactual conditions',fs=10,weight='bold',ha='left')
conds=[('C0','preference only',COL['panel']),('C1','observed demographic',COL['blue2']),('C2','demographic swap',COL['red2']),
       ('C3','true measured personality',COL['green2']),('C4','shuffled profile',COL['gold2']),('C5','one-trait swap',COL['violet2'])]
for i,(cid,label,fc) in enumerate(conds):
    y=0.735-i*0.09
    rr(ax,0.258,y-0.033,0.172,0.055,fc,lw=0.8,r=0.012)
    txt(ax,0.273,y,cid,fs=8.2,weight='bold',ha='left')
    txt(ax,0.31,y,label,fs=7.8,ha='left')
txt(ax,0.255,0.19,'Held fixed:',fs=7.5,weight='bold',ha='left')
txt(ax,0.326,0.19,'history, candidates, task',fs=7.5,c=COL['muted'],ha='left')

rr(ax,0.49,0.14,0.21,0.71,COL['panel'])
txt(ax,0.51,0.82,'3  Controlled LLM audit',fs=10,weight='bold',ha='left')
for i,(lab,sub) in enumerate([('Template','A/B/C'),('Cue','structured / natural'),('Order','deterministic permutations')]):
    y=0.73-i*0.09
    rr(ax,0.515,y-0.03,0.16,0.052,COL['white'],r=0.012)
    txt(ax,0.53,y,lab,fs=7.9,weight='bold',ha='left'); txt(ax,0.595,y,sub,fs=7.2,c=COL['muted'],ha='left')
models=['GPT-5.6','Claude 5','Gemini 3.8','DeepSeek 4.1','Qwen 3.8','Llama 4']
for i,m in enumerate(models):
    col=i%2; row=i//2; x=0.535+col*0.085; y=0.42-row*0.085
    ax.add_patch(Circle((x,y),0.027,fc=COL['white'],ec=COL['line'],lw=1.0)); icon_model(ax,x,y,0.018,COL['blue'])
    txt(ax,x,y-0.045,m,fs=6.8)
txt(ax,0.595,0.17,'3 main repetitions + RQ3 stress tests',fs=7.4,c=COL['muted'])

rr(ax,0.735,0.14,0.245,0.71,COL['white'])
txt(ax,0.755,0.82,'4  Consequence-aware evaluation',fs=10,weight='bold',ha='left')
icon_scale(ax,0.77,0.75,0.026,COL['red'])
txt(ax,0.805,0.76,'Fairness consequences',fs=8.2,weight='bold',ha='left')
txt(ax,0.77,0.69,r'$\Delta U_{cf}=U(a)-U(a\prime)$',fs=10,ha='left')
txt(ax,0.77,0.64,r'$\Delta E_{cf}=TV(E(a),E(a\prime))$',fs=10,ha='left')
txt(ax,0.77,0.585,'GUD   |   IOD   |   metadata coverage',fs=7.6,c=COL['muted'],ha='left')
icon_user(ax,0.77,0.50,0.024,COL['green'])
txt(ax,0.805,0.505,'Personality value',fs=8.2,weight='bold',ha='left')
txt(ax,0.77,0.445,r'$PVA=U(C_3)-U(C_4)$',fs=10,ha='left')
icon_shield(ax,0.77,0.355,0.024,COL['violet'])
txt(ax,0.805,0.36,'Mitigation',fs=8.2,weight='bold',ha='left')
txt(ax,0.77,0.30,r'$PAIR(i)=Rel(i)-\lambda Var_{cf}(i)$',fs=9.4,ha='left')
txt(ax,0.77,0.24,'utility-fairness Pareto, not fairness-only wins',fs=7.4,c=COL['muted'],ha='left')

for x1,x2 in [(0.2,0.235),(0.455,0.49),(0.70,0.735)]: arrow(ax,x1,0.495,x2,0.495,COL['muted'],lw=1.35)
ax.plot([0.02,0.98],[0.09,0.09],c=COL['line'],lw=0.8)
txt(ax,0.02,0.055,r'Primary principle: recommendation change $\neq$ unfairness; impact must be preference-conditioned and statistically tested.',fs=8.2,weight='bold',ha='left')
save(fig,'faireval_framework.pdf')

# Figure 2: experimental condition geometry
fig, ax = plt.subplots(figsize=(11.4,4.8)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
txt(ax,0.02,0.94,'FairEval experimental geometry',fs=12,weight='bold',ha='left')
txt(ax,0.02,0.895,'Two causal contrasts + one robustness layer; no post-hoc best-prompt selection.',fs=8.5,c=COL['muted'],ha='left')

rr(ax,0.02,0.18,0.29,0.62,COL['panel'])
txt(ax,0.04,0.75,'RQ1 - Demographic consequence',fs=9.5,weight='bold',ha='left')
for x,y,lab,fc in [(0.075,0.58,'C0\npreference',COL['white']),(0.165,0.58,'C1\nobserved',COL['blue2']),(0.255,0.58,'C2\nswapped',COL['red2'])]:
    rr(ax,x-0.038,y-0.042,0.076,0.084,fc,r=0.014); txt(ax,x,y,lab,fs=7.6,weight='bold')
arrow(ax,0.11,0.58,0.125,0.58,COL['muted']); arrow(ax,0.205,0.58,0.215,0.58,COL['red'])
txt(ax,0.165,0.43,r'$CUG=U(C_1)-U(C_2)$',fs=10)
txt(ax,0.165,0.37,r'$CEG=TV(E(C_1),E(C_2))$',fs=10)
txt(ax,0.165,0.275,'Observed fields only\nage robustness pre-registered',fs=7.0,c=COL['muted'])

rr(ax,0.35,0.18,0.29,0.62,COL['white'])
txt(ax,0.37,0.75,'RQ2 - Grounded personality value',fs=9.5,weight='bold',ha='left')
for x,y,lab,fc in [(0.405,0.58,'C0\nnone',COL['panel']),(0.495,0.58,'C3\ntrue',COL['green2']),(0.585,0.58,'C4\nshuffled',COL['gold2'])]:
    rr(ax,x-0.038,y-0.042,0.076,0.084,fc,r=0.014); txt(ax,x,y,lab,fs=7.6,weight='bold')
arrow(ax,0.445,0.58,0.455,0.58,COL['green']); arrow(ax,0.535,0.58,0.545,0.58,COL['gold'])
txt(ax,0.495,0.43,r'$PVA=U(C_3)-U(C_4)$',fs=10)
txt(ax,0.495,0.35,'C5: one trait changed; other four fixed',fs=7.7,weight='bold')
txt(ax,0.495,0.275,'Measured Big Five only\nshuffled profile preserves distribution',fs=7.0,c=COL['muted'])

rr(ax,0.68,0.18,0.30,0.62,COL['panel'])
txt(ax,0.70,0.75,'RQ3/RQ4 - Reliability and mitigation',fs=9.5,weight='bold',ha='left')
for i,(a,b) in enumerate([('Prompt','A / B / C'),('Cue','structured / natural'),('Order','3 deterministic permutations'),('K','5 / 10 / 20'),('Repeat','3 main; 10 stress-test')]):
    y=0.64-i*0.075
    rr(ax,0.71,y-0.027,0.105,0.047,COL['white'],r=0.01); txt(ax,0.722,y,a,fs=7.3,weight='bold',ha='left')
    txt(ax,0.83,y,b,fs=7.2,c=COL['muted'],ha='left')
txt(ax,0.83,0.255,r'$PAIR=Rel-\lambda Var_{cf}$',fs=10)

txt(ax,0.02,0.10,'Inference: paired permutation (primary) + bootstrap CI + Holm within-RQ; malformed outputs remain outcomes.',fs=8.1,weight='bold',ha='left')
save(fig,'faireval_conditions.pdf')

# Figure 3: evaluation/mitigation flow
fig, ax = plt.subplots(figsize=(11.8,4.3)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
txt(ax,0.02,0.93,'From frozen cell to auditable claim',fs=12,weight='bold',ha='left')
steps=[
    (0.04,'Frozen\ncell',COL['blue2']), (0.20,'Prompt +\nSHA-256',COL['white']), (0.36,'LLM\nresponse',COL['panel']),
    (0.52,'Validate +\n1 format repair',COL['gold2']), (0.68,'Utility /\nfairness',COL['red2']), (0.84,'Paired\ninference',COL['green2'])]
for x,label,fc in steps:
    rr(ax,x,0.52,0.12,0.16,fc,r=0.018); txt(ax,x+0.06,0.60,label,fs=8.5,weight='bold')
for i in range(len(steps)-1): arrow(ax,steps[i][0]+0.12,0.60,steps[i+1][0],0.60,COL['muted'],lw=1.4)
rr(ax,0.04,0.27,0.92,0.13,COL['panel'],r=0.014)
txt(ax,0.06,0.355,'Provenance',fs=8.2,weight='bold',ha='left')
txt(ax,0.06,0.31,'plan cell ID  |  code commit  |  model/version  |  sampling applied  |  prompt/request/response hashes  |  validity',fs=7.5,c=COL['muted'],ha='left')
arrow(ax,0.74,0.52,0.74,0.45,COL['violet'],style='-|>')
rr(ax,0.64,0.08,0.20,0.12,COL['violet2'],r=0.014); txt(ax,0.74,0.14,'RQ4: prompt baseline + PAIR',fs=8.2,weight='bold')
arrow(ax,0.84,0.14,0.91,0.14,COL['violet']); txt(ax,0.96,0.14,'Pareto',fs=8,weight='bold')
save(fig,'faireval_evaluation_pipeline.pdf')

print('\n'.join(str(p) for p in sorted(OUT.glob('*.pdf'))))
