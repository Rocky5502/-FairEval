from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
C={'ink':'#172033','muted':'#65758b','line':'#cbd5e1','panel':'#f8fafc','white':'#ffffff','blue':'#1d4ed8','blue2':'#dbeafe','red':'#b42318','red2':'#fee4e2','green':'#15803d','green2':'#dcfce7','gold':'#a16207','gold2':'#fef3c7','violet':'#6d28d9','violet2':'#ede9fe','cyan':'#0e7490','cyan2':'#cffafe'}

def rr(ax,x,y,w,h,fc,ec=None,lw=1.0,r=0.018):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.006,rounding_size={r}',facecolor=fc,edgecolor=ec or C['line'],linewidth=lw); ax.add_patch(p); return p

def ar(ax,x1,y1,x2,y2,color=None,lw=1.2,ms=10):
    p=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=ms,linewidth=lw,color=color or C['muted'],shrinkA=2,shrinkB=2); ax.add_patch(p)

def tx(ax,x,y,s,fs=9,w='normal',c=None,ha='center',va='center'):
    ax.text(x,y,s,fontsize=fs,fontweight=w,color=c or C['ink'],ha=ha,va=va)

def user(ax,x,y,s=.018,c=None):
    c=c or C['ink']; ax.add_patch(Circle((x,y+s*.65),s*.36,fill=False,ec=c,lw=1.1)); ax.add_patch(FancyBboxPatch((x-s*.56,y-s*.65),s*1.12,s*.78,boxstyle='round,pad=0.001,rounding_size=0.005',fill=False,ec=c,lw=1.1))

def model(ax,x,y,s=.018,c=None):
    c=c or C['ink']; ax.add_patch(Rectangle((x-s*.72,y-s*.52),s*1.44,s*1.04,fill=False,ec=c,lw=1.0))
    for dx in (-.35,0,.35):
        for dy in (-.22,.22): ax.add_patch(Circle((x+dx*s,y+dy*s),s*.075,fc=c,ec='none'))

def scale(ax,x,y,s=.018,c=None):
    c=c or C['ink']; ax.plot([x,x],[y-s*.7,y+s*.7],c=c,lw=1.1); ax.plot([x-s*.72,x+s*.72],[y+s*.34,y+s*.34],c=c,lw=1.1)
    for dx in (-.55,.55): ax.plot([x+dx*s,x+(dx-.18)*s,x+(dx+.18)*s,x+dx*s],[y+s*.34,y-s*.18,y-s*.18,y+s*.34],c=c,lw=.85)

def shield(ax,x,y,s=.018,c=None):
    c=c or C['ink']; pts=[(x,y+s*.8),(x-s*.65,y+s*.35),(x-s*.48,y-s*.5),(x,y-s*.82),(x+s*.48,y-s*.5),(x+s*.65,y+s*.35)]; ax.add_patch(Polygon(pts,closed=True,fill=False,ec=c,lw=1.1))

def save(fig,name):
    fig.savefig(OUT/name,bbox_inches='tight',pad_inches=.04); plt.close(fig)

# Fig 1
fig,ax=plt.subplots(figsize=(12.6,5.9)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
tx(ax,.02,.955,'FairEval: consequence-aware evaluation with real, synthetic, hosted, and local evidence',11.5,'bold',ha='left'); tx(ax,.02,.918,'Recommendation change is sensitivity; harmfulness requires preference-conditioned consequence evidence.',8.2,c=C['muted'],ha='left')
rr(ax,.02,.14,.205,.72,C['panel']); tx(ax,.04,.825,'1  Evidence',10,'bold',ha='left'); user(ax,.052,.766,.024,C['blue']); tx(ax,.078,.778,'6 real-world datasets',8.3,'bold',ha='left'); tx(ax,.078,.747,'measured / observed / logged',7,c=C['muted'],ha='left')
for i,(h,sub) in enumerate([('3 personality','movies - music - short video'),('2 demographic','MovieLens-1M - Last.fm-1K'),('1 generalization','MIND logged news impressions')]):
    y=.675-i*.09; rr(ax,.047,y-.027,.151,.056,C['white'],r=.010); tx(ax,.06,y+.006,h,7.3,'bold',ha='left'); tx(ax,.06,y-.015,sub,6.5,c=C['muted'],ha='left')
rr(ax,.043,.245,.159,.126,C['cyan2'],C['cyan'],r=.012); tx(ax,.057,.34,'FairSynth-360',8,'bold',C['cyan'],ha='left'); tx(ax,.057,.304,'360 synthetic users',6.9,ha='left'); tx(ax,.057,.278,'A/B/C identity irrelevant',6.9,ha='left'); tx(ax,.057,.252,'synthetic OCEAN control',6.9,ha='left'); tx(ax,.122,.185,'Real / synthetic\nreported separately',6.2,'bold',C['muted'])
rr(ax,.255,.14,.205,.72,C['white']); tx(ax,.275,.825,'2  Matched conditions',10,'bold',ha='left')
for i,(cid,lab,fc) in enumerate([('C0','preference only',C['panel']),('C1','observed context',C['blue2']),('C2','identity swap',C['red2']),('C3','true personality',C['green2']),('C4','shuffled profile',C['gold2']),('C5','one-trait swap',C['violet2'])]):
    y=.735-i*.082; rr(ax,.28,y-.028,.154,.052,fc,r=.010); tx(ax,.293,y,cid,7.6,'bold',ha='left'); tx(ax,.326,y,lab,7.1,ha='left')
tx(ax,.357,.212,'LOCK  history - candidates - task - order',6.2,'bold',C['red']); tx(ax,.357,.174,'Change one registered context factor',6.4,'bold')
rr(ax,.49,.14,.245,.72,C['panel']); tx(ax,.51,.825,'3  Eight model configurations',10,'bold',ha='left'); tx(ax,.51,.784,'6 hosted/API',7.7,'bold',C['blue'],ha='left')
for i,label in enumerate(['GPT-5.6','Claude 5','Gemini 3.8','DeepSeek 4.1','Qwen 3.8','Llama 4']):
    col=i%3; row=i//3; x=.535+col*.073; y=.705-row*.105; ax.add_patch(Circle((x,y),.024,fc=C['white'],ec=C['line'],lw=.9)); model(ax,x,y,.016,C['blue']); tx(ax,x,y-.041,label,6.1)
rr(ax,.512,.315,.197,.165,C['green2'],C['green'],r=.012); tx(ax,.526,.45,'2 local open-weight',7.7,'bold',C['green'],ha='left')
for i,(lab,lic) in enumerate([('Qwen2.5-7B','Apache-2.0'),('Phi-3.5-mini','MIT')]):
    y=.402-i*.064; model(ax,.535,y,.015,C['green']); tx(ax,.558,y+.01,lab,6.9,'bold',ha='left'); tx(ax,.558,y-.013,lic,6.1,c=C['muted'],ha='left')
tx(ax,.51,.268,'RTX 5090 32 GB target - one model at a time',6.3,c=C['muted'],ha='left'); tx(ax,.51,.225,'Local-only diagnostics',7.1,'bold',C['violet'],ha='left'); tx(ax,.51,.195,'token log-p - NLL - margin',6.5,ha='left'); tx(ax,.51,.169,'auxiliary / uncalibrated',6.2,c=C['muted'],ha='left')
rr(ax,.765,.14,.215,.72,C['white']); tx(ax,.785,.825,'4  Auditable consequences',10,'bold',ha='left'); scale(ax,.797,.755,.024,C['red']); tx(ax,.827,.759,'Identity consequence',7.8,'bold',ha='left'); tx(ax,.793,.697,r'$CUG=U(C_1)-U(C_2)$',9.2,ha='left'); tx(ax,.793,.655,'IOD - GUD - CEG*',7.1,c=C['muted'],ha='left'); user(ax,.797,.585,.022,C['green']); tx(ax,.827,.59,'Personality value',7.8,'bold',ha='left'); tx(ax,.793,.53,r'$PVA=U(C_3)-U(C_4)$',9.2,ha='left'); shield(ax,.797,.455,.022,C['violet']); tx(ax,.827,.46,'Contextual PAIR',7.8,'bold',ha='left'); tx(ax,.793,.405,r'$Fit_a-\lambda Var_{cf}$',9.0,ha='left'); tx(ax,.793,.363,'validation-frozen Pareto point',6.4,c=C['muted'],ha='left'); rr(ax,.788,.245,.166,.075,C['cyan2'],r=.010); tx(ax,.801,.292,'FairSynth sanity',7,'bold',C['cyan'],ha='left'); tx(ax,.801,.265,'known identity irrelevance',6.4,ha='left'); tx(ax,.785,.19,'* CEG only with complete auditable metadata',5.9,c=C['muted'],ha='left')
for a,b in [(.225,.255),(.46,.49),(.735,.765)]: ar(ax,a,.50,b,.50,lw=1.3)
ax.plot([.02,.98],[.092,.092],c=C['line'],lw=.8); tx(ax,.02,.055,'Primary rule: change != unfairness. Real/synthetic and hosted/local evidence are reported in separate strata.',7.7,'bold',ha='left'); save(fig,'faireval_framework.pdf')

# Fig 2
fig,ax=plt.subplots(figsize=(11.8,4.9)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); tx(ax,.02,.94,'FairEval experimental geometry',11.5,'bold',ha='left'); tx(ax,.02,.898,'Two consequence contrasts, a controlled synthetic sanity layer, and factorized reliability - no post-hoc best-prompt selection.',8,c=C['muted'],ha='left')
rr(ax,.02,.19,.275,.62,C['panel']); tx(ax,.04,.765,'RQ1  demographic consequence',9.1,'bold',ha='left')
for x,lab,fc in [(.065,'C0\npreference',C['white']),(.155,'C1\nobserved',C['blue2']),(.245,'C2\nswapped',C['red2'])]: rr(ax,x-.034,.55,.068,.085,fc,r=.011); tx(ax,x,.592,lab,7,'bold')
ar(ax,.102,.592,.115,.592); ar(ax,.192,.592,.205,.592,C['red']); tx(ax,.158,.46,r'$CUG=U(C_1)-U(C_2)$',9.2); tx(ax,.158,.405,'RBO/Jaccard = sensitivity only',6.8,c=C['muted']); tx(ax,.158,.30,'MovieLens-1M + Last.fm-1K',7,'bold'); tx(ax,.158,.258,'gender primary - age robustness',6.4,c=C['muted'])
rr(ax,.32,.19,.275,.62,C['white']); tx(ax,.34,.765,'RQ2  measured personality value',9.1,'bold',ha='left')
for x,lab,fc in [(.365,'C0\nnone',C['panel']),(.455,'C3\ntrue',C['green2']),(.545,'C4\nshuffled',C['gold2'])]: rr(ax,x-.034,.55,.068,.085,fc,r=.011); tx(ax,x,.592,lab,7,'bold')
ar(ax,.402,.592,.415,.592,C['green']); ar(ax,.492,.592,.505,.592,C['gold']); tx(ax,.458,.46,r'$PVA=U(C_3)-U(C_4)$',9.2); tx(ax,.458,.405,'C5: one measured trait changed',6.8,c=C['muted']); tx(ax,.458,.30,'3 measured-personality datasets',7,'bold'); tx(ax,.458,.258,'whole-profile derangement preserves covariance',6.3,c=C['muted'])
rr(ax,.62,.19,.36,.62,C['panel']); tx(ax,.64,.765,'RQ3  reliability + controlled sanity',9.1,'bold',ha='left')
for i,(h,d) in enumerate([('Prompt','A / B / C'),('Cue','structured / 1st-person / profile'),('Order','3 deterministic permutations'),('K','5 / 10 / 20'),('Repeat','3 main; 10 stress-test')]):
    y=.665-i*.066; rr(ax,.65,y-.022,.094,.043,C['white'],r=.009); tx(ax,.661,y,h,6.8,'bold',ha='left'); tx(ax,.758,y,d,6.5,c=C['muted'],ha='left')
rr(ax,.65,.255,.295,.105,C['cyan2'],C['cyan'],r=.012); tx(ax,.667,.329,'FairSynth-360',7.2,'bold',C['cyan'],ha='left'); tx(ax,.667,.298,'A/B/C has zero relevance effect by construction',6.3,ha='left'); tx(ax,.667,.27,'synthetic personality tested separately',6.3,ha='left'); tx(ax,.02,.108,'Inference: user is the unit - paired permutation primary - paired bootstrap CI - Wilcoxon sensitivity - Holm within pre-registered families.',7.6,'bold',ha='left'); save(fig,'faireval_conditions.pdf')

# Fig 3
fig,ax=plt.subplots(figsize=(12.2,4.5)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); tx(ax,.02,.93,'From frozen cell to auditable claim',11.5,'bold',ha='left'); tx(ax,.02,.89,'Every empirical paper cell is generated from hashed artifacts; local internal diagnostics remain an auxiliary branch.',7.9,c=C['muted'],ha='left')
steps=[(.025,.13,'Frozen\ninstance',C['blue2'],C['blue']),(.19,.14,'Immutable\nrun cell',C['panel'],C['ink']),(.36,.14,'Validated\nranking',C['green2'],C['green']),(.53,.14,'User-level\nestimand',C['gold2'],C['gold']),(.70,.14,'Inference\nartifact',C['violet2'],C['violet']),(.87,.105,'Paper\nclaim',C['white'],C['ink'])]
for x,w,lab,fc,ec in steps: rr(ax,x,.49,w,.17,fc,ec=ec,r=.015); tx(ax,x+w/2,.575,lab,8,'bold')
for a,b in [(.155,.19),(.33,.36),(.50,.53),(.67,.70),(.84,.87)]: ar(ax,a,.575,b,.575,lw=1.25)
for x,s in [(0.09,'release hash\nsplit hash\ncandidate hash'),(.26,'prompt hash\nmodel/revision\nseed/order/K'),(.43,'exact K\nunique IDs\nrepair audit'),(.60,'CUG / PVA\nIOD / RBO\nuser unit'),(.77,'bootstrap CI\npermutation\nHolm / effect'),(.92,'no manual\nresult entry')]: tx(ax,x,.425,s,6.4,c=C['muted'])
rr(ax,.30,.16,.31,.15,C['cyan2'],C['cyan'],r=.012); tx(ax,.317,.278,'Local open-weight side channel',7.3,'bold',C['cyan'],ha='left'); tx(ax,.317,.242,'token log-p - NLL - perplexity - score margin',6.5,ha='left'); tx(ax,.317,.208,'descriptive only; not calibrated uncertainty',6.2,c=C['muted'],ha='left'); ar(ax,.43,.49,.43,.315,C['cyan'],1.0)
rr(ax,.65,.14,.31,.18,C['red2'],C['red'],r=.012); tx(ax,.667,.285,'RQ4 contextual PAIR',7.3,'bold',C['red'],ha='left'); tx(ax,.667,.242,r'$PAIR_a(i)=\alpha b(i)+(1-\alpha)r_a(i)-\lambda Var_{a\prime}r_{a\prime}(i)$',7.0,ha='left'); tx(ax,.667,.195,'alpha/lambda validation-only -> frozen test operating point',6.2,c=C['muted'],ha='left'); ar(ax,.60,.49,.73,.325,C['red'],1.0); save(fig,'faireval_evaluation_pipeline.pdf')

print(OUT)
