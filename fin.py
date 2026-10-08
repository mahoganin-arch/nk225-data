import pickle, numpy as np, pandas as pd, sys
tf=sys.argv[1]; TOP=int(sys.argv[2])
d,X,E,cols=pickle.load(open(f'P_{tf}.pkl','rb')); Xv=X.values; ci={c:i for i,c in enumerate(cols)}
yr=d.t.dt.year.values
def cmask(cond,bucket):
    if cond=='なし': return np.ones(len(d),bool)
    m=np.ones(len(d),bool)
    for f,b in zip(cond.split(' & '),bucket.split(' & ')): m&=(d['F_'+f].astype(str).values==b)
    return m
def smask(sig):
    for lab in ('g2','g3','g5','pat'):
        if sig in set(d[lab].unique()): return (d[lab]==sig).values
def nonover(mask,col):
    c=ci[col]; r=Xv[:,c]; e=E[:,c]; idx=np.where(mask)[0]; free=0; out=[]
    for i in idx:
        if i<free or np.isnan(r[i]): continue
        out.append((yr[i],r[i])); free=i+int(e[i])+1
    return pd.DataFrame(out,columns=['yr','ret'])
def st(z):
    if len(z)<5: return dict(n=len(z),wr=np.nan,pf=np.nan,mean=np.nan)
    p=z.ret[z.ret>0].sum(); n=-z.ret[z.ret<0].sum(); return dict(n=len(z),wr=(z.ret>0).mean(),pf=p/max(n,1e-9),mean=z.ret.mean())
K=pd.read_pickle(f'KL_{tf}.pkl'); K=K[K.ok].copy()
K['ex_te']=(K.mean_te-K.base_te)/(K.mean_te/K.t_te); K['ex_tr']=(K.mean_tr-K.base_tr)/(K.mean_tr/K.t_tr); K['score']=np.minimum(K.ex_tr,K.ex_te*1.3)
K=K.sort_values('score',ascending=False); K['side']=K.sig.str[:2]
K=K.drop_duplicates(['side','cond','bucket']).head(TOP)
rows=[]
for r in K.itertuples():
    cm=cmask(r.cond,r.bucket); on=nonover(cm&smask(r.sig),r.rule); off=nonover(cm,r.rule)
    a=st(on[on.yr<=2022]); b=st(on[on.yr>=2023]); a0=st(off[off.yr<=2022]); b0=st(off[off.yr>=2023])
    ys=on.groupby('yr').ret.sum(); pos=int((ys>0).sum())
    rows.append(dict(sig=r.sig,rule=r.rule,cond=r.cond,bucket=r.bucket,n1=a['n'],wr1=a['wr'],pf1=a['pf'],m1=a['mean'],off_pf1=a0['pf'],off_m1=a0['mean'],n2=b['n'],wr2=b['wr'],pf2=b['pf'],m2=b['mean'],off_pf2=b0['pf'],off_m2=b0['mean'],posyrs=pos,nyrs=len(ys),yrs=' '.join(f"{int(y)%100}:{v:+.1f}" for y,v in ys.items())))
R=pd.DataFrame(rows); R.to_pickle(f'FIN_{tf}.pkl')
good=R[(R.pf1>=1.2)&(R.pf2>=1.2)&(R.pf1>R.off_pf1)&(R.pf2>R.off_pf2)&(R.n1>=80)&(R.n2>=40)]
print(tf,'精査',len(R),'件 → 重複なしでも両期間でPF1.2以上かつスパイクなしを上回る:',len(good))
good=good.assign(s=np.minimum(good.pf1-good.off_pf1,good.pf2-good.off_pf2)).sort_values('s',ascending=False)
for r in good.head(14).itertuples():
    print(f"{r.sig} {r.rule} [{r.cond}={r.bucket}]\n   17-22: n{r.n1} 勝{100*r.wr1:.0f}% PF{r.pf1:.2f}(なし{r.off_pf1:.2f}) 平均{r.m1:+.3f}%(なし{r.off_m1:+.3f}) | 23-26: n{r.n2} 勝{100*r.wr2:.0f}% PF{r.pf2:.2f}(なし{r.off_pf2:.2f}) 平均{r.m2:+.3f}%(なし{r.off_m2:+.3f}) | プラス年{r.posyrs}/{r.nyrs} [{r.yrs}]")
