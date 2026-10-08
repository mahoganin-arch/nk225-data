import numpy as np, pandas as pd, itertools, time, sys
from numba import njit
from feat import load
EX=['T3','T5','T10','T20','T40','髭S+T10','髭S+T20','髭S+T40','髭S+1R','髭S+2R','髭S+3R','髭S+建値+T40','ATR1.5S+T20','ATR1S+1T','ATR1S+2T','ATR1.5S+3T','トレール2ATR','トレール3ATR','平均足','平均足+髭S','RSI50','RSI50+髭S','遅行線','遅行線+髭S','SMA20割れ','基準線割れ']
MODES=['成行','確認','指値']
NE=26
@njit(cache=True)
def stp(di,lo,hi,op,lv,e):
    if di==0:
        if lo<=lv: return True, min(lv,op)-e
    else:
        if hi>=lv: return True, e-max(lv,op)
    return False,0.0
@njit(cache=True)
def tgt(di,lo,hi,lv):
    return (hi>=lv) if di==0 else (lo<=lv)
@njit(cache=True)
def sim(o,h,l,c,atr,ha,rsi,ch,sma,kij):
    n=len(o); out=np.full((3,2,NE,n),np.nan); exo=np.full((3,2,NE,n),np.nan); M=60; TN=(3,5,10,20,40)
    for i in range(60,n-M-6):
        a=atr[i]
        for m in range(3):
            for di in range(2):
                d=1.0 if di==0 else -1.0
                je=-1; e=0.0
                if m==0: je=i+1; e=o[i+1]
                elif m==1:
                    lv=h[i]+5 if di==0 else l[i]-5
                    for j in range(i+1,i+4):
                        if di==0 and h[j]>=lv: je=j; e=max(lv,o[j]); break
                        if di==1 and l[j]<=lv: je=j; e=min(lv,o[j]); break
                else:
                    lv=(c[i]+l[i])/2 if di==0 else (c[i]+h[i])/2
                    for j in range(i+1,i+4):
                        if di==0 and l[j]<=lv: je=j; e=min(lv,o[j]); break
                        if di==1 and h[j]>=lv: je=j; e=max(lv,o[j]); break
                if je<0: continue
                ws=l[i] if di==0 else h[i]; risk=d*(e-ws); sv=risk>0
                done=np.zeros(NE,np.bool_); ret=np.zeros(NE); kd=np.zeros(NE)
                be=False; best2=e; best3=e
                for j in range(je,je+M):
                    k=j-je+1; first=(m>0 and j==je)
                    lo=l[j]; hi=h[j]; op=e if first else o[j]; cl=d*(c[j]-e)
                    for x in range(5):
                        if not done[x] and k==TN[x]: done[x]=True; ret[x]=cl
                    if sv:
                        hit,r=stp(di,lo,hi,op,ws,e)
                        for x,N in ((5,10),(6,20),(7,40)):
                            if not done[x]:
                                if hit: done[x]=True; ret[x]=r
                                elif k==N: done[x]=True; ret[x]=cl
                        for x,kk in ((8,1.0),(9,2.0),(10,3.0)):
                            if not done[x]:
                                if hit: done[x]=True; ret[x]=r
                                elif (not first) and tgt(di,lo,hi,e+d*kk*risk): done[x]=True; ret[x]=kk*risk
                                elif k==40: done[x]=True; ret[x]=cl
                        if not done[11]:
                            hb,rb=stp(di,lo,hi,op,e if be else ws,e)
                            if hb: done[11]=True; ret[11]=rb
                            elif k==40: done[11]=True; ret[11]=cl
                            elif (not first) and tgt(di,lo,hi,e+d*risk): be=True
                        for x,fl in ((19,0),(21,1),(23,2)):
                            if not done[x] and hit: done[x]=True; ret[x]=r
                    h1,r1=stp(di,lo,hi,op,e-d*a,e); h15,r15=stp(di,lo,hi,op,e-d*1.5*a,e)
                    if not done[12]:
                        if h15: done[12]=True; ret[12]=r15
                        elif k==20: done[12]=True; ret[12]=cl
                    for x,kk in ((13,1.0),(14,2.0)):
                        if not done[x]:
                            if h1: done[x]=True; ret[x]=r1
                            elif (not first) and tgt(di,lo,hi,e+d*kk*a): done[x]=True; ret[x]=kk*a
                            elif k==40: done[x]=True; ret[x]=cl
                    if not done[15]:
                        if h15: done[15]=True; ret[15]=r15
                        elif (not first) and tgt(di,lo,hi,e+d*3*a): done[15]=True; ret[15]=3*a
                        elif k==40: done[15]=True; ret[15]=cl
                    if not done[16]:
                        ht,rt=stp(di,lo,hi,op,best2-d*2*a,e)
                        if ht: done[16]=True; ret[16]=rt
                        elif k==M: done[16]=True; ret[16]=cl
                        elif d*(c[j]-best2)>0: best2=c[j]
                    if not done[17]:
                        ht,rt=stp(di,lo,hi,op,best3-d*3*a,e)
                        if ht: done[17]=True; ret[17]=rt
                        elif k==M: done[17]=True; ret[17]=cl
                        elif d*(c[j]-best3)>0: best3=c[j]
                    fha=((not ha[j]) and ha[j-1]) if di==0 else (ha[j] and (not ha[j-1]))
                    fr=(rsi[j]<50 and rsi[j-1]>=50) if di==0 else (rsi[j]>50 and rsi[j-1]<=50)
                    fc=(ch[j]==-1 and ch[j-1]==1) if di==0 else (ch[j]==1 and ch[j-1]==-1)
                    fs=(c[j]<sma[j] and c[j-1]>=sma[j-1]) if di==0 else (c[j]>sma[j] and c[j-1]<=sma[j-1])
                    fk=(c[j]<kij[j] and c[j-1]>=kij[j-1]) if di==0 else (c[j]>kij[j] and c[j-1]<=kij[j-1])
                    for x,f in ((18,fha),(19,fha),(20,fr),(21,fr),(22,fc),(23,fc),(24,fs),(25,fk)):
                        if not done[x] and (f or k==M): done[x]=True; ret[x]=cl
                    for x in range(NE):
                        if done[x] and kd[x]==0: kd[x]=k
                for x in range(NE):
                    if done[x] and (sv or not (x in (5,6,7,8,9,10,11,19,21,23))): out[m,di,x,i]=ret[x]; exo[m,di,x,i]=je+kd[x]-1-i
    return out,exo

COST=10.0
CORE=['RSI50','一目:雲','一目:遅行線','終値vsSMA200','ADX×DI','出来高(粗)','時間帯','髭/ATR','1時間足RSI','4時間足RSI','日足RSI','4時間足:終値vsSMA20','日足:終値vsSMA20','ストキャス%K','MACDヒスト','20本高安スイープ','前日高安','直前5本騰落(ATR)','BB位置','ATR水準','直前5本にスパイク','前足','日足200日線乖離','日足環境','日足60日騰落']
def stats(Xs,keys):
    G=lambda Z: Z.groupby(keys).sum()
    return dict(n=Xs.notna().groupby(keys).sum(),s=G(Xs),q=G(Xs**2),p=G(Xs.clip(lower=0)),w=G((Xs>0).astype(float)))
def daily():
    d=pd.read_csv('dd.csv'); d['t']=pd.to_datetime(d.time,unit='s',utc=True).dt.tz_convert('Asia/Tokyo')
    d['date']=d.t.dt.tz_localize(None).dt.normalize().astype('datetime64[ns]')
    s200=d.close.rolling(200).mean(); up=s200>s200.shift(20)
    d['F_日足環境']=np.select([(d.close>s200)&up,(d.close<s200)&~up],['上昇','下落'],'中間')
    d['dev200']=(d.close/s200-1)*100; d['r60']=(d.close/d.close.shift(60)-1)*100
    d['F_日足200日線乖離']=pd.cut(d.dev200,[-99,-5,0,5,10,99],labels=['-5%以下','-5〜0%','0〜5%','5〜10%','10%超']).astype(str)
    d['F_日足60日騰落']=pd.cut(d.r60,[-99,-5,0,5,10,99],labels=['-5%以下','-5〜0%','0〜5%','5〜10%','10%超']).astype(str)
    c=['F_日足環境','F_日足200日線乖離','F_日足60日騰落','dev200','r60']; D=d[['date']+c].copy(); D[c]=D[c].shift(2); return D
def prep(tf,f):
    d=load(f); d['date_']=d.t.dt.tz_localize(None).dt.normalize().astype('datetime64[ns]')
    d=pd.merge_asof(d,daily(),left_on='date_',right_on='date'); n=len(d); f8=lambda x:d[x].values.astype(np.float64)
    out,exo=sim(f8('open'),f8('high'),f8('low'),f8('close'),f8('atr'),d.ha.values,f8('rsi'),f8('ch'),f8('sma20'),f8('kij'))
    out=(out-COST)/d.close.values[None,None,None,:]*100
    cols=[m+'|'+('買' if di==0 else '売')+'|'+e for m in MODES for di in range(2) for e in EX]
    X=pd.DataFrame(out.reshape(len(cols),n).T,columns=cols); E=exo.reshape(len(cols),n).T
    yr=d.t.dt.year.values; d['s3']=np.where(yr<=2019,0,np.where(yr<=2022,1,2))
    return d,X,E,cols
def run(tf,d,X,cols):
    n=len(d); s2=(d.s3.values==2).astype(int)
    feats=[c for c in d.columns if c.startswith('F_')]
    if tf=='1時間': feats=[x for x in feats if '1時間足' not in x]
    conds=[('なし',None)]+[(f[2:],[f]) for f in feats]+[(a+' & '+b,['F_'+a,'F_'+b]) for a,b in itertools.combinations([x for x in CORE if 'F_'+x in feats],2)]
    keep=[]; tot=0; totok=0; ncol=len(cols); colarr=np.array(cols)
    for name,fs in conds:
        key=pd.Series('-',index=d.index) if fs is None else (d[fs[0]].astype(str) if len(fs)==1 else d[fs[0]].astype(str)+' & '+d[fs[1]].astype(str))
        base=X.groupby([pd.Series(s2),key]).mean()
        for lab in ('g2','g3','g5','pat'):
            msk=(d[lab]!='').values
            Xs=X[msk]; S=stats(Xs,[d.s3[msk].values,key[msk].values,d[lab][msk].values])
            idx=S['n'].index.droplevel(0).unique()
            def part(st,k):
                try: return S[st].xs(k,level=0).reindex(idx).fillna(0).values
                except KeyError: return np.zeros((len(idx),ncol))
            A={st:[part(st,k) for k in (0,1,2)] for st in S}
            ntr=A['n'][0]+A['n'][1]; str_=A['s'][0]+A['s'][1]; qtr=A['q'][0]+A['q'][1]; ptr=A['p'][0]+A['p'][1]; wtr=A['w'][0]+A['w'][1]
            nte=A['n'][2]; ste=A['s'][2]; qte=A['q'][2]; pte=A['p'][2]; wte=A['w'][2]
            with np.errstate(all='ignore'):
                mtr=str_/ntr; sdtr=np.sqrt(np.maximum(qtr/ntr-mtr**2,1e-12)); ttr=mtr/(sdtr/np.sqrt(ntr)); pftr=ptr/np.maximum(ptr-str_,1e-12)
                mte=ste/nte; sdte=np.sqrt(np.maximum(qte/nte-mte**2,1e-12)); tte=mte/(sdte/np.sqrt(nte)); pfte=pte/np.maximum(pte-ste,1e-12)
                m0=A['s'][0]/A['n'][0]; m1=A['s'][1]/A['n'][1]
                bk=idx.get_level_values(0)
                btr=base.xs(0,level=0).reindex(bk).values; bte=base.xs(1,level=0).reindex(bk).values
                pop=(ntr>=150)&(nte>=50)&np.array(['nan' not in b for b in bk])[:,None]
                ok=(pfte>=1.15)&(mte>bte)
                sel=pop&(pftr>=1.25)&(ttr>=2.5)&((mtr-btr)/(sdtr/np.sqrt(ntr))>=2.0)&(m0>0)&(m1>0)&(A['n'][0]>=50)&(A['n'][1]>=50)
            tot+=int(pop.sum()); totok+=int((pop&ok).sum())
            r,cc=np.where(sel)
            for a,b in zip(r,cc):
                keep.append(dict(tf=tf,cond=name,bucket=idx[a][0],sig=idx[a][1],rule=colarr[b],n_tr=ntr[a,b],wr_tr=wtr[a,b]/ntr[a,b],pf_tr=pftr[a,b],mean_tr=mtr[a,b],base_tr=btr[a,b],t_tr=ttr[a,b],
                                 n_te=nte[a,b],wr_te=wte[a,b]/nte[a,b],pf_te=pfte[a,b],mean_te=mte[a,b],base_te=bte[a,b],t_te=tte[a,b],ok=bool(ok[a,b])))
    return pd.DataFrame(keep),tot,totok
if __name__=='__main__':
    import pickle
    for tf,f in (('1時間','L60.csv'),('30分','L30.csv')):
        t0=time.time(); d,X,E,cols=prep(tf,f)
        pickle.dump((d,X.astype('float32'),E.astype('float32'),cols),open(f'P_{tf}.pkl','wb'),protocol=4)
        K,tot,totok=run(tf,d,X,cols); K.to_pickle(f'KL_{tf}.pkl')
        print(tf,'足数',len(d),'候補総数',tot,'無作為合格率%.1f%%'%(100*totok/max(tot,1)),'探索合格',len(K),'うち検証合格',int(K.ok.sum()) if len(K) else 0,'(%.0f%%)'%(100*K.ok.mean() if len(K) else 0),'検証PF中央値%.2f'%(K.pf_te.median() if len(K) else float('nan')),round(time.time()-t0),'s',flush=True)
