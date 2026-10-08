import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from numba import njit
from ld import bars
from s1 import rma,rsi_,atr_,ichi
@njit(cache=True)
def sim(mask,conf,o,h,l,c,atr,sm,sk,stopArr,tm,tk,tgtArr,trm,trk,trArr,part,exA,endA,tmax,warm):
    n=len(c); R=np.full(n,np.nan); HL=np.zeros(n); free=0
    for i in range(warm,n-70):
        if not mask[i] or i<free: continue
        a=atr[i]
        if conf:
            lv=h[i]+5; j=-1
            for q in range(i+1,i+4):
                if h[q]>=lv: j=q; break
            if j<0: continue
            e=max(lv,o[j])
        else: j=i+1; e=o[j]
        st=-1e18
        if sm==1: st=l[i]-5
        elif sm==2: st=e-sk*a
        elif sm==3: st=stopArr[i]-5
        elif sm==4: st=e-sk/100*abs(e)
        if st>=e: st=e-0.25*a
        tg=1e18
        if tm==1: tg=e+tk*a
        elif tm==2 and st>-1e17: tg=e+tk*(e-st)
        elif tm==3 and tgtArr[i]>e: tg=tgtArr[i]
        hh=h[i]; x=np.nan; banked=False; pv=0.0; t=j
        while True:
            if tm==4: tg=tgtArr[t-1] if tgtArr[t-1]>e else 1e18
            if t>j and o[t]<=st: x=o[t]
            elif l[t]<=st and not (t==j and conf and st>=e): x=st
            if np.isnan(x):
                if part>0 and not banked and h[t]>=e+part*a: banked=True; pv=max(e+part*a,o[t]) if t>j else e+part*a
                if t>j and o[t]>=tg: x=o[t]
                elif h[t]>=tg: x=tg
            if np.isnan(x) and (exA[t] or endA[t] or t-j+1>=tmax): x=c[t]
            if not np.isnan(x): break
            if h[t]>hh: hh=h[t]
            if trm==1: st=max(st,hh-trk*a)
            elif trm==2: st=max(st,c[t]-trk*a)
            elif trm==3: st=max(st,trArr[t])
            elif trm==4 and hh>=e+trk*a: st=max(st,e)
            t+=1
        pnl=(x-e) if not banked else 0.5*(pv-e)+0.5*(x-e)
        R[i]=(pnl-10)/abs(c[i])*100; HL[i]=t-j+1; free=t+1
    return R,HL
def prep(b,tf,mirror=False):
    o,h,l,c=b.open,b.high,b.low,b.close
    if mirror: o,h,l,c=-o,-l,-h,-c
    a=atr_(h,l,c); A=dict(o=o.values,h=h.values,l=l.values,c=c.values,atr=a.values)
    for n in (5,10,20): A[f'low{n}']=l.rolling(n).min().values
    s20=c.rolling(20).mean(); sd=c.rolling(20).std(ddof=0); A['bbU']=(s20+2*sd).values; A['sma20']=s20.values
    dd=b.t.dt.date; A['pdh']=dd.map(h.groupby(dd).max().shift()).values.astype(float)
    ten,kij,ct,cb=ichi(h,l,c); r=rsi_(c); hc=(o+h+l+c)/4; ho=hc.ewm(alpha=0.5,adjust=False).mean().shift()
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); sg=m.ewm(span=9,adjust=False).mean()
    X={'平均足が陰転':hc<ho,'RSIが50割れ':(r<50)&(r.shift()>=50),'遅行線が26本前の終値割れ':c<c.shift(25),'終値がSMA20割れ':c<s20,'終値が基準線割れ':c<kij,'終値が転換線割れ':c<ten,
       'MACDがシグナル割れ':(m<sg)&(m.shift()>=sg.shift()),'終値が雲の下限割れ':c<pd.Series(cb,index=c.index),'陰線2連続':(c<o)&(c.shift()<o.shift()),'前の足の安値を終値で割る':c<l.shift()}
    A['X']={k:np.asarray(v.fillna(False),bool) for k,v in X.items()}
    gap=(b.m.shift(-1)-b.m>tf*1.5).values; hr=b.t.dt.hour.values; day=(hr>=8)&(hr<16)
    A['E']={'セッション引け':gap,'日中引け':gap&day,'夜間引け':gap&~day,'週末引け':(b.m.shift(-1)-b.m>60*30).values}
    A['none']=np.zeros(len(b),bool); A['nan']=np.full(len(b),np.nan); return A
def exits():
    L=[]; add=lambda nm,cat,sm=0,sk=0,sa='nan',tm=0,tk=0,ta='nan',trm=0,trk=0,tra='nan',part=0,ex=None,en=None,tmax=60: L.append(dict(nm=nm,cat=cat,sm=sm,sk=sk,sa=sa,tm=tm,tk=tk,ta=ta,trm=trm,trk=trk,tra=tra,part=part,ex=ex,en=en,tmax=tmax))
    for n in (3,5,10,20,40,60): add(f'{n}本後','時間',tmax=n)
    for k in ('セッション引け','日中引け','夜間引け','週末引け'): add(k,'時間',en=k,tmax=400)
    for T in (20,40):
        add(f'損切り:シグナル足の安値 +{T}本','損切り',sm=1,tmax=T)
        for k in (1,1.5,2,3): add(f'損切り:ATR{k}倍 +{T}本','損切り',sm=2,sk=k,tmax=T)
        for n in (5,10,20): add(f'損切り:直近{n}本安値 +{T}本','損切り',sm=3,sa=f'low{n}',tmax=T)
        for k in (0.5,1.0,2.0): add(f'損切り:{k}% +{T}本','損切り',sm=4,sk=k,tmax=T)
        for k in (1,2,3,5): add(f'利確:ATR{k}倍 +{T}本','利確',tm=1,tk=k,tmax=T)
        add(f'利確:前日高値 +{T}本','利確',tm=3,ta='pdh',tmax=T); add(f'利確:BB+2σ到達 +{T}本','利確',tm=4,ta='bbU',tmax=T)
        for k in (1,2,3): add(f'損切りATR1.5倍・利確はその{k}倍 +{T}本','損切り+利確',sm=2,sk=1.5,tm=2,tk=k,tmax=T)
        add(f'損切りATR2倍・利確ATR3倍 +{T}本','損切り+利確',sm=2,sk=2,tm=1,tk=3,tmax=T)
        add(f'損切りATR3倍・利確ATR2倍 +{T}本','損切り+利確',sm=2,sk=3,tm=1,tk=2,tmax=T)
        add(f'建値移動(+1ATRで) 損切りATR1.5倍 +{T}本','追随',sm=2,sk=1.5,trm=4,trk=1,tmax=T)
        add(f'半分をATR1倍で利確・残りは{T}本後','分割',part=1,tmax=T); add(f'半分をATR2倍で利確・残りは{T}本後','分割',part=2,tmax=T)
    for k in (2,3,4): add(f'トレール:高値からATR{k}倍','追随',sm=2,sk=k,trm=1,trk=k)
    for k in (2,3): add(f'トレール:終値からATR{k}倍','追随',sm=2,sk=k,trm=2,trk=k)
    for n in (5,10,20): add(f'トレール:直近{n}本安値','追随',sm=3,sa=f'low{n}',trm=3,tra=f'low{n}')
    for k in ('平均足が陰転','RSIが50割れ','遅行線が26本前の終値割れ','終値がSMA20割れ','終値が基準線割れ','終値が転換線割れ','MACDがシグナル割れ','終値が雲の下限割れ','陰線2連続','前の足の安値を終値で割る'): add('指標:'+k,'指標',ex=k)
    return L
def runx(A,mask,conf,x,warm=300):
    return sim(mask,conf,A['o'],A['h'],A['l'],A['c'],A['atr'],x['sm'],float(x['sk']),A[x['sa']],x['tm'],float(x['tk']),A[x['ta']],x['trm'],float(x['trk']),A[x['tra']],float(x['part']),
               A['X'][x['ex']] if x['ex'] else A['none'],A['E'][x['en']] if x['en'] else A['none'],x['tmax'],warm)
def stat(R,HL,sel):
    z=R[sel&~np.isnan(R)]; hl=HL[sel&~np.isnan(R)]
    if len(z)<20: return dict(n=len(z),PF=np.nan,平均=np.nan,勝率=np.nan,本数=np.nan)
    return dict(n=len(z),PF=round(z[z>0].sum()/max(-z[z<0].sum(),1e-9),2),平均=round(z.mean(),3),勝率=round(100*(z>0).mean()),本数=round(hl.mean(),1))
def signals(b,tf):
    o,h,l,c,v=b.open,b.high,b.low,b.close,b.volume; body=(c-o).abs(); up=h-np.maximum(o,c); lo=np.minimum(o,c)-l; a=atr_(h,l,c)
    sp=lambda k,u:((body>0)&((up if u else lo)>=k*body)&~((lo if u else up)>=k*body))
    ten,kij,ct,cb=ichi(h,l,c); hr=b.t.dt.hour+b.t.dt.minute/60; S={}
    if tf==60:
        S['A']=(sp(2,True)&(hr>=8.75)&(hr<11.5),False); S['B']=(sp(3,True)&(c>ct)&(v>v.rolling(20).mean()*1.5),False)
        S['C']=(sp(5,False)&(c<cb)&(ten<kij)&(c<l.shift(26)),True)
    else:
        ph=h.shift().rolling(20).max(); pl=l.shift().rolling(20).min(); mom=(c.shift(1)-c.shift(6))/a
        S['D']=(sp(2,False)&(l<pl)&(c>pl)&~((h>ph)&(c<ph))&(mom>-2)&(mom<=-0.7),True)
    return {k:(np.asarray(m.fillna(False),bool),cf) for k,(m,cf) in S.items()}
if __name__=='__main__':
    b1=bars('h1.csv'); b3=bars('m30.csv'); EX=exits(); print(len(EX),'通りのエグジット')
    rows=[]
    for b,tf,nm in ((b1,60,'1時間'),(b3,30,'30分')):
        p2=(b.t.dt.year.values>=2023); allm=np.ones(len(b),bool); SG=signals(b,tf)
        for side,mir in (('買い',False),('売り',True)):
            A=prep(b,tf,mir)
            for x in EX:
                R,HL=runx(A,allm,False,x); r=dict(足=nm,入り方='全足(中立)',売買=side,分類=x['cat'],決済=x['nm'])
                for lab,sel in (('17-22',~p2),('23-26',p2)):
                    s=stat(R,HL,sel); r.update({lab+' n':s['n'],lab+' PF':s['PF'],lab+' 平均%':s['平均'],lab+' 勝率':s['勝率'],lab+' 保有本数':s['本数']})
                rows.append(r)
                if not mir:
                    for k,(m,cf) in SG.items():
                        R,HL=runx(A,m,cf,x); r=dict(足=nm,入り方='候補'+k,売買=side,分類=x['cat'],決済=x['nm'])
                        for lab,sel in (('17-22',~p2),('23-26',p2)):
                            s=stat(R,HL,sel); r.update({lab+' n':s['n'],lab+' PF':s['PF'],lab+' 平均%':s['平均'],lab+' 勝率':s['勝率'],lab+' 保有本数':s['本数']})
                        rows.append(r)
    T=pd.DataFrame(rows); T.to_pickle('TX.pkl'); T.to_csv('/mnt/user-data/outputs/stage1_3_エグジット一覧.csv',index=False,encoding='utf-8-sig'); print(len(T))
