import pandas as pd, numpy as np, warnings, sys; warnings.filterwarnings('ignore')
from ld import bars
from s1 import rma,rsi_,atr_,ichi
def feats(d,tfmin):
    o,h,l,c,v=d.open,d.high,d.low,d.close,d.volume; t=d.t
    body=(c-o).abs(); top=np.maximum(o,c); bot=np.minimum(o,c); up=h-top; lo=bot-l; rng=h-l; a=atr_(h,l,c); bull=c>o; bear=c<o
    E={}
    for k in (2,3,5):
        E[f'髭|上髭{k}倍']=(body>0)&(up>=k*body)&~(lo>=k*body); E[f'髭|下髭{k}倍']=(body>0)&(lo>=k*body)&~(up>=k*body)
    E['髭|上髭がATR以上']=up>=a; E['髭|下髭がATR以上']=lo>=a
    E['実体|大陽線(実体>値幅60%,値幅>ATR)']=bull&(body>0.6*rng)&(rng>a); E['実体|大陰線(同)']=bear&(body>0.6*rng)&(rng>a)
    E['実体|特大陽線(実体>2ATR)']=bull&(body>2*a); E['実体|特大陰線(実体>2ATR)']=bear&(body>2*a)
    pb=(c.shift()-o.shift()); 
    E['実体|陽の包み足']=bull&(pb<0)&(c>=o.shift())&(o<=c.shift())&(body>body.shift()); E['実体|陰の包み足']=bear&(pb>0)&(c<=o.shift())&(o>=c.shift())&(body>body.shift())
    ins=(h<=h.shift())&(l>=l.shift()); E['実体|はらみ足']=ins
    E['実体|はらみ足から上放れ']=ins.shift().fillna(False)&(c>h.shift()); E['実体|はらみ足から下放れ']=ins.shift().fillna(False)&(c<l.shift())
    for n in (3,5):
        E[f'実体|陽線{n}連続']=bull.rolling(n).sum()==n; E[f'実体|陰線{n}連続']=bear.rolling(n).sum()==n
    E['値幅|NR7(7本で最小の値幅)']=rng==rng.rolling(7).min()
    E['値幅|値幅2ATR超で陽線']=(rng>2*a.shift())&bull; E['値幅|値幅2ATR超で陰線']=(rng>2*a.shift())&bear
    sq=(a.rolling(20).mean()/a.rolling(100).mean()); E['値幅|収縮(ATR20本/100本<0.7)']=sq<0.7
    gapt=(d.m.diff()>tfmin*1.5); g=(o-c.shift())/a.shift()
    E['窓|セッション間の上窓(>0.5ATR)']=gapt&(g>0.5); E['窓|セッション間の下窓(<-0.5ATR)']=gapt&(g<-0.5)
    wk=(d.m.diff()>60*30); E['窓|週明け上窓(>0.5ATR)']=wk&(g>0.5); E['窓|週明け下窓(<-0.5ATR)']=wk&(g<-0.5)
    for n in (20,50,100):
        ph=h.shift().rolling(n).max(); pl=l.shift().rolling(n).min()
        E[f'高安|{n}本高値を終値で更新']=c>ph; E[f'高安|{n}本安値を終値で更新']=c<pl
        E[f'高安|{n}本高値スイープ']=(h>ph)&(c<ph); E[f'高安|{n}本安値スイープ']=(l<pl)&(c>pl)
    dd=t.dt.date; dh=h.groupby(dd).max(); dl=l.groupby(dd).min(); pdh=dd.map(dh.shift()); pdl=dd.map(dl.shift())
    E['高安|前日高値を終値で上抜け']=(c>pdh)&(c.shift()<=pdh); E['高安|前日安値を終値で下抜け']=(c<pdl)&(c.shift()>=pdl)
    E['高安|前日高値スイープ']=(h>pdh)&(c<pdh); E['高安|前日安値スイープ']=(l<pdl)&(c>pdl)
    ten,kij,ct,cb=ichi(h,l,c); ct=pd.Series(ct,index=c.index); cb=pd.Series(cb,index=c.index)
    L={'SMA20':c.rolling(20).mean(),'SMA75':c.rolling(75).mean(),'SMA200':c.rolling(200).mean(),'基準線':kij}
    sid=gapt.cumsum(); tp=(h+l+c)/3; L['VWAP']=(tp*v).groupby(sid).cumsum()/v.groupby(sid).cumsum()
    for k,s in L.items():
        E[f'押し戻り|{k}に上から触れて反発']=(l<=s)&(c>s)&(c.shift()>s.shift()); E[f'押し戻り|{k}に下から触れて反落']=(h>=s)&(c<s)&(c.shift()<s.shift())
        E[f'抜け|{k}を上抜け']=(c>s)&(c.shift()<=s.shift()); E[f'抜け|{k}を下抜け']=(c<s)&(c.shift()>=s.shift())
    E['押し戻り|雲上限に上から触れて反発']=(l<=ct)&(c>ct)&(c.shift()>ct.shift()); E['押し戻り|雲下限に下から触れて反落']=(h>=cb)&(c<cb)&(c.shift()<cb.shift())
    E['抜け|雲を上抜け']=(c>ct)&(c.shift()<=ct.shift()); E['抜け|雲を下抜け']=(c<cb)&(c.shift()>=cb.shift())
    E['抜け|転換線が基準線を上抜け']=(ten>kij)&(ten.shift()<=kij.shift()); E['抜け|転換線が基準線を下抜け']=(ten<kij)&(ten.shift()>=kij.shift())
    s20,s75=L['SMA20'],L['SMA75']; E['抜け|SMA20/75ゴールデンクロス']=(s20>s75)&(s20.shift()<=s75.shift()); E['抜け|SMA20/75デッドクロス']=(s20<s75)&(s20.shift()>=s75.shift())
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); sg=m.ewm(span=9,adjust=False).mean()
    E['抜け|MACDがシグナルを上抜け']=(m>sg)&(m.shift()<=sg.shift()); E['抜け|MACDがシグナルを下抜け']=(m<sg)&(m.shift()>=sg.shift())
    r=rsi_(c)
    E['行き過ぎ|RSI30未満']=r<30; E['行き過ぎ|RSI70超']=r>70; E['行き過ぎ|RSI20未満']=r<20; E['行き過ぎ|RSI80超']=r>80
    E['行き過ぎ|RSIが30を上抜け']=(r>30)&(r.shift()<=30); E['行き過ぎ|RSIが70を下抜け']=(r<70)&(r.shift()>=70)
    ll=l.rolling(14).min(); hh=h.rolling(14).max(); k_=(100*(c-ll)/(hh-ll)).rolling(3).mean()
    E['行き過ぎ|ストキャス20未満']=k_<20; E['行き過ぎ|ストキャス80超']=k_>80
    sd=c.rolling(20).std(ddof=0); 
    E['行き過ぎ|BB-2σを安値で割る']=l<s20-2*sd; E['行き過ぎ|BB+2σを高値で超える']=h>s20+2*sd
    E['行き過ぎ|BB-2σの外から内へ戻る']=(c>s20-2*sd)&(c.shift()<(s20-2*sd).shift()); E['行き過ぎ|BB+2σの外から内へ戻る']=(c<s20+2*sd)&(c.shift()>(s20+2*sd).shift())
    E['行き過ぎ|強気ダイバージェンス(20本安値更新・RSIは非更新)']=(l<l.shift().rolling(20).min())&(r>r.shift().rolling(20).min())
    E['行き過ぎ|弱気ダイバージェンス(20本高値更新・RSIは非更新)']=(h>h.shift().rolling(20).max())&(r<r.shift().rolling(20).max())
    mom=(c-c.shift(5))/a; E['行き過ぎ|5本で+2ATR超の急騰']=mom>2; E['行き過ぎ|5本で-2ATR超の急落']=mom<-2
    vr=v/v.rolling(20).mean()
    E['出来高|1.5倍超で陽線']=(vr>1.5)&bull; E['出来高|1.5倍超で陰線']=(vr>1.5)&bear; E['出来高|2.5倍超で陽線']=(vr>2.5)&bull; E['出来高|2.5倍超で陰線']=(vr>2.5)&bear; E['出来高|0.5倍未満']=vr<0.5
    # 時間：セッション基準
    nxt=(d.m.shift(-1)-d.m>tfmin*1.5); hr=t.dt.hour; day=(hr>=8)&(hr<16)
    E['時間|日中の最初の足']=gapt&day; E['時間|日中の最後の足']=nxt&day; E['時間|夜間の最初の足']=gapt&~day; E['時間|夜間の最後の足']=nxt&~day
    pos=d.groupby(sid).cumcount()
    E['時間|日中2〜3本目(1時間換算)']=day&(pos>=60//tfmin)&(pos<180//tfmin)
    E['時間|日中後半(12時以降)']=day&(hr>=12); E['時間|夜間前半(〜21時)']=~day&(hr>=16)&(hr<21); E['時間|夜間NY時間(21〜2時)']=(hr>=21)|(hr<2); E['時間|夜間明け方(2時〜)']=(hr>=2)&(hr<8)
    for i,n in enumerate('月火水木金'): E[f'時間|{n}曜']=t.dt.dayofweek==i
    dom=t.dt.day; E['時間|SQ週(第2金曜の週)']=((dom-t.dt.dayofweek+4)>=8)&((dom-t.dt.dayofweek+4)<=14)&(t.dt.dayofweek<=4)
    E['時間|月初(1〜3日)']=dom<=3; E['時間|月末(26日〜)']=dom>=26
    return {k:np.asarray(pd.Series(x).fillna(False),bool) for k,x in E.items()},a
def nonover(idx,h):
    out=[]; free=0
    for i in idx:
        if i>=free: out.append(i); free=i+h+1
    return np.array(out,int)
def run(f,tfmin,name):
    d=bars(f); E,a=feats(d,tfmin); o,c=d.open.values,d.close.values; n=len(d); yr=d.t.dt.year.values
    P=np.where(yr<=2022,0,1); ok=np.arange(n)>=260; rows=[]
    for H in (5,20,40):
        fr=np.full(n,np.nan); fr[:n-H]=(c[H:]-o[1:n-H+1])/c[:n-H]*100   # 次足始値→H本目終値
        base=[np.nanmean(fr[ok&(P==p)]) for p in (0,1)]
        for k,mk in E.items():
            idx=nonover(np.where(mk&ok&~np.isnan(fr))[0],H); r=dict(足=name,H=H,分類=k.split('|')[0],形=k.split('|')[1])
            for p,lab in ((0,'17-22'),(1,'23-26')):
                z=fr[idx[P[idx]==p]]; nn=len(z)
                r[lab+' n']=nn; r[lab+' 平均%']=round(z.mean(),3) if nn>=30 else np.nan
                r[lab+' 超過%']=round(z.mean()-base[p],3) if nn>=30 else np.nan
                r[lab+' t']=round((z.mean()-base[p])/(z.std()/np.sqrt(nn)),1) if nn>=30 else np.nan
            rows.append(r)
        print(name,'H',H,'基準平均%',[round(b,3) for b in base])
    return pd.DataFrame(rows)
if __name__=='__main__':
    T=pd.concat([run('h1.csv',60,'1時間'),run('m30.csv',30,'30分')]); T.to_csv('stage1_entry.csv',index=False,encoding='utf-8-sig'); T.to_pickle('T.pkl')
    print(len(T),'行')
