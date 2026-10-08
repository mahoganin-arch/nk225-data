import pandas as pd, numpy as np
def rma(x,n): return x.ewm(alpha=1/n,adjust=False).mean()
def rsi_(c,n=14):
    dl=c.diff(); return 100-100/(1+rma(dl.clip(lower=0),n)/rma(-dl.clip(upper=0),n))
def load(f):
    d=pd.read_csv(f); d.columns=[x.lower() for x in d.columns]
    d['t']=pd.to_datetime(d.time,unit='s',utc=True).dt.tz_convert('Asia/Tokyo'); d=d.drop_duplicates('t').sort_values('t').reset_index(drop=True)
    o,h,l,c,v=d.open,d.high,d.low,d.close,d.volume
    body=(c-o).abs(); top=np.maximum(o,c); bot=np.minimum(o,c); up=h-top; lo=bot-l
    us=(body>0)&(up>=3*body); ls=(body>0)&(lo>=3*body); ou=us&~ls; ol=ls&~us; bull=c>o
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); atr=rma(tr,14); d['atr']=atr
    d['pat']=np.select([ou&bull,ou&~bull,ol&bull,ol&~bull],['上陽','上陰','下陽','下陰'],'')
    d['grp']=np.select([ou,ol],['上髭','下髭'],'')
    for n in (1,3,5,10,20): d[f'f{n}']=(c.shift(-n)-o.shift(-1))/atr
    F={}
    F['RSI']=pd.cut(rsi_(c),[0,30,40,50,60,70,100]).astype(str)
    for n in (20,75,200):
        s=c.rolling(n).mean(); F[f'終値vsSMA{n}']=np.where(c>s,'上','下')
        F[f'SMA{n}傾き']=np.where(s>s.shift(5),'上向き','下向き')
        F[f'SMA{n}乖離(ATR)']=pd.qcut((c-s)/atr,5,labels=['Q1最も下','Q2','Q3','Q4','Q5最も上']).astype(str)
    s20=c.rolling(20).mean();s75=c.rolling(75).mean();s200=c.rolling(200).mean()
    F['MA並び']=np.select([(s20>s75)&(s75>s200),(s20<s75)&(s75<s200)],['上昇PO','下降PO'],'その他')
    ten=(h.rolling(9).max()+l.rolling(9).min())/2; kij=(h.rolling(26).max()+l.rolling(26).min())/2
    sa=((ten+kij)/2).shift(25); sb=((h.rolling(52).max()+l.rolling(52).min())/2).shift(25)
    ct=np.maximum(sa,sb); cb=np.minimum(sa,sb)
    F['一目:雲']=np.select([c>ct,c<cb],['雲の上','雲の下'],'雲の中')
    F['一目:転換vs基準']=np.where(ten>kij,'転換>基準','転換<基準')
    F['一目:終値vs基準線']=np.where(c>kij,'上','下')
    F['一目:遅行線']=np.select([c>top.shift(26),c<bot.shift(26)],['上','下'],'重なり')
    F['一目:三役']=np.select([(c>ct)&(ten>kij)&(c>h.shift(26)),(c<cb)&(ten<kij)&(c<l.shift(26))],['三役好転','三役逆転'],'その他')
    ll=l.rolling(14).min(); hh=h.rolling(14).max(); k=(100*(c-ll)/(hh-ll)).rolling(3).mean(); dd=k.rolling(3).mean()
    F['ストキャス%K']=pd.cut(k,[-1,20,50,80,101],labels=['0-20','20-50','50-80','80-100']).astype(str)
    F['ストキャスK vs D']=np.where(k>dd,'K>D','K<D')
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); sg=m.ewm(span=9,adjust=False).mean(); hs=m-sg
    F['MACD符号']=np.where(m>0,'MACD>0','MACD<0'); F['MACDヒスト']=np.where(hs>0,'ヒスト>0','ヒスト<0')
    F['MACDヒスト方向']=np.where(hs>hs.shift(),'拡大(上)','縮小(下)')
    upm=h.diff(); dnm=-l.diff(); pdm=np.where((upm>dnm)&(upm>0),upm,0.0); mdm=np.where((dnm>upm)&(dnm>0),dnm,0.0)
    pdi=100*rma(pd.Series(pdm),14)/atr; mdi=100*rma(pd.Series(mdm),14)/atr; adx=rma(100*(pdi-mdi).abs()/(pdi+mdi),14)
    F['ADX']=pd.cut(adx,[0,15,25,35,100],labels=['-15','15-25','25-35','35-']).astype(str)
    F['DI']=np.where(pdi>mdi,'+DI>-DI','+DI<-DI')
    F['ADX×DI']=np.where(adx>=25,np.where(pdi>mdi,'強トレンド上','強トレンド下'),'ADX25未満')
    vr=v/v.rolling(20).mean(); F['出来高/20平均']=pd.cut(vr,[0,0.7,1.0,1.5,2.5,999],labels=['-0.7','0.7-1','1-1.5','1.5-2.5','2.5-']).astype(str)
    F['値幅/ATR']=pd.cut((h-l)/atr,[0,0.7,1.0,1.5,99],labels=['-0.7','0.7-1','1-1.5','1.5-']).astype(str)
    F['髭/ATR']=pd.cut(np.maximum(up,lo)/atr,[0,0.5,1.0,99],labels=['-0.5','0.5-1','1-']).astype(str)
    hr=d.t.dt.hour+d.t.dt.minute/60
    F['時間帯']=np.select([(hr>=8.75)&(hr<11.5),(hr>=11.5)&(hr<15.75),(hr>=16.5)&(hr<21),(hr>=21)|(hr<2)],['日中前半','日中後半','夜間16:30-21','夜間21-2時'],'夜間2時-')
    F['曜日']=d.t.dt.dayofweek.map({0:'月',1:'火',2:'水',3:'木',4:'金',5:'土',6:'日'})
    # 上位足（確定済みの足のみ使用）
    x=d.set_index('t')
    for rule,nm in (('1h','1時間足'),('4h','4時間足'),('1D','日足')):
        g=x.resample(rule).agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
        g['r']=rsi_(g.close); g['s']=g.close.rolling(20).mean()
        g=g.shift(1); g.index=g.index  # 直前の確定足
        j=pd.merge_asof(d[['t']],g[['r','s','close']].reset_index(),on='t')
        F[f'{nm}RSI']=np.where(j.r.isna(),'nan',np.where(j.r>50,'50超','50未満'))
        F[f'{nm}:終値vsSMA20']=np.where(j.s.isna(),'nan',np.where(j.close>j.s,'上','下'))
    # ---- 追加特徴量 ----
    rt_up=up/body.replace(0,np.nan); rt_lo=lo/body.replace(0,np.nan)
    for k_ in (2,3,5):
        d[f'g{k_}']=np.select([(rt_up>=k_)&~(rt_lo>=k_),(rt_lo>=k_)&~(rt_up>=k_)],[f'上髭{k_}倍',f'下髭{k_}倍'],'')
    ph=h.shift(1).rolling(20).max(); pl=l.shift(1).rolling(20).min()
    F['20本高安スイープ']=np.select([(h>ph)&(c<ph),(l<pl)&(c>pl)],['高値スイープ','安値スイープ'],'なし')
    F['20本高安更新']=np.select([h>ph,l<pl],['高値更新','安値更新'],'なし')
    dd_=d.t.dt.date; dh=h.groupby(dd_).max(); dl_=l.groupby(dd_).min()
    pdh=dd_.map(dh.shift(1)); pdl=dd_.map(dl_.shift(1))
    F['前日高安']=np.select([(h>pdh)&(c<pdh),(l<pdl)&(c>pdl),c>pdh,c<pdl],['前日高値スイープ','前日安値スイープ','前日高値の上','前日安値の下'],'前日レンジ内')
    F['直前5本騰落(ATR)']=pd.cut((c.shift(1)-c.shift(6))/atr,[-99,-2,-0.7,0.7,2,99],labels=['-2以下','-2〜-0.7','±0.7','0.7〜2','2以上']).astype(str)
    sd=c.rolling(20).std(ddof=0); m20=c.rolling(20).mean()
    F['BB位置']=np.select([h>m20+2*sd,l<m20-2*sd],['上限タッチ','下限タッチ'],'バンド内')
    F['ATR水準']=pd.cut(atr/atr.rolling(200).mean(),[0,0.8,1.2,99],labels=['低','中','高']).astype(str)
    sp=((rt_up>=3)|(rt_lo>=3)).astype(float); F['直前5本にスパイク']=np.where(sp.shift(1).rolling(5).sum()>0,'あり','なし')
    pb=(c.shift(1)-o.shift(1))/atr; F['前足']=np.select([pb>0.7,pb<-0.7],['大陽線','大陰線'],'小')
    d['sma20']=m20; d['kij']=kij
    r_=rsi_(c); F['RSI50']=np.where(r_>50,'50超','50未満')
    F['出来高(粗)']=pd.cut(vr,[0,1.0,1.5,999],labels=['-1','1-1.5','1.5-']).astype(str)
    hac=(o+h+l+c)/4; hao=np.empty(len(d)); hao[0]=(o[0]+c[0])/2
    for i in range(1,len(d)): hao[i]=(hao[i-1]+hac[i-1])/2
    d['ha']=(hac>=hao).values; d['rsi']=r_
    st=np.where(c>top.shift(26),1,np.where(c<bot.shift(26),-1,0)); st=pd.Series(st).replace(0,np.nan).ffill().fillna(0).values; d['ch']=st
    for kx,vx in F.items(): d['F_'+kx]=np.asarray(vx)
    d=d.iloc[260:].reset_index(drop=True)
    mid=d.t.iloc[len(d)//2]; d['half']=np.where(d.t<mid,0,1)
    return d
