import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from ld import bars
def rma(x,n): return x.ewm(alpha=1/n,adjust=False).mean()
def rsi_(c,n=14):
    dl=c.diff(); return 100-100/(1+rma(dl.clip(lower=0),n)/rma(-dl.clip(upper=0),n))
def atr_(h,l,c,n=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); return rma(tr,n)
def adx_(h,l,c,n=14):
    a=atr_(h,l,c,n); up=h.diff(); dn=-l.diff()
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=h.index); mdm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=h.index)
    pdi=100*rma(pdm,n)/a; mdi=100*rma(mdm,n)/a; return rma(100*(pdi-mdi).abs()/(pdi+mdi),n),pdi,mdi
def ichi(h,l,c):
    ten=(h.rolling(9).max()+l.rolling(9).min())/2; kij=(h.rolling(26).max()+l.rolling(26).min())/2
    sa=((ten+kij)/2).shift(25); sb=((h.rolling(52).max()+l.rolling(52).min())/2).shift(25)
    return ten,kij,np.maximum(sa,sb),np.minimum(sa,sb)
def daily():
    d=pd.read_csv('d1.csv'); d.columns=[x.lower() for x in d.columns]
    d['t']=pd.to_datetime(d.time,unit='s',utc=True).dt.tz_convert('Asia/Tokyo'); d=d.sort_values('t').reset_index(drop=True)
    o,h,l,c=d.open,d.high,d.low,d.close; R={}
    for n in (25,75,200): 
        s=c.rolling(n).mean(); R[f'終値vs{n}日線']=np.where(c>s,'上','下'); R[f'{n}日線の傾き']=np.where(s>s.shift(20 if n>25 else 5),'上向き','下向き')
    s25,s75,s200=[c.rolling(n).mean() for n in (25,75,200)]
    R['200日線:位置×傾き']=np.select([(c>s200)&(s200>s200.shift(20)),(c<s200)&(s200<s200.shift(20))],['上×上向き','下×下向き'],'中間')
    R['MA並び(25/75/200)']=np.select([(s25>s75)&(s75>s200),(s25<s75)&(s75<s200)],['上昇順','下降順'],'その他')
    R['26週線(130日近似)']=np.where(c>c.rolling(130).mean(),'上','下')
    for n in (60,250):
        dd=(c/h.rolling(n).max()-1)*100; R[f'{n}日高値からの下落率']=pd.cut(dd,[-99,-20,-10,-5,-2,1],labels=['-20%超','-20〜-10%','-10〜-5%','-5〜-2%','-2%以内']).astype(str)
    for n in (5,20,60):
        r=(c/c.shift(n)-1)*100; b=[-99,-10,-5,0,5,10,99] if n>5 else [-99,-5,-2,0,2,5,99]
        R[f'{n}日騰落率']=pd.cut(r,b).astype(str)
    ten,kij,ct,cb=ichi(h,l,c)
    R['日足一目:雲']=np.select([c>ct,c<cb],['雲の上','雲の下'],'雲の中')
    R['日足一目:三役']=np.select([(c>ct)&(ten>kij)&(c>h.shift(26)),(c<cb)&(ten<kij)&(c<l.shift(26))],['三役好転','三役逆転'],'その他')
    R['日足:転換vs基準']=np.where(ten>kij,'転換>基準','転換<基準')
    adx,pdi,mdi=adx_(h,l,c); R['日足ADX×DI']=np.select([(adx>=25)&(pdi>mdi),(adx>=25)&(pdi<mdi)],['強トレンド上','強トレンド下'],'ADX25未満')
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); sg=m.ewm(span=9,adjust=False).mean()
    R['日足MACD']=np.select([(m>0)&(m>sg),(m>0)&(m<=sg),(m<=0)&(m>sg)],['0上・シグナル上','0上・シグナル下','0下・シグナル上'],'0下・シグナル下')
    R['日足RSI']=pd.cut(rsi_(c),[0,30,40,50,60,70,100]).astype(str)
    a=atr_(h,l,c); R['日足ATR水準(対250日平均)']=pd.cut(a/a.rolling(250).mean(),[0,0.8,1.2,1.6,99],labels=['低(-0.8)','中','高(1.2-1.6)','非常に高(1.6-)']).astype(str)
    R['日足ATR:拡大/縮小']=np.where(a>a.shift(10),'拡大中','縮小中')
    hh=h.rolling(20).max(); ll=l.rolling(20).min()
    R['波の形(20日高安)']=np.select([(hh>hh.shift(20))&(ll>ll.shift(20)),(hh<hh.shift(20))&(ll<ll.shift(20))],['切り上げ','切り下げ'],'混在')
    R['前日の足']=np.select([(c-o)>0.7*a,(c-o)<-0.7*a],['大陽線','大陰線'],'小')
    sd=c.rolling(20).std(ddof=0); m20=c.rolling(20).mean(); R['日足BB位置']=pd.cut((c-m20)/sd,[-9,-2,-1,0,1,2,9]).astype(str)
    for k,v in R.items(): d['R_'+k]=np.asarray(v)
    d['valid']=np.arange(len(d))>=260
    return d
if __name__=='__main__':
    d=daily(); c=d.close
    d['n1']=(c.shift(-1)/c-1)*100; d['n5']=(c.shift(-5)/c-1)*100
    yr=d.t.dt.year; d['P']=np.select([yr<=2016,yr<=2022],['08-16','17-22'],'23-26'); d=d[d.valid]
    print('基準(全日) 翌日平均%:',d.groupby('P').n1.mean().round(3).to_dict(),'日数',d.groupby('P').size().to_dict())
    rows=[]
    for col in [x for x in d.columns if x.startswith('R_')]:
        for b,g in d.groupby(col):
            if b=='nan': continue
            r=dict(定義=col[2:],区分=b)
            for p in ('08-16','17-22','23-26'):
                z=g[g.P==p]; base=d[d.P==p].n1.mean()
                r[p+' 日数']=len(z); r[p+' 翌日%']=round(z.n1.mean(),3) if len(z)>=20 else np.nan
                r[p+' t']=round((z.n1.mean()-base)/(z.n1.std()/np.sqrt(len(z))),1) if len(z)>=20 else np.nan
                r[p+' 5日%']=round(z.n5.mean(),2) if len(z)>=20 else np.nan
            rows.append(r)
    T=pd.DataFrame(rows); T.to_csv('stage1_regime.csv',index=False,encoding='utf-8-sig')
    pd.set_option('display.width',250,'display.max_rows',500,'display.unicode.east_asian_width',True)
    print(T.to_string(index=False))
