import pickle, numpy as np, pandas as pd
d,X,E,cols=pickle.load(open('P_1時間.pkl','rb')); Xv=X.values; ci={c:i for i,c in enumerate(cols)}; yr=d.t.dt.year.values
def nonover(mask,col):
    c=ci[col]; r=Xv[:,c]; e=E[:,c]; free=0; out=[]
    for i in np.where(mask)[0]:
        if i<free or np.isnan(r[i]): continue
        out.append((yr[i],r[i])); free=i+int(e[i])+1
    return pd.DataFrame(out,columns=['yr','ret'])
def s(z):
    if len(z)<5: return 'n<5'
    return f"n{len(z):4d} 勝{100*(z.ret>0).mean():2.0f}% PF{z.ret[z.ret>0].sum()/max(-z.ret[z.ret<0].sum(),1e-9):.2f} 平均{z.ret.mean():+.3f}%"
def show(nm,mask,col):
    z=nonover(mask,col); ys=z.groupby('yr').ret.sum(); print(f"{nm:34s} {col:14s} 17-22: {s(z[z.yr<=2022])} | 23-26: {s(z[z.yr>=2023])} | プラス年{int((ys>0).sum())}/{len(ys)}")
vol=(d['F_出来高(粗)']=='1.5-').values; am=(d['F_時間帯']=='日中前半').values
body=(d.close-d.open).abs(); rng=(d.high-d.low)
G={k:(d[l]==k).values for l,k in (('g2','上髭2倍'),('g3','上髭3倍'),('g5','上髭5倍'),('g2','下髭2倍'),('g3','下髭3倍'))}
G['両髭/その他の小実体(実体<値幅の25%)']=((body<0.25*rng)&(d.g2=='')).values
G['大実体(実体>値幅の60%)陽線']=((body>0.6*rng)&(d.close>d.open)).values
G['大実体陰線']=((body>0.6*rng)&(d.close<d.open)).values
G['全足']=np.ones(len(d),bool)
print('■出来高1.5倍以上の足'); 
for k,m in G.items(): show(k,m&vol,'成行|買|T40')
print('■日中前半の足')
for k,m in G.items(): show(k,m&am,'成行|買|T40')
print('■保有本数・方向（上髭2倍×出来高1.5倍以上）')
for c in ('成行|買|T5','成行|買|T10','成行|買|T20','成行|買|T40','成行|売|T40','成行|買|トレール3ATR','指値|買|T40','確認|買|T40'): show('上髭2倍&出来高',G['上髭2倍']&vol,c)
print('■出来高条件なし'); 
for k in ('上髭2倍','上髭3倍','下髭2倍','下髭3倍'): show(k,G[k],'成行|買|T40')
m=G['上髭2倍']&vol; print('シグナル足の開始時刻分布:',d.t[m].dt.strftime('%H:%M').value_counts().head(8).to_dict())
print('出来高1.5倍以上の全足の時刻分布:',d.t[vol].dt.strftime('%H:%M').value_counts().head(6).to_dict())
print('■持ち越し仮説: 日足200日線+10%超での下髭買い(T40)')
hi=(d['F_日足200日線乖離']=='10%超').values
for k in ('下髭2倍','下髭3倍','上髭2倍','全足'): show(k+' & 乖離10%超',G[k]&hi,'成行|買|T40')
for k in ('下髭3倍','全足'): show(k+' & 乖離10%以下',G[k]&~hi,'成行|買|T40')
print('■日中前半×上髭2倍 の年別(%)'); z=nonover(G['上髭2倍']&am,'成行|買|T40'); print(z.groupby('yr').ret.agg(['count','sum','mean']).round(2).T.to_string())
print('■時刻別 上髭2倍 買T40'); 
for hh in ('08:30','09:30','10:30','12:30','14:30'): show('開始'+hh,G['上髭2倍']&(d.t.dt.strftime('%H:%M')==hh).values,'成行|買|T40'); show('  全足 '+hh,(d.t.dt.strftime('%H:%M')==hh).values,'成行|買|T40')
