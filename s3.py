import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from ld import bars
from s1 import rma,rsi_,atr_,adx_,ichi
def cut(x,b,fmt='{:g}'):
    return pd.cut(x,b).astype(str)
def regime(o,h,l,c):
    R={}; a=atr_(h,l,c); S={n:c.rolling(n).mean() for n in (5,10,20,25,50,75,100,200)}
    for n,s in S.items(): R[f'MA位置|終値vsSMA{n}']=np.where(c>s,'上','下')
    for n in (20,50,100,200): R[f'MA傾き|SMA{n}の傾き']=np.where(S[n]>S[n].shift(max(3,n//10)),'上向き','下向き')
    for n in (50,100,200): R[f'MA位置×傾き|SMA{n}']=np.select([(c>S[n])&(S[n]>S[n].shift(n//10)),(c<S[n])&(S[n]<S[n].shift(n//10))],['上×上向き','下×下向き'],'中間')
    for x,y,z in ((5,20,50),(20,50,100),(25,75,200),(50,100,200)):
        R[f'MA並び|{x}/{y}/{z}']=np.select([(S[x]>S[y])&(S[y]>S[z]),(S[x]<S[y])&(S[y]<S[z])],['上昇順','下降順'],'その他')
    for x,y in ((5,20),(20,50),(50,200)): R[f'MAクロス状態|SMA{x}vs{y}']=np.where(S[x]>S[y],'短期が上','短期が下')
    for n,b in ((20,[-99,-3,-1.5,0,1.5,3,99]),(50,[-99,-5,-2.5,0,2.5,5,99]),(100,[-99,-7,-3.5,0,3.5,7,99]),(200,[-99,-10,-5,0,5,10,99])):
        R[f'MA乖離(ATR倍)|SMA{n}']=cut((c-S[n])/a,b)
    for n in (60,120,250):
        HH=h.rolling(n).max(); LL=l.rolling(n).min()
        ih=h.rolling(n).apply(np.argmax,raw=True); il=l.rolling(n).apply(np.argmin,raw=True); upleg=ih>il
        retr=pd.Series(np.where(upleg,(HH-c)/(HH-LL),(c-LL)/(HH-LL)),index=c.index)
        lab=pd.cut(retr,[-0.001,0.236,0.382,0.5,0.618,0.786,1.001],labels=['0-23.6%','23.6-38.2%','38.2-50%','50-61.8%','61.8-78.6%','78.6-100%']).astype(str)
        R[f'フィボナッチ|{n}本の波']=np.where(lab=='nan','nan',np.where(upleg,'上昇後の押し ','下落後の戻り ')+lab)
        R[f'高値からの距離(ATR倍)|{n}本高値']=cut((HH-c)/a,[-1,1,3,6,10,999])
        R[f'安値からの距離(ATR倍)|{n}本安値']=cut((c-LL)/a,[-1,1,3,6,10,999])
    for n,b in ((5,[-99,-3,-1.5,0,1.5,3,99]),(20,[-99,-6,-3,0,3,6,99]),(60,[-99,-10,-5,0,5,10,99])): R[f'騰落(ATR倍)|{n}本']=cut((c-c.shift(n))/a,b)
    ten,kij,ct,cb=ichi(h,l,c); ct=pd.Series(ct,index=c.index); cb=pd.Series(cb,index=c.index)
    R['一目|雲']=np.select([c>ct,c<cb],['雲の上','雲の下'],'雲の中')
    R['一目|三役']=np.select([(c>ct)&(ten>kij)&(c>h.shift(26)),(c<cb)&(ten<kij)&(c<l.shift(26))],['三役好転','三役逆転'],'その他')
    R['一目|転換vs基準']=np.where(ten>kij,'転換>基準','転換<基準')
    R['一目|基準線の向き']=np.select([kij>kij.shift(5),kij<kij.shift(5)],['上向き','下向き'],'横ばい')
    R['一目|遅行線vs26本前のローソク足']=np.select([c>h.shift(25),c<l.shift(25)],['上','下'],'足の中')
    R['一目|遅行線vs26本前の雲']=np.select([c>ct.shift(25),c<cb.shift(25)],['雲の上','雲の下'],'雲の中')
    sa=(ten+kij)/2; sb=(h.rolling(52).max()+l.rolling(52).min())/2; R['一目|先行する雲の色']=np.where(sa>sb,'陽転(A>B)','陰転(A<B)')
    R['一目|雲の厚さ(ATR倍)']=cut((ct-cb)/a,[-1,1,3,999])
    R['一目|基準線乖離(ATR倍)']=cut((c-kij)/a,[-99,-3,-1.5,0,1.5,3,99])
    adx,pdi,mdi=adx_(h,l,c); R['ADX×DI|']=np.select([(adx>=25)&(pdi>mdi),(adx>=25)&(pdi<mdi),(pdi>mdi)],['強トレンド上','強トレンド下','弱・DI上'],'弱・DI下')
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); sg=m.ewm(span=9,adjust=False).mean()
    R['MACD|']=np.select([(m>0)&(m>sg),(m>0)&(m<=sg),(m<=0)&(m>sg)],['0上・シグナル上','0上・シグナル下','0下・シグナル上'],'0下・シグナル下')
    R['RSI|']=cut(rsi_(c),[0,30,40,50,60,70,100])
    R['ATR水準|対250本平均']=cut(a/a.rolling(250).mean(),[0,0.8,1.2,1.6,99]); R['ATR|拡大/縮小']=np.where(a>a.shift(10),'拡大中','縮小中')
    hh=h.rolling(20).max(); ll=l.rolling(20).min()
    R['波の形|20本高安']=np.select([(hh>hh.shift(20))&(ll>ll.shift(20)),(hh<hh.shift(20))&(ll<ll.shift(20))],['切り上げ','切り下げ'],'混在')
    sd=c.rolling(20).std(ddof=0); R['BB位置|']=cut((c-S[20])/sd,[-9,-2,-1,0,1,2,9])
    R['BB幅|対100本平均']=cut((sd/sd.rolling(100).mean()),[0,0.7,1.3,99])
    # スーパートレンド(10,3)
    a10=atr_(h,l,c,10).values; hl2=((h+l)/2).values; cv=c.values; n=len(c); ub=hl2+3*a10; lb=hl2-3*a10; dr=np.ones(n)
    for i in range(1,n):
        if not np.isnan(ub[i-1]):
            if not (ub[i]<ub[i-1] or cv[i-1]>ub[i-1]): ub[i]=ub[i-1]
            if not (lb[i]>lb[i-1] or cv[i-1]<lb[i-1]): lb[i]=lb[i-1]
        dr[i]=1 if cv[i]>ub[i-1] else (-1 if cv[i]<lb[i-1] else dr[i-1])
    R['スーパートレンド|(10,3)']=np.where(dr>0,'上昇','下降')
    hc=(o+h+l+c)/4; ho=hc.ewm(alpha=0.5,adjust=False).mean().shift(); R['平均足|色']=np.where(hc>ho,'陽','陰')
    st=(hc>ho); R['平均足|同色の連続']=np.where(st.rolling(5).sum()==5,'陽5連続以上',np.where(st.rolling(5).sum()==0,'陰5連続以上','その他'))
    R['直前の足|大きさ']=np.select([(c-o)>0.7*a,(c-o)<-0.7*a],['大陽線','大陰線'],'小')
    return pd.DataFrame({k:np.asarray(v) for k,v in R.items()},index=c.index)
def table(lab,fr,P,plist,Heff,tf,minn=30):
    rows=[]
    for col in lab.columns:
        x=lab[col].values
        for b in pd.unique(x):
            if b=='nan': continue
            r=dict(足=tf,分類=col.split('|')[0],定義=col.split('|')[1],区分=b); mk=(x==b)
            for p in plist:
                sel=mk&(P==p)&~np.isnan(fr); base=np.nanmean(fr[(P==p)]); z=fr[sel]; ne=len(z)/Heff
                r[p+' 割合%']=round(100*sel.sum()/max((P==p).sum(),1),1)
                ok=ne>=minn; r[p+' 平均%']=round(z.mean(),3) if ok else np.nan
                r[p+' t']=round((z.mean()-base)/(z.std()/np.sqrt(ne)),1) if ok else np.nan
            rows.append(r)
    return pd.DataFrame(rows)
if __name__=='__main__':
    out=[]
    # 日足（翌日の騰落、2008年〜）
    d=pd.read_csv('d1.csv'); d.columns=[x.lower() for x in d.columns]; d['t']=pd.to_datetime(d.time,unit='s',utc=True).dt.tz_convert('Asia/Tokyo'); d=d.sort_values('t').reset_index(drop=True)
    LD=regime(d.open,d.high,d.low,d.close); fr=((d.close.shift(-1)/d.close-1)*100).values.copy(); fr[:260]=np.nan
    yr=d.t.dt.year.values; P=np.select([yr<=2016,yr<=2022],['08-16','17-22'],'23-26')
    TD=table(LD,fr,P,['08-16','17-22','23-26'],1,'日足'); TD.to_pickle('TD.pkl')
    # 4時間・1時間（1時間足で、次足始値→20本後終値）
    b=bars('h1.csv'); n=len(b); H=20; c=b.close.values; o=b.open.values
    fr=np.full(n,np.nan); fr[:n-H]=(c[H:]-o[1:n-H+1])/c[:n-H]*100; fr[:300]=np.nan
    P=np.where(b.t.dt.year.values<=2022,'17-22','23-26')
    L1=regime(b.open,b.high,b.low,b.close); T1=table(L1,fr,P,['17-22','23-26'],H,'1時間')
    x=b.set_index('t'); h4=x.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    L4=regime(h4.open,h4.high,h4.low,h4.close).shift(1).fillna('nan'); L4b=L4.reindex(x.index,method='ffill').reset_index(drop=True); fr4=fr.copy(); fr4[:1200]=np.nan
    T4=table(L4b,fr4,P,['17-22','23-26'],H,'4時間')
    # 日足の局面を1時間足に対応（確定済みの前日足）
    k=np.searchsorted(d.t.values,b.t.values,side='right')-1; LDb=LD.iloc[np.clip(k-1,0,None)].reset_index(drop=True)
    X={}
    X['複合|日足SMA75 × 4時間SMA50']=('日足'+LDb['MA位置|終値vsSMA75']+'・4H'+L4b['MA位置|終値vsSMA50']).values
    X['複合|日足SMA200 × 4時間SMA50']=('日足'+LDb['MA位置|終値vsSMA200']+'・4H'+L4b['MA位置|終値vsSMA50']).values
    X['複合|日足雲 × 4時間雲']=('日足'+LDb['一目|雲']+'・4H'+L4b['一目|雲']).values
    X['複合|4時間雲 × 1時間雲']=('4H'+L4b['一目|雲']+'・1H'+L1['一目|雲']).values
    X['複合|4時間SMA50 × 1時間SMA50']=('4H'+L4b['MA位置|終値vsSMA50']+'・1H'+L1['MA位置|終値vsSMA50']).values
    X['複合|日足SMA75 × 4時間SMA50 × 1時間SMA50']=('日足'+LDb['MA位置|終値vsSMA75']+'・4H'+L4b['MA位置|終値vsSMA50']+'・1H'+L1['MA位置|終値vsSMA50']).values
    X['複合|日足三役 × 1時間三役']=('日足'+LDb['一目|三役']+'・1H'+L1['一目|三役']).values
    LX=pd.DataFrame(X); LX=LX.where(~LX.apply(lambda s:s.str.contains('nan')),'nan'); TX=table(LX,fr4,P,['17-22','23-26'],H,'複合')
    TI=pd.concat([T1,T4,TX]); TI.to_pickle('TI.pkl')
    TD.to_csv('/mnt/user-data/outputs/stage1_1b_局面一覧_日足_拡張.csv',index=False,encoding='utf-8-sig')
    TI.to_csv('/mnt/user-data/outputs/stage1_1b_局面一覧_4時間_1時間_複合.csv',index=False,encoding='utf-8-sig')
    print(len(LD.columns),'定義/足', len(TD),len(T1),len(T4),len(TX))
