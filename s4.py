import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from ld import bars
from s1 import atr_
from s2 import nonover
from s3 import table
D=pd.read_csv('d1.csv'); D.columns=[x.lower() for x in D.columns]; D['t']=pd.to_datetime(D.time,unit='s',utc=True).dt.tz_convert('Asia/Tokyo'); D=D.sort_values('t').reset_index(drop=True)
td=D.t.dt.tz_localize(None).dt.normalize()+pd.to_timedelta((D.t.dt.hour>=16).astype(int),unit='D'); td=td+pd.to_timedelta((td.dt.dayofweek==5)*2+(td.dt.dayofweek==6)*1,unit='D')
D['wk']=td.dt.strftime('%G-%V'); W=D.groupby('wk',sort=False).agg(high=('high','max'),low=('low','min'),close=('close','last')); Wp=W.shift(1); D['wH']=D.wk.map(Wp.high); D['wL']=D.wk.map(Wp.low); D['wC']=D.wk.map(Wp.close)
def piv(H,L,C):
    P=(H+L+C)/3; return dict(P=P,R1=2*P-L,S1=2*P-H,R2=P+(H-L),S2=P-(H-L))
def profile(h,l,v,N,bw=0.002):
    n=len(h); lo=np.floor(np.log(l)/bw).astype(int); hi=np.floor(np.log(h)/bw).astype(int); off=lo.min()-2; lo-=off; hi-=off; M=hi.max()+3
    hist=np.zeros(M); poc=np.full(n,np.nan); vah=np.full(n,np.nan); val=np.full(n,np.nan); dens=np.full(n,np.nan); cen=lambda k:np.exp((k+off+0.5)*bw)
    for i in range(n):
        hist[lo[i]:hi[i]+1]+=v[i]/(hi[i]-lo[i]+1)
        if i>=N: j=i-N; hist[lo[j]:hi[j]+1]-=v[j]/(hi[j]-lo[j]+1)
        if i>=N-1:
            p=int(hist.argmax()); tot=hist.sum(); acc=hist[p]; a=b=p
            while acc<0.7*tot:
                ua=hist[b+1] if b+1<M else -1; da=hist[a-1] if a>0 else -1
                if ua>=da: b+=1; acc+=ua
                else: a-=1; acc+=da
            poc[i]=cen(p); vah[i]=np.exp((b+off+1)*bw); val[i]=np.exp((a+off)*bw); dens[i]=hist[hi[i]]/hist[p]  # 近似：高値側ビン
    return poc,vah,val,dens
def touch(E,name,o,h,l,c,lv):
    lv=pd.Series(lv,index=c.index); pc=c.shift(); 
    E[f'{name}に上から触れて反発']=(l<=lv)&(c>lv)&(pc>lv); E[f'{name}に下から触れて反落']=(h>=lv)&(c<lv)&(pc<lv)
    E[f'{name}を上抜け']=(c>lv)&(pc<=lv); E[f'{name}を下抜け']=(c<lv)&(pc>=lv)
def build(f,tfmin):
    b=bars(f); o,h,l,c,v=b.open,b.high,b.low,b.close,b.volume; a=atr_(h,l,c); S={}; E={}
    k=np.searchsorted(D.t.values,b.t.values,side='right')-1; kp=np.clip(k-1,0,None)
    dp=piv(D.high.values[kp],D.low.values[kp],D.close.values[kp]); wp=piv(D.wH.values[np.clip(k,0,None)],D.wL.values[np.clip(k,0,None)],D.wC.values[np.clip(k,0,None)])
    for nm,p in (('日次ピボット',dp),('週次ピボット',wp)):
        S[f'ピボット|{nm}のどの帯か']=np.select([c<p['S2'],c<p['S1'],c<p['P'],c<p['R1'],c<p['R2']],['S2より下','S2〜S1','S1〜P','P〜R1','R1〜R2'],'R2より上'); S[f'ピボット|{nm}のどの帯か']=np.where(np.isnan(p['P']),'nan',S[f'ピボット|{nm}のどの帯か'])
        for lvn in ('P','R1','S1','R2','S2'): touch(E,f'ピボット|{nm}{lvn}',o,h,l,c,p[lvn])
    pdH,pdL,pdC=D.high.values[kp],D.low.values[kp],D.close.values[kp]
    S['ピボット|前日終値との位置']=np.where(c>pdC,'前日終値より上','前日終値より下'); touch(E,'ピボット|前日終値',o,h,l,c,pdC)
    for N,lab in ((120*60//tfmin,'約1週間'),(500*60//tfmin,'約1か月')):
        poc,vah,val,dens=profile(h.values,l.values,v.values,N)
        S[f'価格帯別出来高|{lab}:バリューエリア']=np.where(np.isnan(poc),'nan',np.select([c>vah,c<val],['上限より上','下限より下'],'エリア内'))
        S[f'価格帯別出来高|{lab}:POCとの位置']=np.where(np.isnan(poc),'nan',pd.cut((c-poc)/a,[-999,-6,-3,-1,1,3,6,999]).astype(str))
        S[f'価格帯別出来高|{lab}:今の価格帯の厚み']=np.where(np.isnan(dens),'nan',pd.cut(pd.Series(dens),[-1,0.2,0.5,0.8,1.01],labels=['薄い(〜20%)','やや薄い','やや厚い','厚い(80%〜)']).astype(str))
        touch(E,f'価格帯別出来高|{lab}POC',o,h,l,c,poc); touch(E,f'価格帯別出来高|{lab}エリア上限',o,h,l,c,vah); touch(E,f'価格帯別出来高|{lab}エリア下限',o,h,l,c,val)
    for step in (1000,500):
        r=c%step; S[f'切りの良い価格|{step}円刻みの中の位置']=pd.cut(r/step,[-0.001,0.1,0.4,0.6,0.9,1.0],labels=['直上(0-10%)','下寄り','中央','上寄り','直下(90-100%)']).astype(str)
        up=np.ceil(c.shift()/step)*step; dn=np.floor(c.shift()/step)*step; pc=c.shift()
        E[f'切りの良い価格|{step}円の節目を上抜け']=(c>up)&(pc<up); E[f'切りの良い価格|{step}円の節目を下抜け']=(c<dn)&(pc>dn)
        E[f'切りの良い価格|{step}円の節目に下から触れて反落']=(h>=up)&(c<up)&(pc<up); E[f'切りの良い価格|{step}円の節目に上から触れて反発']=(l<=dn)&(c>dn)&(pc>dn)
    return b,pd.DataFrame({k:np.asarray(x) for k,x in S.items()}),{k:np.asarray(pd.Series(np.asarray(x)).fillna(False),bool) for k,x in E.items()}
ST=[];EV=[]
for f,tf,nm in (('h1.csv',60,'1時間'),('m30.csv',30,'30分')):
    b,S,E=build(f,tf); n=len(b); c=b.close.values; o=b.open.values; yr=b.t.dt.year.values; warm=600*60//tf
    Hs=20*60//tf; fr=np.full(n,np.nan); fr[:n-Hs]=(c[Hs:]-o[1:n-Hs+1])/c[:n-Hs]*100; fr[:warm]=np.nan
    ST.append(table(S,fr,np.where(yr<=2022,'17-22','23-26'),['17-22','23-26'],Hs,nm))
    P=np.where(yr<=2022,0,1); ok=np.arange(n)>=warm
    for H in (5,20,40):
        fr=np.full(n,np.nan); fr[:n-H]=(c[H:]-o[1:n-H+1])/c[:n-H]*100; base=[np.nanmean(fr[ok&(P==p)]) for p in (0,1)]
        for k,mk in E.items():
            idx=nonover(np.where(mk&ok&~np.isnan(fr))[0],H); r=dict(足=nm,H=H,分類=k.split('|')[0],形=k.split('|')[1])
            for p,lab in ((0,'17-22'),(1,'23-26')):
                z=fr[idx[P[idx]==p]]; nn=len(z); g=nn>=30
                r[lab+' n']=nn; r[lab+' 平均%']=round(z.mean(),3) if g else np.nan; r[lab+' 超過%']=round(z.mean()-base[p],3) if g else np.nan; r[lab+' t']=round((z.mean()-base[p])/(z.std()/np.sqrt(nn)),1) if g else np.nan
            EV.append(r)
ST=pd.concat(ST); EV=pd.DataFrame(EV)
ST.to_csv('/mnt/user-data/outputs/stage1_1c_ピボット_出来高_節目_位置.csv',index=False,encoding='utf-8-sig'); EV.to_csv('/mnt/user-data/outputs/stage1_2c_ピボット_出来高_節目_反応.csv',index=False,encoding='utf-8-sig')
pd.set_option('display.width',260,'display.max_rows',500,'display.unicode.east_asian_width',True)
V=ST.dropna(subset=['17-22 t','23-26 t']); V=V.assign(z=(V['17-22 t']+V['23-26 t'])/np.sqrt(2))
print('位置: 区分',len(V),' 17-22|t|>=2:',int((V['17-22 t'].abs()>=2).sum()),' 23-26:',int((V['23-26 t'].abs()>=2).sum()),' 期待',round(len(V)*.0455,1),' |z|>=2:',int((V.z.abs()>=2).sum()))
print(V[(V.z.abs()>=1.8)|((V['17-22 平均%']<0)&(V['23-26 平均%']<0))].sort_values('z').to_string(index=False))
V=EV.dropna(subset=['17-22 t','23-26 t']); V=V.assign(z=(V['17-22 t']+V['23-26 t'])/np.sqrt(2))
print('反応: 検定',len(V),' 17-22|t|>=2:',int((V['17-22 t'].abs()>=2).sum()),' 23-26:',int((V['23-26 t'].abs()>=2).sum()),' 期待',round(len(V)*.0455,1),' |z|>=2:',int((V.z.abs()>=2).sum()),' |z|>=3:',int((V.z.abs()>=3).sum()),'期待',round(len(V)*.0027,1))
print(V[V.z.abs()>=2].sort_values('z').to_string(index=False))
print('両期間マイナス(H>=20):'); print(V[(V.H>=20)&(V['17-22 平均%']<0)&(V['23-26 平均%']<0)].sort_values('z').to_string(index=False))
