import pandas as pd, numpy as np
from scipy import stats
def ld(f):
    x=pd.read_csv(f,encoding='utf-8-sig'); e=x[x['タイプ']=='ロングエントリー'].set_index('トレード番号'); q=x[x['タイプ']=='ロング決済'].set_index('トレード番号')
    t=pd.DataFrame({'et':pd.to_datetime(e['日時']),'ep':e['価格 JPY'],'ret':e['リターン %'].astype(float),'pl':e['純損益 JPY'].astype(float),'mfe':e['最大順行幅 %'].astype(float),'mae':e['最大逆行幅 %'].astype(float),'cm':e['シグナル'],'xt':q['日時']})
    return t[t.xt!='未決済'].copy()
d=pd.read_csv('dd.csv'); d['t']=pd.to_datetime(d.time,unit='s',utc=True).dt.tz_convert('Asia/Tokyo')
d['date']=d.t.dt.tz_localize(None).dt.normalize().astype('datetime64[ns]')
d['s200']=d.close.rolling(200).mean(); d['up']=d.s200>d.s200.shift(20)
d['reg']=np.select([(d.close>d.s200)&d.up,(d.close<d.s200)&~d.up],['上昇','下落'],'中間')
tr=pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()],axis=1).max(axis=1); d['atrp']=tr.rolling(20).mean()/d.close*100
d['r60']=(d.close/d.close.shift(60)-1)*100; d['r20']=(d.close/d.close.shift(20)-1)*100; d['dev200']=(d.close/d.s200-1)*100
d['s50']=d.close.rolling(50).mean(); d['a50']=d.close>d.s50
FE=['reg','atrp','r60','r20','dev200','a50']
D=d[['date']+FE].copy(); D[FE]=D[FE].shift(2)
def st(z,col='ret'):
    if len(z)==0: return 'n=0'
    p=z[col][z[col]>0].sum(); n=-z[col][z[col]<0].sum()
    return f"n={len(z):4d} 勝{100*(z[col]>0).mean():3.0f}% PF{p/max(n,1e-9):5.2f} 平均{z[col].mean():+.3f}%"
def prep(f):
    t=ld(f); t['d']=t.et.dt.normalize().astype('datetime64[ns]'); m=pd.merge_asof(t.sort_values('d'),D.sort_values('date'),left_on='d',right_on='date'); m['yr']=m.et.dt.year; return m
if __name__=='__main__':
    ON=prep('on.csv'); OFF=prep('off.csv')
    z=ON[(ON.et>='2023-01-20')&(ON.et<='2026-10-07')]; print('照合 2023-01-20〜: n',len(z),'勝率%.0f%%'%((z.pl>0).mean()*100),'円PF %.2f'%(z.pl[z.pl>0].sum()/-z.pl[z.pl<0].sum()))
    print('全期間  ON:',st(ON),' OFF:',st(OFF))
    print('\n年別')
    for y in range(2017,2027):
        dy=d[d.date.dt.year==y]; print(y,' ON:',st(ON[ON.yr==y]),' | OFF:',st(OFF[OFF.yr==y]),' | 日経%+.0f%%'%(100*(dy.close.iloc[-1]/dy.close.iloc[0]-1)))
    print('\n相場環境別(200日線)')
    for r in ('上昇','中間','下落'): print(r,' ON:',st(ON[ON.reg==r]),' | OFF:',st(OFF[OFF.reg==r]))
    print('\n環境×期間')
    for r in ('上昇','中間','下落'):
        for a,b in ((2017,2022),(2023,2026)):
            print(r,a,b,' ON:',st(ON[(ON.reg==r)&ON.yr.between(a,b)]),' | OFF:',st(OFF[(OFF.reg==r)&OFF.yr.between(a,b)]))
    for l,a,b in (('全期間',2017,2026),('2017-22',2017,2022),('2023-26',2023,2026)):
        x=ON[ON.yr.between(a,b)].ret; y=OFF[OFF.yr.between(a,b)].ret; t1=stats.ttest_ind(x,y,equal_var=False); t2=stats.ttest_1samp(x,0)
        print(l,'ON平均%+.3f%% (t=%.2f)  OFF平均%+.3f%%  差t=%.2f'%(x.mean(),t2[0],y.mean(),t1[0]), ' ボラ調整平均 ON %.3f OFF %.3f'%((ON[ON.yr.between(a,b)].ret/ON[ON.yr.between(a,b)].atrp).mean(),(OFF[OFF.yr.between(a,b)].ret/OFF[OFF.yr.between(a,b)].atrp).mean()))
    print('\n他の環境指標でONを分割')
    for nm,bins in (('r60',[-99,-5,0,5,10,99]),('r20',[-99,-3,0,3,99]),('dev200',[-99,-5,0,5,10,99]),('atrp',[0,1.2,1.6,2.2,99])):
        for k,g in ON.groupby(pd.cut(ON[nm],bins),observed=True): print(' ',nm,k,st(g),' | OFF',st(OFF[pd.cut(OFF[nm],bins)==k]))
