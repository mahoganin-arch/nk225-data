import pandas as pd, re
def bars(f):
    t=pd.read_csv(f); rows=[]
    for s in t['シグナル'].dropna().unique():
        if '~' not in str(s): continue
        body=s.split('~',1)[1]
        for p in body.split(';'):
            q=p.split(',')
            if len(q)==6:
                try: rows.append([float(x) for x in q])
                except: pass
    d=pd.DataFrame(rows,columns=['m','open','high','low','close','volume']).drop_duplicates('m').sort_values('m').reset_index(drop=True)
    d['t']=pd.to_datetime(d.m*60,unit='s',utc=True).dt.tz_convert('Asia/Tokyo'); return d
if __name__=='__main__':
    for f in ('h1.csv','m30.csv'):
        d=bars(f); print(f,len(d),d.t.iloc[0],d.t.iloc[-1]); print(d.t.dt.strftime('%H:%M').value_counts().sort_index().to_dict())
        print(d.groupby(d.t.dt.year).size().to_dict())
    x=pd.read_csv('d1.csv'); x['t']=pd.to_datetime(x.time,unit='s',utc=True).dt.tz_convert('Asia/Tokyo'); print(len(x),x.t.iloc[0],x.t.iloc[-1])
