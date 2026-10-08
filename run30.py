import sys, pickle, itertools, numpy as np, pandas as pd, sim3
part=int(sys.argv[1]); NP=int(sys.argv[2])
d,X,E,cols=pickle.load(open('P_30分.pkl','rb')); X=X.astype('float64')
orig=itertools.combinations
# 条件リストを分割して実行
def run_part():
    import types
    src=sim3.run
    feats=[c for c in d.columns if c.startswith('F_')]
    conds=[('なし',None)]+[(f[2:],[f]) for f in feats]+[(a+' & '+b,['F_'+a,'F_'+b]) for a,b in itertools.combinations([x for x in sim3.CORE if 'F_'+x in feats],2)]
    return conds[part::NP]
conds=run_part()
# sim3.run を条件リスト指定で呼べるように差し替え
import inspect
code=inspect.getsource(sim3.run).replace("def run(tf,d,X,cols):","def run(tf,d,X,cols,conds_):").replace("    for name,fs in conds:","    conds=conds_\n    for name,fs in conds:")
ns={}; exec(code,sim3.__dict__,ns)
K,tot,totok=ns['run']('30分',d,X,cols,conds)
pickle.dump((K,tot,totok),open(f'KL30_{part}.pkl','wb')); print(part,len(conds),tot,totok,len(K))
