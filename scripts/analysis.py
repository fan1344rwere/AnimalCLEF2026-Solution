import pandas as pd
import numpy as np

meta = pd.read_csv('./data/metadata.csv')
te = meta[meta.split=='test']
tr = meta[meta.split=='train']

print("=" * 60)
print("DEEP DATA ANALYSIS")
print("=" * 60)

for d in ['LynxID2025','SalamanderID2025','SeaTurtleID2022','TexasHornedLizards']:
    t = tr[tr.dataset==d]
    e = te[te.dataset==d]
    print(f"\n{'─'*50}")
    print(f"{d}")
    print(f"{'─'*50}")
    if len(t)>0:
        vc = t.identity.value_counts()
        print(f"  TRAIN: {len(t)} imgs, {t.identity.nunique()} identities")
        print(f"  imgs/id: min={vc.min()} med={vc.median():.0f} max={vc.max()} mean={vc.mean():.1f}")
        print(f"  ids with 1 img: {(vc==1).sum()}")
        print(f"  ids with 2 img: {(vc==2).sum()}")
        print(f"  ids with 3-5 img: {((vc>=3)&(vc<=5)).sum()}")
        print(f"  ids with 6+ img: {(vc>=6).sum()}")
    if 'orientation' in meta.columns:
        to = t.orientation.value_counts() if len(t)>0 else {}
        eo = e.orientation.value_counts() if len(e)>0 else {}
        print(f"  TRAIN orientations: {dict(to)}")
        print(f"  TEST  orientations: {dict(eo)}")
    print(f"  TEST:  {len(e)} imgs")

print("\n" + "=" * 60)
print("SUBMISSION CLUSTER ANALYSIS")
print("=" * 60)

for v in ['ov14','ov16','ov18']:
    try:
        s = pd.read_csv(f'./{v}/submission.csv')
        print(f"\n{v}:")
        for d in ['LynxID2025','SalamanderID2025','SeaTurtleID2022','TexasHornedLizards']:
            sp = s[s.cluster.str.contains(d)]
            vc = sp.cluster.value_counts()
            singletons = (vc==1).sum()
            print(f"  {d}: {len(sp)} imgs → {sp.cluster.nunique()} cl | "
                  f"sizes: min={vc.min()} med={vc.median():.0f} max={vc.max()} | "
                  f"singletons={singletons} ({100*singletons/len(vc):.0f}%)")
    except Exception as ex:
        print(f"  {v}: {ex}")
