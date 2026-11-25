import math, random, wave, struct, pathlib
R=32000
N=round(R*120/81)
out=pathlib.Path(__file__).parent/'app/src/main/assets'
out.mkdir(parents=True,exist_ok=True)
rng=random.Random(81)
def drum(t,kind):
    if kind==0:
        return .85*math.sin(2*math.pi*(52*t+3*(1-math.exp(-28*t))))*math.exp(-13*t)+.15*rng.uniform(-1,1)*math.exp(-65*t)
    if kind==1:
        return .38*math.sin(2*math.pi*178*t)*math.exp(-24*t)+.42*rng.uniform(-1,1)*math.exp(-35*t)
    return .34*(rng.uniform(-1,1)-rng.uniform(-1,1))*math.exp(-55*t)
def stab(t):
    a=min(1,t/.008)*math.exp(-18*t)
    return a*sum((math.sin(2*math.pi*f*t)+.22*math.sin(4*math.pi*f*t)) for f in (146.832,174.614,220.0,293.665))*.16
patterns=[[(0,0,1),(3,1,.8),(4,0,.85),(6,1,.8)],[(2,3,.9),(6,3,1)],[(0,2,.5),(1,2,.36),(2,2,.65),(3,2,.28),(4,2,.48),(5,2,.35),(6,2,.7),(7,2,.4)],[(0,0,1),(2,1,.65),(4,1,.8),(5,1,.7),(6,1,.9),(7,0,1)]]
for i,events in enumerate(patterns):
    a=[0.0]*N
    for step,k,v in events:
        start=round(N*step/8)
        for j in range(min(N-start,int(R*.28))):
            t=j/R
            a[start+j]+=v*(stab(t) if k==3 else drum(t,k))
    peak=max(.01,max(abs(x) for x in a))
    with wave.open(str(out/('pad%d.wav'%(i+1))),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(R)
        w.writeframes(b''.join(struct.pack('<h',int(x/peak*22000)) for x in a))
