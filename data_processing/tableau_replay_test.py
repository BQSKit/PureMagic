#!/usr/bin/env python3
"""Checks tableau_replay.Tableau against dense 3-qubit matrices: random sequences of
S_P factors and S, SX, CX conjugations, compared as Paulis up to sign."""
import itertools, random, numpy as np, sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from tableau_replay import Tableau
I=np.eye(2); X=np.array([[0,1],[1,0]]); Z=np.diag([1,-1]); Y=1j*X@Z
P1={'I':I,'X':X,'Y':Y,'Z':Z}
n=3
def pauli(lab):  # lab: dict q->b
    m=np.array([[1]])
    for q in range(n): m=np.kron(m,P1[lab.get(q,'I')])
    return m
def S(lab): P=pauli(lab); return (np.eye(2**n)-1j*P)/np.sqrt(2)
CXm={}
def gate(g,qs):
    if g=='S': return S({qs[0]:'Z'})
    if g=='SX': return S({qs[0]:'X'})
    c,t=qs; return (np.eye(2**n)+pauli({c:'Z'})+pauli({t:'X'})-pauli({c:'Z',t:'X'}))/2
allp=[dict((q,b) for q,b in zip(range(n),bs) if b!='I') for bs in itertools.product('IXYZ',repeat=n)]
def label(M):
    for p in allp:
        if abs(abs(np.trace(pauli(p).conj().T@M))-2**n)<1e-6: return sorted(p.items())
    return None
rng=random.Random(1); bad=0
for trial in range(300):
    W=np.eye(2**n); t=Tableau(n)
    for step in range(12):
        r=rng.random()
        if r<0.4:
            L={q:rng.choice('XYZ') for q in rng.sample(range(n),rng.randint(1,n))}
            W=W@S(L); t.left_S(t.enc(L.items()))
        else:
            g=rng.choice(['S','SX','CX'])
            qs=tuple(rng.sample(range(n),2 if g=='CX' else 1)); C=gate(g,qs)
            if r<0.8: W=C@W@C.conj().T; t.clifford_push(g,qs)
            else: W=C@W; t.clifford_absorb(g,qs)
        for p in allp[1:]:
            want=label(W.conj().T@pauli(p)@W)
            got=t.dec(t.apply(t.enc(p.items())))
            if want!=got: bad+=1
print("dense mismatches:",bad)
sys.exit(1 if bad else 0)
