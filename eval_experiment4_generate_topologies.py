#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--levels", type=float, nargs="+", default=[0.10, 0.30, 0.50])
parser.add_argument("--graphs-per-level", type=int, default=20)
parser.add_argument("--base-seed", type=int, default=241000)
parser.add_argument("--tolerance", type=float, default=0.01)
parser.add_argument("--max-proposals", type=int, default=1_000_000)
parser.add_argument("--outdir", type=str, default="results/experiment4")
args = parser.parse_args()

N_COMMUNITIES = 5
COMMUNITY_SIZE = 20
N = 100
DEGREE = 4
COMMUNITY = np.repeat(np.arange(N_COMMUNITIES), COMMUNITY_SIZE)

def add_edge(A,i,j):
    A[i,j]=A[j,i]=1.0

def remove_edge(A,i,j):
    A[i,j]=A[j,i]=0.0

def build_nominal_v4():
    A=np.zeros((N,N),dtype=float)
    for c in range(N_COMMUNITIES):
        base=c*COMMUNITY_SIZE
        for li in range(COMMUNITY_SIZE):
            i=base+li
            for off in (1,2):
                j=base+(li+off)%COMMUNITY_SIZE
                add_edge(A,i,j)
    for c in range(N_COMMUNITIES):
        d=(c+1)%N_COMMUNITIES
        u1,u2=c*COMMUNITY_SIZE+4,c*COMMUNITY_SIZE+5
        v1,v2=d*COMMUNITY_SIZE+12,d*COMMUNITY_SIZE+13
        remove_edge(A,u1,u2); remove_edge(A,v1,v2)
        add_edge(A,u1,v1); add_edge(A,u2,v2)
    return A

A_NOMINAL=build_nominal_v4()

def adjacency_to_edges(A):
    ii,jj=np.where(np.triu(A,1)>0.5)
    return {tuple(sorted((int(i),int(j)))) for i,j in zip(ii,jj)}

NOMINAL_EDGES=adjacency_to_edges(A_NOMINAL)
N_EDGES=len(NOMINAL_EDGES)

def canonical(i,j): return (i,j) if i<j else (j,i)

def mixing_type(e):
    i,j=e
    return tuple(sorted((int(COMMUNITY[i]),int(COMMUNITY[j]))))

def mixing_counts(edges):
    d={}
    for e in edges:
        k=mixing_type(e); d[k]=d.get(k,0)+1
    return d

NOMINAL_MIXING=mixing_counts(NOMINAL_EDGES)

def edges_to_A(edges):
    A=np.zeros((N,N),dtype=float)
    for i,j in edges: A[i,j]=A[j,i]=1.0
    return A

def degree_vector(edges):
    d=np.zeros(N,dtype=int)
    for i,j in edges: d[i]+=1; d[j]+=1
    return d

def connected(edges):
    nb=[[] for _ in range(N)]
    for i,j in edges: nb[i].append(j); nb[j].append(i)
    seen={0}; stack=[0]
    while stack:
        i=stack.pop()
        for j in nb[i]:
            if j not in seen:
                seen.add(j); stack.append(j)
    return len(seen)==N

def q_rewire(edges):
    return 1.0-len(edges & NOMINAL_EDGES)/N_EDGES

def lambda2(A):
    L=np.diag(A.sum(1))-A
    vals=np.linalg.eigvalsh(L)
    return float(np.sort(np.real(vals))[1])

def mean_clustering(A):
    vals=[]
    for i in range(N):
        nbr=np.where(A[i]>0.5)[0]
        k=len(nbr)
        if k<2: vals.append(0.0); continue
        sub=A[np.ix_(nbr,nbr)]
        vals.append(float(sub.sum()/(k*(k-1))))
    return float(np.mean(vals))

def modularity(A):
    k=A.sum(1); m2=float(A.sum()); Q=0.0
    for c in range(N_COMMUNITIES):
        idx=np.where(COMMUNITY==c)[0]
        Q += A[np.ix_(idx,idx)].sum() - np.outer(k[idx],k[idx]).sum()/m2
    return float(Q/m2)

def bridge_nodes(edges):
    s=set()
    for i,j in edges:
        if COMMUNITY[i]!=COMMUNITY[j]:
            s.add(i); s.add(j)
    return sorted(s)

def candidate_pairings(e1,e2):
    a,b=e1; c,d=e2
    return [(canonical(a,c),canonical(b,d)),(canonical(a,d),canonical(b,c))]

def valid_pair(n1,n2,o1,o2,edges):
    if n1[0]==n1[1] or n2[0]==n2[1] or n1==n2: return False
    if {n1,n2}=={o1,o2}: return False
    rem=edges-{o1,o2}
    return n1 not in rem and n2 not in rem

def preserves_mixing(o1,o2,n1,n2):
    return sorted([mixing_type(o1),mixing_type(o2)])==sorted([mixing_type(n1),mixing_type(n2)])

def generate(target_q, seed, tol, max_props):
    rng=np.random.default_rng(seed)
    edges=set(NOMINAL_EDGES)
    best=set(edges); best_err=abs(target_q)
    for proposal in range(1,max_props+1):
        el=list(edges)
        ids=rng.choice(len(el),2,replace=False)
        e1,e2=el[int(ids[0])],el[int(ids[1])]
        if len(set(e1+e2))<4: continue
        pairs=candidate_pairings(e1,e2)
        if rng.random()<0.5: pairs.reverse()
        for n1,n2 in pairs:
            if not valid_pair(n1,n2,e1,e2,edges): continue
            if not preserves_mixing(e1,e2,n1,n2): continue
            prop=(edges-{e1,e2})|{n1,n2}
            if not connected(prop): continue
            old_err=abs(q_rewire(edges)-target_q)
            new_err=abs(q_rewire(prop)-target_q)
            if new_err<=old_err or rng.random()<0.02:
                edges=prop
                if new_err<best_err:
                    best_err=new_err; best=set(edges)
                break
        if abs(q_rewire(edges)-target_q)<=tol:
            return edges, proposal
    if abs(q_rewire(best)-target_q)>tol:
        raise RuntimeError(f"Could not reach q={target_q:.3f}; best={q_rewire(best):.3f}")
    return best, max_props

assert len(NOMINAL_EDGES)==200
assert np.all(degree_vector(NOMINAL_EDGES)==DEGREE)
assert connected(NOMINAL_EDGES)

out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
levels=np.array(args.levels,float)
L=len(levels); G=args.graphs_per_level
bank=np.zeros((L,G,N,N),np.float32)
qvals=np.zeros((L,G)); l2vals=np.zeros((L,G)); cvals=np.zeros((L,G)); mvals=np.zeros((L,G))
bcounts=np.zeros((L,G),int); seeds=np.zeros((L,G),int)
rows=[]; seen=set()

print("="*80)
print("EXPERIMENT IV | UNSEEN MODULAR TOPOLOGY BANK")
print("="*80)
print("Nominal lambda2:", f"{lambda2(A_NOMINAL):.6f}")
print("Nominal modularity:", f"{modularity(A_NOMINAL):.6f}")
print("Nominal clustering:", f"{mean_clustering(A_NOMINAL):.6f}")
print("Nominal bridge nodes:", len(bridge_nodes(NOMINAL_EDGES)))

for li,target in enumerate(levels):
    gi=0; attempt=0
    print(f"\nTarget q={target:.2f}")
    while gi<G:
        attempt+=1
        seed=args.base_seed+li*10000+attempt
        edges,props=generate(float(target),seed,args.tolerance,args.max_proposals)
        h=tuple(sorted(edges))
        if h in seen: continue
        seen.add(h)
        if not np.all(degree_vector(edges)==DEGREE): raise RuntimeError("Degree changed")
        if not connected(edges): raise RuntimeError("Disconnected graph")
        if mixing_counts(edges)!=NOMINAL_MIXING: raise RuntimeError("Mixing matrix changed")
        A=edges_to_A(edges)
        q=q_rewire(edges); l2=lambda2(A); cc=mean_clustering(A); mm=modularity(A); bn=len(bridge_nodes(edges))
        bank[li,gi]=A; qvals[li,gi]=q; l2vals[li,gi]=l2; cvals[li,gi]=cc; mvals[li,gi]=mm; bcounts[li,gi]=bn; seeds[li,gi]=seed
        rows.append(dict(level_index=li,graph_index=gi,target_rewiring=float(target),actual_rewiring=q,
                         edge_retention=1-q,seed=seed,lambda2=l2,mean_clustering=cc,
                         modularity_fixed_partition=mm,bridge_node_count=bn,proposals=props))
        print(f" graph {gi+1:02d}/{G} | q={q:.3f} | λ2={l2:.5f} | C={cc:.4f} | Q={mm:.4f} | bridges={bn}")
        gi+=1

npz=out/"unseen_topology_bank.npz"
np.savez_compressed(npz, adjacency_nominal=A_NOMINAL.astype(np.float32), adjacency_bank=bank,
                    target_rewiring=levels, actual_rewiring=qvals, lambda2=l2vals,
                    mean_clustering=cvals, modularity=mvals, bridge_node_count=bcounts,
                    seeds=seeds, community=COMMUNITY)

meta=out/"unseen_topology_bank_metadata.csv"
with meta.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

summary=out/"unseen_topology_bank_summary.csv"
with summary.open("w",newline="",encoding="utf-8") as f:
    fields=["target_rewiring","n_graphs","actual_rewiring_mean","actual_rewiring_std",
            "lambda2_mean","lambda2_std","mean_clustering_mean","mean_clustering_std",
            "modularity_mean","modularity_std","bridge_node_count_mean","bridge_node_count_std"]
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for li,target in enumerate(levels):
        w.writerow(dict(target_rewiring=float(target),n_graphs=G,
                        actual_rewiring_mean=float(qvals[li].mean()),actual_rewiring_std=float(qvals[li].std(ddof=1)),
                        lambda2_mean=float(l2vals[li].mean()),lambda2_std=float(l2vals[li].std(ddof=1)),
                        mean_clustering_mean=float(cvals[li].mean()),mean_clustering_std=float(cvals[li].std(ddof=1)),
                        modularity_mean=float(mvals[li].mean()),modularity_std=float(mvals[li].std(ddof=1)),
                        bridge_node_count_mean=float(bcounts[li].mean()),bridge_node_count_std=float(bcounts[li].std(ddof=1))))

print("\nSaved:", npz)
print("Saved:", meta)
print("Saved:", summary)
