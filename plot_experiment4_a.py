#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
mpl.rcParams['font.family']='DejaVu Serif'
mpl.rcParams['axes.linewidth']=0.9

METHOD_ORDER=['nrpc_gl','nrpc_local','random','betweenness','full','uncontrolled']
LABELS={
 'nrpc_gl':'NRPC-GL','nrpc_local':'NRPC-L','random':r'Random-$k$',
 'betweenness':r'Betweenness-$k$','full':'Full consensus','uncontrolled':'Uncontrolled'}
MARKERS={'nrpc_gl':'o','nrpc_local':'^','random':'*','betweenness':'d','full':'s','uncontrolled':'X'}
SIZES={'nrpc_gl':60,'nrpc_local':60,'random':75,'betweenness':55,'full':55,'uncontrolled':60}


def detect(df, candidates, explicit=None, required=True):
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Column {explicit!r} not found. Available: {list(df.columns)}")
        return explicit
    low={c.lower():c for c in df.columns}
    for x in candidates:
        if x.lower() in low:
            return low[x.lower()]
    for x in candidates:
        for c in df.columns:
            if x.lower() in c.lower():
                return c
    if required:
        raise ValueError(f"Could not detect column from {candidates}. Available: {list(df.columns)}")
    return None


def norm_method(x):
    k=str(x).strip().lower().replace('-','_').replace(' ','_')
    aliases={
      'gl':'nrpc_gl','global_local':'nrpc_gl','nrpc_global_local':'nrpc_gl','nrpc_gl':'nrpc_gl',
      'local':'nrpc_local','nrpc_l':'nrpc_local','nrpc_local':'nrpc_local',
      'random_k':'random','random_topk':'random','random':'random',
      'betweenness_k':'betweenness','betweenness':'betweenness',
      'consensus':'full','full_consensus':'full','full':'full',
      'no_control':'uncontrolled','nocontrol':'uncontrolled','uncontrolled':'uncontrolled'}
    return aliases.get(k,k)


def latest_csv(root):
    cand=sorted(root.glob('results/experiment4/**/*enriched.csv'), key=lambda p:p.stat().st_mtime)
    if not cand:
        raise FileNotFoundError('No enriched CSV found under results/experiment4; pass --csv explicitly.')
    return cand[-1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',type=Path,default=None)
    ap.add_argument('--outdir',type=Path,default=Path('figures/experiment4'))
    ap.add_argument('--beta',type=float,default=5.0)
    ap.add_argument('--topology-col',default=None)
    ap.add_argument('--lambda2-col',default=None)
    ap.add_argument('--resilience-col',default=None)
    ap.add_argument('--method-col',default=None)
    ap.add_argument('--beta-col',default=None)
    ap.add_argument('--no-full',action='store_true')
    ap.add_argument('--no-uncontrolled',action='store_true')
    args=ap.parse_args()

    path=args.csv or latest_csv(Path.cwd())
    df=pd.read_csv(path)

    top=detect(df,['topology_id','graph_id','topology_name','topology','graph','network_id'],args.topology_col)
    lam=detect(df,['lambda2','lambda_2','algebraic_connectivity','laplacian_lambda2','spectral_gap'],args.lambda2_col)
    res=detect(df,['resilience','rho','resilience_mean','mean_resilience'],args.resilience_col)
    met=detect(df,['method','controller','policy','model'],args.method_col)
    bet=detect(df,['beta','control_gain','gain'],args.beta_col,required=False)

    print(f'[input] {path}')
    print(f'[detected] topology={top}, lambda2={lam}, resilience={res}, method={met}, beta={bet}')

    work=df.copy()
    work['_method']=work[met].map(norm_method)
    work[lam]=pd.to_numeric(work[lam],errors='coerce')
    work[res]=pd.to_numeric(work[res],errors='coerce')
    if bet is not None:
        bvals=pd.to_numeric(work[bet],errors='coerce')
        work=work[np.isclose(bvals,args.beta)].copy()
        if work.empty:
            avail=sorted(pd.to_numeric(df[bet],errors='coerce').dropna().unique())
            raise ValueError(f'No rows for beta={args.beta:g}. Available: {avail}')

    work=work.dropna(subset=[top,lam,res,'_method'])
    agg=(work.groupby([top,'_method'],as_index=False)
         .agg(lambda2=(lam,'mean'),resilience_mean=(res,'mean'),resilience_std=(res,'std'),n=(res,'count')))
    agg['resilience_sem']=agg['resilience_std']/np.sqrt(agg['n'].clip(lower=1))
    agg['method']=agg['_method']

    args.outdir.mkdir(parents=True,exist_ok=True)
    beta_suffix=f"_beta{str(args.beta).replace('.','p')}" if bet is not None else ''
    agg_path=args.outdir/f'fig4A_unseen_topologies_aggregated{beta_suffix}.csv'
    agg.to_csv(agg_path,index=False)

    methods=['nrpc_gl','nrpc_local','random','betweenness']
    if not args.no_full: methods.append('full')
    if not args.no_uncontrolled: methods.append('uncontrolled')

    fig,ax=plt.subplots(figsize=(8.5,5.9))

    for m in methods:
        d=agg[agg['method']==m].sort_values('lambda2')
        if d.empty:
            continue

        # Connect topology-level observations to make the trend easier
        # to follow visually. This is only a guide, not a fitted curve.
        ax.plot(
            d['lambda2'],
            d['resilience_mean'],
            linewidth=1.35,
            alpha=.90,
            zorder=2,
        )

        ax.scatter(
            d['lambda2'],
            d['resilience_mean'],
            s=SIZES[m],
            marker=MARKERS[m],
            alpha=.96,
            label=LABELS[m],
            edgecolors='black',
            linewidths=.45,
            zorder=3,
        )

    ax.set_xlabel(r'Algebraic Connectivity $\lambda_2(L)$',fontsize=15,fontweight='bold')
    ax.set_ylabel(r'Mean Resilience $\rho$',fontsize=16,fontweight='bold')

    # title='Transfer to Unseen Network Topologies'
    # if bet is not None:
    #     title += rf' ($\beta={args.beta:g}$)'

    #ax.set_title(title,fontsize=14,fontweight='bold',pad=10)
    ax.grid(alpha=.13,linewidth=.75)
    ax.tick_params(labelsize=11.5)

    leg=ax.legend(
        ncol=2,
        fontsize=10.3,
        frameon=True,
        fancybox=True,
        framealpha=.95,
        edgecolor='0.45',
        facecolor='white',
        borderpad=.65,
    )
    leg.get_frame().set_linewidth(.9)

    fig.tight_layout()

    pdf=args.outdir/f'fig4A_unseen_topologies{beta_suffix}.pdf'
    png=args.outdir/f'fig4A_unseen_topologies{beta_suffix}.png'
    fig.savefig(pdf,bbox_inches='tight')
    fig.savefig(png,dpi=300,bbox_inches='tight')
    plt.close(fig)

    print(f'[saved] {pdf}')
    print(f'[saved] {png}')
    print(f'[saved] {agg_path}')
    print(f'[topologies] {agg[top].nunique()}')
    print(f'[methods] {sorted(agg.method.unique())}')

if __name__=='__main__':
    main()
