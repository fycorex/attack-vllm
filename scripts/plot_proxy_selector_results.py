#!/usr/bin/env python3
"""Create the required lightweight paper-facing pilot plots."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    root=Path('outputs/proxy_selector_pilot'); plots=root/'plots'; plots.mkdir(parents=True,exist_ok=True)
    matrix=list(csv.DictReader((root/'summaries/proxy_target_matrix.csv').open()))
    cka=list(csv.DictReader((root/'cka/cka_seed42.csv').open()))
    names=['P1','P2','P3','T1','T2']; table=np.array([[float(next(r['cka'] for r in cka if r['left']==a and r['right']==b)) for b in names] for a in names])
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(table,vmin=0,vmax=1,cmap='viridis');ax.set(xticks=range(5),xticklabels=names,yticks=range(5),yticklabels=names,title='Primary-gallery centered linear CKA');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(plots/'cka_heatmap.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4))
    for r in matrix: ax.scatter(float(r['cka']),float(r['tasr']));ax.annotate(r['pair_id'],(float(r['cka']),float(r['tasr'])),xytext=(3,3),textcoords='offset points')
    ax.set(xlabel='CKA',ylabel='TASR',title='CKA vs final TASR (descriptive)');fig.tight_layout();fig.savefig(plots/'cka_vs_tasr.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7,3),sharey=True)
    for ax,target in zip(axes,['T1','T2']):
        rows=[r for r in matrix if r['target']==target];ax.bar([r['proxy'] for r in rows],[float(r['tasr']) for r in rows]);ax.set_title(target);ax.set_ylabel('TASR')
    fig.tight_layout();fig.savefig(plots/'proxy_tasr_rankings.png',dpi=160);plt.close(fig)
    rank=json.loads((root/'cka/cka_bootstrap.json').read_text())['proxy_rankings'];fig,ax=plt.subplots(figsize=(6,3));
    for i,target in enumerate(('T1','T2')): ax.plot([1,2,3],rank[target]['primary_ranking'],'o-',label=f'{target} primary');ax.plot([1,2,3],rank[target]['stability_ranking'],'x--',label=f'{target} stability')
    ax.set(xticks=[1,2,3],xlabel='Rank',ylabel='Proxy',title='Primary vs stability CKA rankings');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(plots/'cka_ranking_stability.png',dpi=160);plt.close(fig)
if __name__=='__main__': main()
