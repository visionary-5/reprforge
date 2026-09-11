"""Claim-focused plots; draw only observed results, never pending GPU outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY = '#2563A6', '#D17435', '#737B85'
NAMES = {'arxivqa':'ArxivQA', 'docvqa':'DocVQA', 'infovqa':'InfoVQA',
         'tabfquad':'TabFQuAD', 'tatdqa':'TAT-DQA', 'shiftproject':'Shift'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--full', type=Path)
    ap.add_argument('--mechanism', type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'axes.labelcolor':'#26313D','text.color':'#26313D','axes.edgecolor':'#9DA7B2'})
    provenance = {}
    def read(path):
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    def csv_read(path):
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return list(csv.DictReader(path.open()))
    def save(fig, name):
        fig.savefig(args.output / (name+'.pdf'),bbox_inches='tight',pad_inches=.06)
        fig.savefig(args.output / (name+'.png'),bbox_inches='tight',pad_inches=.06,dpi=240)
        plt.close(fig)
    # A histogram answers a concrete question: how much of the target result survives?
    if args.full:
        r = read(args.full/'stale-rankings.json')
        ref, stale, replay = r['target_top10'], r['stale_top10'], r['replay_top10']
        recovered = np.array([len(set(a)&set(b)) for a,b in zip(ref,stale,strict=True)])
        replay_recovered = np.array([len(set(a)&set(b)) for a,b in zip(ref,replay,strict=True)])
        old_exact = sum(a==b for a,b in zip(ref,stale,strict=True))
        new_exact = sum(a==b for a,b in zip(ref,replay,strict=True))
        title = 'ColQwen2.5 v0.1 → v0.2 | independent native target'
    else:
        rows = csv_read(args.data/'ranking-per-query.csv')
        old = [r for r in rows if r['route']=='stale']
        new = [r for r in rows if r['route']=='bf16']
        recovered = np.array([round(float(r['ta10'])*10) for r in old])
        replay_recovered = np.array([round(float(r['ta10'])*10) for r in new])
        old_exact = sum(int(r['ordered10']) for r in old)
        new_exact = sum(int(r['ordered10']) for r in new)
        title = 'ColQwen2 v0.1 → v1.0 | historical shared-prefix target reference'
    n = len(recovered)
    fig,axes = plt.subplots(1,2,figsize=(7.2,2.8),gridspec_kw={'width_ratios':[1.4,1]})
    ax=axes[0]
    x=np.arange(11)
    ax.bar(x-.18,np.bincount(recovered,minlength=11)/n*100,.35,color=ORANGE,label='Keep old index')
    ax.bar(x+.18,np.bincount(replay_recovered,minlength=11)/n*100,.35,color=BLUE,label='ReprForge')
    ax.set_xticks([0,2,4,6,8,10]);ax.set_xlim(-.6,10.7);ax.set_ylim(0,109)
    ax.set_xlabel('Target top-10 pages recovered (out of 10)')
    ax.set_ylabel('Queries (%)');ax.set_title('(a) Which pages are returned?',loc='left',fontsize=10,pad=12)
    ax.text(3.7,56,f'Old index retains\n{recovered.mean():.2f} of 10 pages\non average',color=ORANGE,ha='center',fontsize=9)
    recovered_all=float(np.mean(replay_recovered==10)*100)
    ax.annotate(f'{recovered_all:.1f}%',(10.18,recovered_all),xytext=(8.1,min(101,recovered_all+10)),color=BLUE,
                arrowprops={'arrowstyle':'-','color':BLUE},ha='center',weight='bold')
    ax.legend(frameon=False,loc='upper left',fontsize=8)
    ax=axes[1]
    vals=np.array([old_exact,new_exact])/n*100
    ax.barh([1,0],vals,color=[ORANGE,BLUE],height=.42)
    for y,v,k in zip([1,0],vals,[old_exact,new_exact],strict=True):
        outside=v<40
        ax.text(v+3 if outside else v-3,y,f'{k:,} / {n:,}',ha='left' if outside else 'right',
                va='center',color=([ORANGE,BLUE][1-y] if outside else 'white'),weight='bold',fontsize=10)
    ax.set_yticks([1,0],['Keep old index','ReprForge']);ax.set_xlim(0,105);ax.set_ylim(-.6,1.6)
    ax.set_xticks([0,50,100]);ax.set_xlabel('Queries with identical ordered top-10 (%)')
    ax.set_title('(b) Is the entire ranking identical?',loc='left',fontsize=10,pad=12)
    fig.suptitle(title,fontsize=9,y=1.03,color=GRAY)
    fig.subplots_adjust(wspace=.5,bottom=.22,top=.83,left=.08,right=.98)
    save(fig,'01-target-results')

    # A paired aggregate with a common baseline; dataset details belong in a table.
    if args.full:
        r=read(args.full/'vidore-v0.2-result.json')
        raw,replay_time=r['raw_page_seconds'],r['replay_with_read_seconds']; pages=r['pages']
    else:
        r=next(r for r in csv_read(args.data/'independent-summary.csv') if r['target']=='ColQwen2.5 v0.2')
        raw,replay_time=float(r['raw_seconds']),float(r['replay_seconds']);pages=int(r['pages'])
    ratio=replay_time/raw
    fig,ax=plt.subplots(figsize=(6.3,2.25))
    ax.barh([1,0],[100,ratio*100],height=.36,color=[GRAY,BLUE])
    ax.text(98,1,f'{raw:.1f} s',va='center',ha='right',color='white',weight='bold')
    ax.text(ratio*100+2,0,f'{replay_time:.1f} s  ({ratio*100:.1f}% of raw)',va='center',color=BLUE,weight='bold')
    ax.annotate('',xy=(ratio*100,-.34),xytext=(100,-.34),arrowprops={'arrowstyle':'<->','color':BLUE})
    direction='less' if ratio<=1 else 'more'
    ax.text((100+ratio*100)/2,-.59,f'{abs(1-ratio)*100:.1f}% {direction} measured document encoding time',ha='center',color=BLUE,fontsize=9)
    ax.set_yticks([1,0],['Full target encoding','ReprForge replay']);ax.set_xlim(0,max(105,ratio*105));ax.set_ylim(-.8,1.45)
    ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Document-side time (% of paired raw baseline)')
    ax.set_title(f'ColQwen2.5 v0.1 → v0.2 | {pages:,} pages | A100, BF16, batch 1',fontsize=10,pad=14)
    fig.subplots_adjust(left=.25,right=.97,bottom=.23,top=.8)
    save(fig,'02-recomputation')

    if args.mechanism:
        r=read(args.mechanism/'result.json'); n=r['pages_per_case']
        path=args.mechanism/'pages.jsonl'
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        pages=[json.loads(line) for line in path.read_text().splitlines()]
        specs=[('release_control','Adapter update'),('target_text_embedding','Text embeddings change'),
               ('target_vision','Vision weights change'),('target_merger','Merger weights change')]
        # Both always-visual replay and dependency-validated execution were actually run.
        values=[]
        for case,label in specs:
            c=r['by_case'][case]
            good='raw_fallback' if case in ['target_vision','target_merger'] else 'factorized_replay'
            values.append([c['factorized_replay']/n,c[good]/n])
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={'width_ratios':[1.5,1]})
        from matplotlib.colors import ListedColormap
        ax=axes[0]; vals=np.array(values)
        ax.imshow(vals,cmap=ListedColormap(['#F9E7D8','#DCEAF5']),vmin=0,vmax=1,aspect='auto')
        for y,row in enumerate(vals):
            for x,v in enumerate(row):
                ax.text(x,y,f'{round(v*n)}/{n} exact',va='center',ha='center',weight='bold',color=BLUE if v==1 else ORANGE)
        ax.set_yticks(range(4),[v[1] for v in specs]);ax.set_xticks([0,1],['Always reuse\nvisual state','Dependency-validated\nexecution'])
        ax.tick_params(length=0);ax.set_title('(a) Does reuse preserve the target?',loc='left',fontsize=10,pad=12)
        for sp in ax.spines.values():sp.set_visible(False)
        ax=axes[1]
        skipped=[100*sum(p['factorized_admitted'] and p['replay_vision_calls']==0 for p in pages if p['case']==case)/n for case,_ in specs]
        ax.barh([3,2,1,0],skipped,height=.5,color=BLUE)
        for y in [3,2]:ax.text(50,y,'Replay',ha='center',va='center',color='white',weight='bold')
        for y in [1,0]:ax.text(3,y,'Raw fallback',va='center',color=GRAY)
        ax.set_ylim(-.5,3.5);ax.set_xlim(0,105);ax.set_yticks([]);ax.set_xticks([0,50,100])
        ax.set_xlabel('Pages skipping target vision (%)')
        ax.set_title('(b) What computation is reused?',loc='left',fontsize=10,pad=12)
        fig.suptitle(f'{n} fixed pages per condition | independent native reference | controlled changes',fontsize=9,y=1.03,color=GRAY)
        fig.subplots_adjust(left=.26,right=.98,bottom=.25,top=.83,wspace=.25)
        save(fig,'03-validity-and-reuse')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    main()
