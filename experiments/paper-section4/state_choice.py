"""Show state-choice evidence only after both prespecified conditions complete."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pages',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--expected-pages',type=int,default=200)
    args=ap.parse_args()
    records=[json.loads(line) for line in args.pages.read_text().splitlines()]
    cases=['release_control','target_text_embedding']
    groups={c:[r for r in records if r['case']==c] for c in cases}
    n=args.expected_pages
    for group in groups.values():
        assert [r['index'] for r in group]==list(range(n))
        assert len({r['page']['image_sha256'] for r in group})==n
    assert [r['page'] for r in groups[cases[0]]]==[r['page'] for r in groups[cases[1]]]
    checks=['stale_fused_replay','factorized_replay']
    counts={c:[sum(r['checks'][k]['bit_equal'] for r in groups[c]) for k in checks] for c in cases}
    args.output.mkdir(parents=True,exist_ok=True)
    blue,orange,gray='#2563A6','#D17435','#737B85'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.4,2.9),gridspec_kw={'width_ratios':[1.35,1]})
    ax=axes[0];ax.set_xlim(-.2,10.2);ax.set_ylim(-.6,2.45);ax.axis('off')
    def box(x,y,w,text,color,fill):
        ax.add_patch(FancyBboxPatch((x,y),w,.45,boxstyle='round,pad=.03,rounding_size=.035',edgecolor=color,facecolor=fill,lw=.9))
        ax.text(x+w/2,y+.225,text,ha='center',va='center',fontsize=8,color=color)
    for y,label,target_text in [(1.25,'Retain source text',False),(0,'Rebuild target text',True)]:
        ax.text(0,y+.63,label,weight='bold',fontsize=9,color=blue if target_text else orange)
        box(0,y,2.4,'Saved visual\ntokens',gray,'#F2F4F6')
        ax.text(2.7,y+.22,'+',ha='center',va='center')
        box(3,y,2.8,'Target text\nembeddings' if target_text else 'Source text\nembeddings',blue if target_text else orange,'#E7F0F9' if target_text else '#FCEDDF')
        ax.annotate('',xy=(7,y+.225),xytext=(5.9,y+.225),arrowprops={'arrowstyle':'->','color':gray})
        box(7,y,2.85,'Target suffix',gray,'#F2F4F6')
    ax.text(0,-.44,'Same input IDs / mask / grid; target positions rebuilt',fontsize=6.8,color=gray)
    ax.set_title('(a) What crosses the version boundary?',loc='left',fontsize=10,pad=8)
    ax=axes[1]
    for i,(case,label) in enumerate(zip(cases,['Adapter update','+ text embedding\nchange'],strict=True)):
        for j,(color,route) in enumerate(zip([orange,blue],['Retain source text','Rebuild target text'],strict=True)):
            v=counts[case][j]/n*100
            x=i+(j-.5)*.32
            ax.bar(x,v,width=.29,color=color,label=route if i==0 else None)
            ax.text(x,v+3,f'{counts[case][j]}/{n}',ha='center',va='bottom',fontsize=7.5,color=color)
    ax.set_xticks([0,1],['Adapter update','+ text embedding\nchange'])
    ax.set_ylim(0,122);ax.set_yticks([0,50,100]);ax.set_ylabel('Bitwise-equal pages (%)')
    ax.set_title('(b) Independent raw-target checks',loc='left',fontsize=10,pad=8)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=8,bbox_to_anchor=(.68,-.035))
    fig.subplots_adjust(left=.02,right=.98,top=.86,bottom=.23,wspace=.28)
    for extension in ['pdf','png']:
        fig.savefig(args.output/('03-state-choice.'+extension),bbox_inches='tight',dpi=260)
    plt.close(fig)
    data={'conditions':counts,'pages_per_condition':n,'snapshot_sha256':hashlib.sha256(args.pages.read_bytes()).hexdigest(),
          'scope':'Two completed prespecified conditions from the ongoing 5-condition run; synthetic text perturbation; source fused text reconstructed, not independently persisted.'}
    (args.output/'state-choice-data.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':
    main()
