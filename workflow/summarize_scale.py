"""Summarize the paired trials. Optional plotting uses a locally installed matplotlib."""
import argparse
import collections
import json
import math
import statistics
from pathlib import Path

MED = statistics.median


def measure(report):
    data = collections.defaultdict(dict)
    for r in report["samples"]:
        key = r["phase"], r["orders"], r["mode"]
        data[key][r["trial"]] = r["ms"]
    return data


def paired(data,phase,n,base,mode):
    times=data[phase,n,mode]
    ratios=[data[phase,n,base][i]/v for i,v in times.items()]
    return MED(ratios), sum(x>1 for x in ratios)


def stable_start(report,data,phase,base,mode):
    sizes=sorted(map(int,report["cases"]))
    first=None
    for n in reversed(sizes):
        ratio,wins=paired(data,phase,n,base,mode)
        if ratio < 1.05 or wins < math.ceil(.7*report["repeat"]):
            break
        first=n
    return first


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("report",type=Path)
    p.add_argument("--plot",type=Path)
    args=p.parse_args()
    r=json.loads(args.report.read_text())
    data=measure(r)
    sizes=sorted(map(int,r["cases"]))
    phases=[p for p in ("cold","warm") if (p,sizes[0],"cpu") in data]
    modes=[m for m in ("legacy","cpu","simt","tile","triton") if (phases[0],sizes[0],m) in data]
    for phase in phases:
        print("\n"+phase+" (median milliseconds)")
        print("orders,ticks,forecasts,"+",".join(modes))
        for n in sizes:
            c=r["cases"][str(n)]
            print(n,c["ticks"],c["forecasts"],*(round(MED(data[phase,n,m].values()),3) for m in modes),sep=",")
        for base in modes[:2]:
            for m in modes[2:]:
                print("stable first order count",base,m,stable_start(r,data,phase,base,m))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,constrained_layout=True)
        x=[r["cases"][str(n)]["forecasts"] for n in sizes]
        for ax,base,title in zip(axes,modes[:2],("Original window-scan CPU","Incremental CPU")):
            for phase,color,label in (("cold","#ae5522","New process"),("warm","#226987","Resident worker")):
                if phase not in phases: continue
                bands=[[paired(data,phase,n,base,m)[0] for m in modes[2:]] for n in sizes]
                ax.fill_between(x,[min(v) for v in bands],[max(v) for v in bands],color=color,alpha=.16)
                ax.plot(x,[MED(v) for v in bands],marker=".",color=color,label=label)
            ax.axhline(1,color="#454545",lw=1,ls="--")
            ax.set(xscale="log",yscale="log",title=title,xlabel="Emitted forecasts")
            ax.grid(True,which="major",alpha=.2)
        axes[0].set_ylabel("End-to-end speedup vs CPU (×)")
        axes[0].legend(frameon=False,loc="lower right")
        fig.suptitle("RTX 3090  ·  CSV input to CSV output")
        fig.savefig(args.plot,dpi=170)
        plt.close(fig)


if __name__=="__main__": main()
