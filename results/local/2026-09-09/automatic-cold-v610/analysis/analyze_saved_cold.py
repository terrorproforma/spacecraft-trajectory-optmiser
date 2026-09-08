#!/usr/bin/env python3
"""CPU-only exact-FP64-input Decimal audit of saved v603/v608/v609 cold outputs.

Run from the repository root. Uses Python's standard library, no CUDA/solver,
and never rewrites an input, log, source file or acceptance threshold.
"""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import tarfile

BASE = Path('results/local/2026-09-09')
CAPTURES = {
    'conditioning': '1eb1b5a3979c5e86522337dfa94f264e354993163ed44129eff361371e3b49cf',
    'difficult': '14fdc84e9b654df4ac16db71f0bce4fccfa62577f9620ae622ad2196e2f17080',
}
SOURCES = {
    '_upstream/pdhcg/src/utils.cu': [291, 324, 834],
    '_upstream/pdhcg/src/solver.cu': [117, 121, 145, 165],
    '_upstream/pdhcg/src/pdhg_core_op.cu': [495, 707, 832, 859, 933, 973],
    '_upstream/pdhcg/src/kernels/pdhcg_kernels.cu': [283],
    '_upstream/pdhcg/src/preconditioner.c': [822],
    'cpp/cuda/src/cooperative_pdhg.cuh': [186, 525, 793, 930],
    'cpp/cuda/src/persistent_pdhcg.cu': [876, 1177],
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dec(value):
    return Decimal.from_float(float(value))


def dot(a, b):
    assert len(a) == len(b)
    return sum((x * y for x, y in zip(a, b)), Decimal(0))


def inf(values):
    return max(map(abs, values), default=Decimal(0))


def stats(values):
    values = list(values)
    nonzero = sorted(abs(float(v)) for v in values if v)
    return dict(count=len(values), nonzero=len(nonzero),
                min_nonzero=nonzero[0] if nonzero else None,
                median_nonzero=nonzero[len(nonzero)//2] if nonzero else None,
                max_abs=nonzero[-1] if nonzero else 0.0)


class Snapshot:
    def __init__(self, label):
        self.path = BASE / f'upstream-identical-capture-v608/fixtures/{label}.txt'
        data = self.path.read_bytes()
        assert sha(data) == CAPTURES[label]
        lines = data.decode().splitlines()
        assert len(lines) == 15 and lines[0] == 'SPACEPDHCG_QOCO_QP_V1'
        self.n, self.p, self.m, nq, na, ng, self.l, ns, shifted = map(int, lines[1].split())
        assert shifted == 0
        self.ptr = [[int(x) for x in lines[k].split()[1:]] for k in (4, 6, 8)]
        self.idx = [[int(x) for x in lines[k].split()[1:]] for k in (5, 7, 9)]
        self.soc = [int(x) for x in lines[10].split()[1:]]
        assert len(self.soc) == ns and self.l + sum(self.soc) == self.m
        values = [dec(x) for x in lines[11].split()[1:]]
        assert len(values) == nq + na + ng + self.n + self.p + self.m
        self.val = [values[:nq], values[nq:nq+na], values[nq+na:nq+na+ng]]
        at = nq + na + ng
        self.c = values[at:at+self.n]
        self.b = values[at+self.n:at+self.n+self.p]
        self.h = values[at+self.n+self.p:]
        assert all(v == 0 for v in self.val[0]) and dec(lines[14]) == 0
        self.row = []
        for matrix, count in enumerate((self.n, self.p, self.m)):
            rows = [[] for _ in range(count)]
            for j in range(self.n):
                for k in range(self.ptr[matrix][j], self.ptr[matrix][j+1]):
                    i, v = self.idx[matrix][k], self.val[matrix][k]
                    rows[i].append((j, v))
                    if matrix == 0 and i != j:
                        rows[j].append((i, v))
            self.row.append(rows)

    def mul(self, matrix, x, transpose=False):
        if not transpose:
            return [sum((v*x[j] for j, v in row), Decimal(0)) for row in self.row[matrix]]
        out = [Decimal(0)] * self.n
        for i, row in enumerate(self.row[matrix]):
            for j, v in row:
                out[j] += v*x[i]
        return out

    def metadata(self):
        scalar = self.row[2][:self.l]
        equality_columns = {j for row in self.row[1] for j,v in row if v}
        incidence = [[] for _ in range(self.n)]
        for i,row in enumerate(self.row[2]):
            for j,v in row:
                if v:
                    incidence[j].append((i,v))
        exact_epigraphs=[]
        for t,cost in enumerate(self.c):
            if cost < 10000 or t in equality_columns or len(incidence[t]) != 2:
                continue
            pair=[]
            for i,v in incidence[t]:
                other=[(j,a) for j,a in self.row[2][i] if j != t and a]
                if i >= self.l or v != -1 or self.h[i] != 0 or len(other) != 1:
                    break
                pair.append(other[0])
            if len(pair)==2 and pair[0][0]==pair[1][0] and {pair[0][1],pair[1][1]}=={Decimal(-1),Decimal(1)}:
                exact_epigraphs.append(dict(t=t,v=pair[0][0],cost=float(cost),rows=[i for i,_ in incidence[t]]))
        return dict(input_sha256=sha(self.path.read_bytes()), n=self.n, p=self.p, m=self.m,
                    nonnegative_rows=self.l, soc_sizes=dict(Counter(self.soc)),
                    P_numerical_nonzeros=0,
                    P_stored_upper=len(self.val[0]), P_stored_full=sum(map(len, self.row[0])),
                    P_stored_offdiagonal_full=sum(i != j for i, row in enumerate(self.row[0]) for j, _ in row),
                    c=stats(self.c), b=stats(self.b), h=stats(self.h),
                    c_magnitude_groups=dict(Counter('zero' if not v else 'abs>=10000' if abs(v)>=10000 else 'abs<1' if abs(v)<1 else 'other' for v in self.c)),
                    A_coefficients=stats(self.val[1]), G_coefficients=stats(self.val[2]),
                    A_row_max=stats(max((abs(v) for _, v in row), default=Decimal(0)) for row in self.row[1]),
                    G_row_max=stats(max((abs(v) for _, v in row), default=Decimal(0)) for row in self.row[2]),
                    nonnegative_singleton_rows=sum(sum(v != 0 for _, v in row)==1 for row in scalar),
                    exact_large_cost_absolute_value_epigraph_count=len(exact_epigraphs),
                    exact_epigraph_targets_distinct=len({r['v'] for r in exact_epigraphs})==len(exact_epigraphs),
                    exact_epigraph_examples=exact_epigraphs[:3])

    def audit(self, raw):
        x, y, z, s = [[dec(v) for v in raw[k]] for k in ('x', 'y', 'z', 's')]
        assert tuple(map(len,(x,y,z,s))) == (self.n,self.p,self.m,self.m)
        px, ax, gx = self.mul(0,x), self.mul(1,x), self.mul(2,x)
        aty, gtz = self.mul(1,y,True), self.mul(2,z,True)
        rd = [q+c+a+g for q,c,a,g in zip(px,self.c,aty,gtz)]
        eq = [a-b for a,b in zip(ax,self.b)]
        ce = [h-g-a for h,g,a in zip(self.h,gx,s)]
        f = dot(x,px)/2 + dot(self.c,x)
        d = -dot(x,px)/2 - dot(self.b,y) - dot(self.h,z)
        scale = max(Decimal(1),abs(f),abs(d))
        primal = max(inf(eq),inf(ce))/(1+max(map(inf,(ax,self.b,gx,self.h,s))))
        dual = inf(rd)/(1+max(map(inf,(px,self.c,aty,gtz))))
        pc = max([Decimal(0)]+[-v for v in s[:self.l]])
        dc = max([Decimal(0)]+[-v for v in z[:self.l]])
        block = inf(a*b for a,b in zip(s[:self.l],z[:self.l]))
        soc_blocks=[]
        start=self.l
        for number,size in enumerate(self.soc):
            end=start+size
            pviolation=max(Decimal(0),dot(s[start+1:end],s[start+1:end]).sqrt()-s[start])
            dviolation=max(Decimal(0),dot(z[start+1:end],z[start+1:end]).sqrt()-z[start])
            comp=dot(s[start:end],z[start:end])
            pc=max(pc,pviolation);dc=max(dc,dviolation);block=max(block,abs(comp))
            soc_blocks.append(dict(block=number,start=start,primal_violation=float(pviolation),
                                   dual_violation=float(dviolation),complementarity=float(comp)))
            start=end
        gap=abs(f-d)/scale
        passes=max(primal,dual,gap,block/scale)<=dec(1e-9) and max(pc,dc)<=dec(1e-8)
        identity=[dot(x,rd),-dot(eq,y),dot(s,z),dot(ce,z)]
        error=f-d-sum(identity,Decimal(0))
        assert abs(error)<Decimal('1e-45')*max(Decimal(1),*(abs(v) for v in identity))
        top=sorted(range(self.n),key=lambda j:abs(rd[j]),reverse=True)[:8]
        return dict(primal=float(primal),dual=float(dual),gap=float(gap),passes=passes,
                    primal_cone_violation=float(pc),dual_cone_violation=float(dc),
                    block_complementarity_normalized=float(block/scale),
                    objective=float(f),dual_objective=float(d),signed_gap=float(f-d),
                    gap_decomposition=dict(zip(('x_dot_stationarity','minus_equality_error_dot_y','s_dot_z','slack_reconstruction_error_dot_z'),map(float,identity))),
                    gap_identity_error=float(error),
                    objective_on_abs_c_ge_10000=float(sum((c*v for c,v in zip(self.c,x) if abs(c)>=10000),Decimal(0))),
                    equality_max_absolute=float(inf(eq)), stationarity_max_absolute=float(inf(rd)),
                    nonnegative_primal_violation=float(max([Decimal(0)]+[-v for v in s[:self.l]])),
                    top_stationarity=[dict(variable=j,residual=float(rd[j]),c=float(self.c[j]),x=float(x[j]),
                                           Aeq_transpose_y=float(aty[j]),G_transpose_z=float(gtz[j])) for j in top],
                    top_equality=[dict(row=i,error=float(eq[i]),y=float(y[i]),error_times_y=float(eq[i]*y[i]))
                                  for i in sorted(range(self.p),key=lambda i:abs(eq[i]*y[i]),reverse=True)[:5]],
                    top_nonnegative_complementarity=[dict(row=i,slack=float(s[i]),z=float(z[i]),product=float(s[i]*z[i]),
                                                          h=float(self.h[i]),coefficients=[[j,float(v)] for j,v in self.row[2][i]])
                                                     for i in sorted(range(self.l),key=lambda i:abs(s[i]*z[i]),reverse=True)[:5]],
                    top_soc_violation=sorted(soc_blocks,key=lambda v:v['primal_violation'],reverse=True)[:3])


def parse(data,prefix):
    records=[json.loads(line.split(' ',1)[1]) for line in data.decode().splitlines() if line.startswith(prefix+' {')]
    assert len(records)==1
    return records[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    archive_paths=[BASE/'persistent-snapshot-v603/real/raw.tar.gz',BASE/'persistent-common-kkt-v609/real/raw-logs.tar.gz']
    manifest_paths=[BASE/'persistent-snapshot-v603/real/report.json',BASE/'upstream-identical-capture-v608/manifest.json',
                    BASE/'upstream-identical-capture-v608/run/report.json',BASE/'persistent-common-kkt-v609/real/manifest.json',
                    BASE/'persistent-common-kkt-v609/real/run/report.json']
    cases=[]
    with tarfile.open(archive_paths[0]) as archive:
        for label in CAPTURES:
            data=archive.extractfile(f'run/{label}-persistent.log').read()
            cases.append(('v603',label,data,'PERSISTENT_REPLAY'))
    for label in CAPTURES:
        data=(BASE/f'upstream-identical-capture-v608/run/{label}-cold.log').read_bytes()
        cases.append(('v608',label,data,'UPSTREAM_REPLAY_RESULT'))
    with tarfile.open(archive_paths[1]) as archive:
        for label in CAPTURES:
            for mode in ('natural','common'):
                data=archive.extractfile(f'run/{label}-cold-{mode}-blocks0.log').read()
                cases.append(('v609-'+mode,label,data,'PERSISTENT_REPLAY'))
    with localcontext() as context:
        context.prec=65
        snapshots={label:Snapshot(label) for label in CAPTURES}
        rows=[]
        for version,label,data,prefix in cases:
            raw=parse(data,prefix)
            metrics=snapshots[label].audit(raw)
            assert metrics['passes'] == raw.get('passes_common_kkt_gate',raw.get('kkt_qualified_original')) == False
            rows.append(dict(version=version,capture=label,log_sha256=sha(data),
                             iterations=raw['iterations'],termination=raw.get('termination_name',raw['termination']),
                             metrics=metrics))
        output=dict(scope='CPU saved-vector/source diagnosis, no new solver calls or timing claim',
                    script_sha256=sha(Path(__file__).read_bytes()),decimal_precision=65,
                    source_identities={p:dict(sha256=sha(Path(p).read_bytes()),reviewed_lines=lines) for p,lines in SOURCES.items()},
                    evidence_identities={str(p).replace('\\','/'):sha(p.read_bytes()) for p in archive_paths+manifest_paths},
                    pinned_upstream_commit='167c8b72b4b96d2f94d405b8763e485514192b81',
                    original_persistent_core_sha256='d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633',
                    common_kkt_core_sha256='a38d579579be2f47f9d8a8290a9440fd43f5beca2864773d9a5d6465a2ee61dd',
                    captures={label:s.metadata() for label,s in snapshots.items()},rows=rows)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
