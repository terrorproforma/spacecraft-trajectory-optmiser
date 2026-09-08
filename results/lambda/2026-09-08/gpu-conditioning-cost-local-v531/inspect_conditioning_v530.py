from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import numpy as np
import scipy.sparse as sp
import clarabel
from analyse_qps_v169 import problem, audit

root = Path('build/performance/conditioning-matrices-v530')
root.mkdir(exist_ok=False)
source = Path('/home/angus/build-qoco-soc-step-v358/source')
lib = Path('/home/angus/build-qoco-soc-step-v358/final/libqoco.so')
binary = (root / 'qoco_snapshot_replay').resolve()
env = {k: v for k, v in os.environ.items() if not k.startswith(('SPACEPDHCG_TEST_', 'QOCO_REPLAY_'))}
env.update(LD_LIBRARY_PATH=str(lib.parent) + ':/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64',
           SPACEPDHCG_TEST_QOCO_IPM_GRAPH='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
report = dict(pid=os.getpid(), complete=False, cases=[], qoco_sha256=hashlib.sha256(lib.read_bytes()).hexdigest())
def save():
    t = root / 'report.tmp'
    t.write_text(json.dumps(report, indent=2))
    t.replace(root / 'report.json')
save()
try:
    code = Path('build/performance/conditioning-paths-v529/qoco_snapshot_replay.cu').read_text()
    code = code.replace('#include "qoco.h"', '#include "qoco.h"\n#include "cuda_types.h"')
    helper = r'''
template<class T> void device_json(const T* x,int n) {
    std::vector<T> v(n); if(n) CHECK(cudaMemcpy(v.data(),x,n*sizeof(T),cudaMemcpyDeviceToHost));
    std::cout<<'['; for(int i=0;i<n;++i) {if(i) std::cout<<','; number(v[i]);} std::cout<<']';
}
void matrix_json(QOCOMatrix* matrix) {
    auto* c=matrix->d_csc_host;
    std::cout<<"{\"p\":";device_json(c->p,c->n+1);
    std::cout<<",\"i\":";device_json(c->i,c->nnz);
    std::cout<<",\"x\":";device_json(c->x,c->nnz);std::cout<<'}';
}
void dump_scaling(QOCOSolver* solver) {
    auto* w=solver->work;auto* d=w->data;auto* s=w->scaling;
    std::cout<<"QP_SCALING {\"k\":";number(s->k);
    std::cout<<",\"kinv\":";number(s->kinv);
    std::cout<<",\"D\":";device_json(s->Druiz->d_data,d->n);
    std::cout<<",\"E\":";device_json(s->Eruiz->d_data,d->p);
    std::cout<<",\"F\":";device_json(s->Fruiz->d_data,d->m);
    std::cout<<",\"P\":";matrix_json(d->P);
    std::cout<<",\"A\":";matrix_json(d->A);
    std::cout<<",\"G\":";matrix_json(d->G);
    std::cout<<",\"c\":";device_json(d->c->d_data,d->n);
    std::cout<<",\"b\":";device_json(d->b->d_data,d->p);
    std::cout<<",\"h\":";device_json(d->h->d_data,d->m);
    std::cout<<"}\n"<<std::flush;
}
'''
    code = code.replace('int main(int argc', helper + '\nint main(int argc')
    code = code.replace('CHECK(qoco_gpu_primal_start(solver,0));', 'dump_scaling(solver);CHECK(qoco_gpu_primal_start(solver,0));')
    (root / 'qoco_snapshot_replay.cu').write_text(code)
    for name in ['analyse_qps_v169.py', 'inspect_conditioning_v530.py']:
        shutil.copy2(Path('build/performance') / name, root / name)
    cmd = ['/usr/local/cuda-12.8/bin/nvcc', '--default-stream', 'per-thread', '-arch=sm_120', '-std=c++17']
    for include in ['include', 'algebra/cuda', 'lib/qdldl/include', 'lib/amd']:
        cmd += ['-I', str(source / include)]
    cmd += [str(root / 'qoco_snapshot_replay.cu'), '-L', str(lib.parent), '-lqoco', '-o', str(binary)]
    report['build_command'] = cmd
    built = subprocess.run(cmd, env=env, text=True, capture_output=True)
    (root / 'build.log').write_text(built.stdout + built.stderr)
    built.check_returncode()
    report['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    with open('/home/angus/.spacepdhcg-gpu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for ruiz in [0, 2]:
            qp = root / f'ruiz{ruiz}.txt'
            lines = Path(f'build/performance/conditioning-qp-reg-v527/ruiz{ruiz}.txt').read_text().splitlines()
            ints = lines[2].split(); ints[3] = '1'; lines[2] = ' '.join(ints)
            qp.write_text('\n'.join(lines) + '\n')
            d = problem(qp)
            for direct in [False, True]:
                env.pop('QOCO_REPLAY_DIRECT_SETUP', None)
                if direct: env['QOCO_REPLAY_DIRECT_SETUP'] = '1'
                name = f'ruiz{ruiz}-direct{int(direct)}'
                log_path = root / (name + '.log')
                with log_path.open('x') as log:
                    child = subprocess.Popen([str(binary), str(qp)], env=env, stdout=log, stderr=subprocess.STDOUT)
                    report.update(stage=name, child_pid=child.pid); save()
                    result = child.wait(timeout=120)
                assert result == 0
                output = log_path.read_text().splitlines()
                scale = next(json.loads(s[len('QP_SCALING '):]) for s in output if s.startswith('QP_SCALING '))
                record = next(json.loads(s[10:]) for s in output if s.startswith('QP_REPLAY '))
                D,E,F = [sp.diags(scale[k]) for k in ['D','E','F']]
                expected = dict(P=sp.triu(scale['k'] * D @ d['P'] @ D + 1e-8 * sp.eye(d['n'])).tocsc(),
                                A=E @ d['A'] @ D, G=F @ d['G'] @ D,
                                c=scale['k'] * (D @ d['c']), b=E @ d['b'], h=F @ d['h'])
                errors = {}
                for key in ['P', 'A', 'G', 'c', 'b', 'h']:
                    if key in ['P', 'A', 'G']:
                        s = scale[key]
                        actual = sp.csc_matrix((s['x'], np.array(s['i'],int), np.array(s['p'],int)), shape=expected[key].shape)
                        difference = (actual - expected[key]).data
                        denominator = np.max(np.abs(expected[key].data), initial=0)
                    else:
                        difference = np.array(scale[key]) - expected[key]
                        denominator = np.max(np.abs(expected[key]), initial=0)
                    errors[key] = float(np.max(np.abs(difference), initial=0) / max(1e-300, denominator))
                row = dict(ruiz=ruiz,direct=direct,k=scale['k'],
                           ranges={k:[min(scale[k]),max(scale[k])] for k in ['D','E','F']},
                           scaled_data_relative_errors=errors, iterations=record['iterations'], audit=audit(d,record))
                report['cases'].append(row);save()
                print(name, 'matrix errors', errors, 'qualified', row['audit']['qualified'], flush=True)
    settings = clarabel.DefaultSettings();settings.verbose=False
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-11
    cones=[clarabel.ZeroConeT(d['p']),clarabel.NonnegativeConeT(d['l'])]+[clarabel.SecondOrderConeT(int(q)) for q in d['soc']]
    sol = clarabel.DefaultSolver(sp.triu(d['P']).tocsc(), d['c'], sp.vstack([d['A'],d['G']]).tocsc(), np.concatenate([d['b'],d['h']]), cones, settings).solve()
    answer=dict(status=1 if str(sol.status)=='Solved' else 0, x=sol.x, y=sol.z[:d['p']], z=sol.z[d['p']:], s=sol.s[d['p']:])
    (root / 'clarabel-solution.json').write_text(json.dumps(answer))
    report['clarabel'] = dict(version=clarabel.__version__,status=str(sol.status), audit=audit(d,answer))
    report['complete'] = True
except Exception as error:
    report['error'] = repr(error)
save()
