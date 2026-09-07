// Exercise the actual device controller independently of QOCO's convergence.
#include "../src/gtoc12_scvx.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
#define CUDA(x) REQUIRE((x)==cudaSuccess)

Settings fixture() {
    Settings p{};
    p.substeps=8; p.polish_substeps=16; p.polish_iterations=4; p.max_iterations=40;
    p.virtual_weight=1e4; p.initial_trust_state=.2; p.initial_trust_control=1;
    p.minimum_trust=1e-6; p.maximum_trust=2; p.ratio_shrink=.25; p.ratio_grow=.7;
    p.shrink_factor=.5; p.grow_factor=1.6; p.defect_tolerance=5e-9;
    p.step_tolerance=1e-7; p.objective_tolerance=1e-6; p.conic_tolerance=1e-9;
    p.minimum_mass=.3; p.radius_floor=.05; p.vinf_max=.2; p.time_limit_s=30;
    return p;
}

void controller() {
    State* ds; Metrics* dm; double *dx,*states,*controls;
    Record* records; spacepdhcg_gtoc12_conic_parameters* parameters;
    spacepdhcg_gtoc12_qoco_report* device_report;
    CUDA(cudaMalloc(&device_report,sizeof(*device_report)));
    CUDA(cudaMalloc(&ds,sizeof(State))); CUDA(cudaMalloc(&dm,sizeof(Metrics)));
    CUDA(cudaMalloc(&dx,100*sizeof(double))); CUDA(cudaMalloc(&states,28*sizeof(double)));
    CUDA(cudaMalloc(&controls,16*sizeof(double))); CUDA(cudaMalloc(&records,44*sizeof(Record)));
    CUDA(cudaMalloc(&parameters,sizeof(*parameters)));
    std::vector<double> x(100,2.0), got(28), zeros(28,0.0);
    CUDA(cudaMemcpy(dx,x.data(),800,cudaMemcpyHostToDevice));
    for (bool device:{false,true}) for (int test=0;test<14;++test) {
        auto p=fixture();
        State host{}; host.command.substeps=8; host.merit=10; host.trust_state=.2;
        host.trust_control=1; host.polish_left=4; host.result.virtual_inf=INFINITY;
        Metrics metrics{1,0,0,0,0,1,0};
        int qualified=1;
        if (test==0) qualified=0;
        if (test==1) { qualified=0; host.trust_state=1e-6; }
        if (test==2) metrics.penalty=.002; // actual negative, predicted positive: reject
        if (test==3) metrics.penalty=.00085; // ratio 1/18: shrink accepted
        if (test==4) metrics.step=0; // enter polish
        if (test==5) { metrics.step=0; p.polish_iterations=0; }
        if (test==6) { metrics.step=0; host.polishing=1; }
        if (test==7) metrics.invalid=1;
        if (test==8) { metrics.fuel=10; metrics.penalty=0; } // zero prediction, ratio=1
        if (test==9) { metrics.fuel=10; metrics.penalty=.001; } // zero prediction, ratio=-1
        if (test==10) { host.result.iterations=39; metrics.defect=1e-4; }
        if (test==11) { host.polishing=1; host.polish_left=1; metrics.penalty=.002; }
        if (test==12) { host.trust_state=host.trust_control=1e-6; metrics.penalty=.002; }
        if (test==13) { metrics.virtual_sum=NAN; }
        const int prior=host.result.iterations;
        CUDA(cudaMemcpy(ds,&host,sizeof(host),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(dm,&metrics,sizeof(metrics),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(states,zeros.data(),28*sizeof(double),cudaMemcpyHostToDevice));
        CUDA(cudaMemset(controls,0,16*sizeof(double)));
        spacepdhcg_gtoc12_qoco_report report{};
        report.qualified=qualified; report.qoco_status=2;
        CUDA(cudaMemcpy(device_report,&report,sizeof(report),cudaMemcpyHostToDevice));
        // Opposite host qualification and status prove the device packet wins.
        decide<<<1,1>>>(ds,p,dm,dx,4,1,1,device ? !qualified : qualified,1,records,parameters,
            device ? device_report : nullptr);
        accept_candidate<<<2,256>>>(ds,4,dx,states,controls);
        CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
        CUDA(cudaMemcpy(&host,ds,sizeof(host),cudaMemcpyDeviceToHost));
        Record record{}; CUDA(cudaMemcpy(&record,records+prior,sizeof(record),cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(got.data(),states,28*sizeof(double),cudaMemcpyDeviceToHost));
        const bool accepted=test==3 || test==4 || test==5 || test==6 || test==8 || test==10;
        REQUIRE(record.accepted==accepted); REQUIRE(host.result.iterations==prior+1);
        REQUIRE(record.qoco_status==(device ? 2 : 1));
        for (double value:got) REQUIRE(value==(accepted ? 2.0 : 0.0));
        if (test==0 || test==7 || test==13) REQUIRE(record.conic_rejected && host.trust_state==.1);
        if (test==1) REQUIRE(host.command.done && host.result.status==2);
        if (test==2 || test==3) REQUIRE(host.trust_state==.1);
        if (test==4) REQUIRE(host.command.refresh && host.command.substeps==16 && host.polishing);
        if (test==5 || test==6) REQUIRE(host.command.done && host.result.status==1);
        if (test==8) REQUIRE(record.ratio==1 && std::abs(host.trust_state-.32)<1e-15);
        if (test==9) REQUIRE(record.ratio==-1);
        if (test==10 || test==11) REQUIRE(host.command.done && host.result.status==0);
        if (test==12) REQUIRE(host.command.done && host.result.diagnostic==2);
        if (accepted) for (int j=0;j<3;++j) REQUIRE(host.result.departure_vinf[j]==2 && host.result.arrival_vinf[j]==2);
    }
    for (int status=0;status<5;++status) for (int feasible=0;feasible<2;++feasible) {
        State host{}; host.result.status=status; host.result.virtual_inf=feasible ? 0 : 1;
        CUDA(cudaMemcpy(ds,&host,sizeof(host),cudaMemcpyHostToDevice));
        finalize<<<1,1>>>(ds,fixture(),0);
        CUDA(cudaMemcpy(&host,ds,sizeof(host),cudaMemcpyDeviceToHost));
        const int expected=status==0 || status==2 ? (feasible ? 1 : 3) : status;
        REQUIRE(host.result.status==expected);
    }
    CUDA(cudaFree(ds)); CUDA(cudaFree(dm)); CUDA(cudaFree(dx)); CUDA(cudaFree(states));
    CUDA(cudaFree(controls)); CUDA(cudaFree(records)); CUDA(cudaFree(parameters));
    CUDA(cudaFree(device_report));
}

void reductions(int nodes,bool poison) {
    const int variables=25*nodes-14, blocks=std::min(256,(variables+255)/256);
    std::vector<double> x(variables,0),ref(7*nodes,0),u(4*nodes,0),weights(nodes,.001),prop(7*(nodes-1),0);
    Metrics expected{};
    for (int i=0;i<7*nodes;++i) x[i]=.01*(i%17-8);
    for (int i=0;i<4*nodes;++i) x[7*nodes+i]=.02*(i%7);
    for (int i=0;i<7*(nodes-1);++i) x[11*nodes+i]=.003*(i%11-5);
    for (int i=0;i<nodes;++i) expected.fuel+=weights[i]*x[7*nodes+4*i+3];
    for (int i=0;i<7*(nodes-1);++i) {
        const double defect=std::abs(x[7+i]);
        expected.penalty+=std::max(defect-1e-9,0.0); expected.defect=std::max(expected.defect,defect);
        expected.virtual_sum+=std::abs(x[11*nodes+i]); expected.virtual_inf=std::max(expected.virtual_inf,std::abs(x[11*nodes+i]));
    }
    for (int i=0;i<11*nodes;++i) expected.step=std::max(expected.step,std::abs(x[i]));
    if (poison) x.back()=NAN; // nonfinite slack outside the state/control/virtual slices
    double *dx,*dr,*du,*df,*dp; int* invalid; Metrics *partial,*out;
    CUDA(cudaMalloc(&dx,x.size()*8)); CUDA(cudaMalloc(&dr,ref.size()*8)); CUDA(cudaMalloc(&du,u.size()*8));
    CUDA(cudaMalloc(&df,weights.size()*8)); CUDA(cudaMalloc(&dp,prop.size()*8));
    CUDA(cudaMalloc(&invalid,4)); CUDA(cudaMalloc(&partial,blocks*sizeof(Metrics))); CUDA(cudaMalloc(&out,sizeof(Metrics)));
    CUDA(cudaMemcpy(dx,x.data(),x.size()*8,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dr,ref.data(),ref.size()*8,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(du,u.data(),u.size()*8,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(df,weights.data(),weights.size()*8,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dp,prop.data(),prop.size()*8,cudaMemcpyHostToDevice)); CUDA(cudaMemset(invalid,0,4));
    reduce_metrics<<<blocks,256>>>(nodes,variables,dx,dx+7*nodes,dx+11*nodes,dr,du,dp,invalid,df,1e-9,nodes,partial);
    finish_metrics<<<1,256>>>(blocks,partial,out);
    Metrics got{}; CUDA(cudaMemcpy(&got,out,sizeof(got),cudaMemcpyDeviceToHost));
    REQUIRE(std::abs(got.fuel-expected.fuel)<1e-11*std::max(1.0,expected.fuel));
    REQUIRE(std::abs(got.penalty-expected.penalty)<1e-11*std::max(1.0,expected.penalty));
    REQUIRE(std::abs(got.virtual_sum-expected.virtual_sum)<1e-11*std::max(1.0,expected.virtual_sum));
    REQUIRE(got.defect==expected.defect && got.virtual_inf==expected.virtual_inf && got.step==expected.step);
    REQUIRE(got.invalid==double(poison));
    if (!poison) for (int test=0;test<6;++test) {
        auto input=x;
        const int node=test>=3 ? nodes-1 : 0;
        const double thrust=test==0 ? .6 : test==1 ? .6000000005 : .6000000038;
        input[7*nodes+4*node]=thrust/.6;
        input[7*nodes+4*node+1]=input[7*nodes+4*node+2]=0;
        CUDA(cudaMemcpy(dx,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        // Inactive ZOH endpoint and infeasible initial references are allowed;
        // active violating candidate nodes are rejected even with a finite CQP.
        const int active=test==3 ? nodes-1 : nodes;
        reduce_metrics<<<blocks,256>>>(nodes,variables,dx,dx+7*nodes,dx+11*nodes,dr,
            test==5 ? nullptr : du,dp,invalid,df,1e-9,active,partial);
        finish_metrics<<<1,256>>>(blocks,partial,out);
        CUDA(cudaMemcpy(&got,out,sizeof(got),cudaMemcpyDeviceToHost));
        REQUIRE(got.invalid==double(test==2 || test==4));
    }
    CUDA(cudaFree(dx)); CUDA(cudaFree(dr)); CUDA(cudaFree(du)); CUDA(cudaFree(df)); CUDA(cudaFree(dp));
    CUDA(cudaFree(invalid)); CUDA(cudaFree(partial)); CUDA(cudaFree(out));
}

int main() {
    controller();
    for (int n:{4,37,4097,10001}) for (bool poison:{false,true}) reductions(n,poison);
    std::puts("PASS: 14 controller branches, 10 final states, 8 multi-block reductions, 24 physical thrust gates");
}
