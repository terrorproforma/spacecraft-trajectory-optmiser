#pragma once
#include "spacepdhcg/cuda/gtoc12_qoco_c_api.h"
#include "spacepdhcg/cuda/gtoc12_discretisation_c_api.h"

// Internal, synchronous rebinding. Caller owns the workspace exclusively and
// must have destroyed all graphs and drained all launches borrowing its data.
int gtoc12_discretisation_rebind(spacepdhcg_gtoc12_discretisation*,double,double,const double*);
int gtoc12_conic_rebind(spacepdhcg_gtoc12_conic*,double,double,const double*,const double*,const double*);
int gtoc12_qoco_acquire(int,int,int,int,double,double,const double*,const double*,const double*,
    double,int,spacepdhcg_gtoc12_qoco**);
void gtoc12_qoco_release(spacepdhcg_gtoc12_qoco*,bool);
