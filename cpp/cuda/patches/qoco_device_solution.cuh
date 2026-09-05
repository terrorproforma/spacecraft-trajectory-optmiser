// SPDX-License-Identifier: Apache-2.0
// Borrow completed, unscaled device vectors until the next solver operation.
// qoco_solve synchronously unscales and copies its solution before returning.
extern "C" int qoco_gpu_get_solution(QOCOSolver* solver, int n, int p, int m,
                                     const double** x, const double** y, const double** z)
{
    static_assert(sizeof(QOCOFloat) == sizeof(double), "device solution ABI requires FP64");
    if (!solver || !solver->work || !solver->sol || !x || !y || !z
        || (solver->sol->status != 1 && solver->sol->status != 2)) return 1;
    const auto* work = solver->work;
    if (work->data->n != n || work->data->p != p || work->data->m != m
        || !work->x || !work->y || !work->z) return 1;
    *x = work->x->d_data;
    *y = work->y->d_data;
    *z = work->z->d_data;
    return (!*x || (p && !*y) || (m && !*z)) ? 1 : 0;
}
