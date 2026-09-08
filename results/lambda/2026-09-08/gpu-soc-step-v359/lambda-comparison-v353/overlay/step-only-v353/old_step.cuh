__device__ QOCOFloat soc_step_length_dev(const QOCOFloat* x,
                                         const QOCOFloat* dx, QOCOInt n,
                                         QOCOFloat alpha_max)
{
  const QOCOFloat two = 2.0;
  const QOCOFloat four = 4.0;

  QOCOFloat alpha = alpha_max;

  // ----------------------------------
  // Scalar safeguard: x0 + alpha dx0 >= 0
  // ----------------------------------
  if (x[0] >= 0.0 && dx[0] < 0.0) {
    QOCOFloat a = -x[0] / dx[0];
    if (a < alpha)
      alpha = a;
  }

  // ----------------------------------
  // Compute quadratic coefficients
  // ----------------------------------

  // a = dx0^2 - ||dx1||^2
  QOCOFloat dx1_norm2 = 0.0;
  for (QOCOInt i = 1; i < n; ++i)
    dx1_norm2 += dx[i] * dx[i];
  QOCOFloat a = dx[0] * dx[0] - dx1_norm2;

  // b = 2*(x0*dx0 - x1^T dx1)
  QOCOFloat x1dx1 = 0.0;
  for (QOCOInt i = 1; i < n; ++i)
    x1dx1 += x[i] * dx[i];
  QOCOFloat b = two * (x[0] * dx[0] - x1dx1);

  // c = x0^2 - ||x1||^2
  QOCOFloat x1_norm2 = 0.0;
  for (QOCOInt i = 1; i < n; ++i)
    x1_norm2 += x[i] * x[i];
  QOCOFloat c = x[0] * x[0] - x1_norm2;

  if (c < 0.0)
    c = 0.0; // numerical safeguard

  // ----------------------------------
  // Discriminant
  // ----------------------------------
  QOCOFloat d = b * b - four * a * c;

  // ----------------------------------
  // Case analysis (same as Clarabel)
  // ----------------------------------

  // No positive root → no restriction
  if ((a > 0.0 && b > 0.0) || d < 0.0)
    return alpha;

  // Linear case
  if (a == 0.0)
    return b < 0.0 ? qoco_min(alpha, -c / b) : alpha;

  // Boundary case
  if (c == 0.0) {
    if (b < 0.0 || (b == 0.0 && a < 0.0)) return 0.0;
    return a < 0.0 ? qoco_min(alpha, -b / a) : alpha;
  }

  // ----------------------------------
  // Stable quadratic root computation
  // ----------------------------------
  QOCOFloat sqrt_d = qoco_sqrt(d);

  QOCOFloat t;
  if (b >= 0.0)
    t = -b - sqrt_d;
  else
    t = -b + sqrt_d;

  QOCOFloat r1 = (two * c) / t;
  QOCOFloat r2 = t / (two * a);

  // Keep only positive roots
  if (r1 < 0.0)
    r1 = QOCOFloat_MAX;
  if (r2 < 0.0)
    r2 = QOCOFloat_MAX;

  QOCOFloat r = qoco_min(r1, r2);

  if (r < alpha)
    alpha = r;

  return alpha;
}

