__global__ void compute_nt_scaling_kernel(QOCOFloat* W, QOCOFloat* WtW,
                                          QOCOFloat* nt_scaling,
                                          QOCOFloat* Winv, QOCOFloat* s,
                                          QOCOFloat* z, QOCOFloat* sbar,
                                          QOCOFloat* zbar, QOCOInt l,
                                          QOCOInt nsoc, const QOCOInt* q)
{
  QOCOInt tid = blockIdx.x * blockDim.x + threadIdx.x;

  /* ================= LP cone ================= */
  if (tid < l) {
    QOCOFloat val = safe_div(s[tid], z[tid]);
    WtW[tid] = val;
    QOCOFloat w = qoco_sqrt(val);

    W[tid] = w;
    nt_scaling[tid] = w;

    QOCOFloat winv = safe_div((QOCOFloat)1.0, w);
    Winv[tid] = winv;
    return;
  }

  /* ================= SOC cones ================= */
  QOCOInt soc_id = tid - l;
  if (soc_id >= nsoc)
    return;

  /* ---- compute SOC offsets ---- */
  QOCOInt idx = l;
  QOCOInt nt_idx = l;
  QOCOInt nt_idx_fast = l;

  for (QOCOInt k = 0; k < soc_id; ++k) {
    idx += q[k];
    nt_idx += (q[k] * q[k] + q[k]) / 2;
    nt_idx_fast += q[k] + 1;
  }

  QOCOInt qi = q[soc_id];


  // Keep normalization and cancellation in wbar in double-double until storage.
  using namespace qoco_cone_arithmetic;
  DD ss=root(inner(&s[idx],&s[idx],qi));
  DD zz=root(inner(&z[idx],&z[idx],qi));
  DD gamma=root(mul(DD(0.5),add(DD(1.0),
      divide(euclidean(&s[idx],&z[idx],qi),mul(ss,zz)))));
  DD denominator=mul(DD(2.0),gamma);
  for(int j=0;j<qi;++j) {
    DD sn=divide(DD(s[idx+j]),ss), zn=divide(DD(z[idx+j]),zz);
    sbar[idx+j]=rounded(divide(add(sn,j==0?zn:neg(zn)),denominator));
  }
  QOCOFloat eta=rounded(root(divide(ss,zz)));
  QOCOFloat finv=safe_div((QOCOFloat)1.0,eta);
  QOCOFloat eta2=eta*eta;
  QOCOFloat f;

  /* Store compact fast scaling parameters [eta, w0, w1[0], ..., w1[qi-2]].
   * sbar currently holds the wbar vector. */
  nt_scaling[nt_idx_fast] = eta;
  nt_scaling[nt_idx_fast + 1] = sbar[idx + 0];
  for (QOCOInt j = 1; j < qi; ++j)
    nt_scaling[nt_idx_fast + 1 + j] = sbar[idx + j];

  /* overwrite zbar with v (needed for the sparse W / Winv blocks) */
  f = safe_div((QOCOFloat)1.0,
               qoco_sqrt((QOCOFloat)2.0 * (sbar[idx + 0] + (QOCOFloat)1.0)));

  zbar[idx + 0] = f * (sbar[idx + 0] + (QOCOFloat)1.0);
  for (QOCOInt j = 1; j < qi; ++j)
    zbar[idx + j] = f * sbar[idx + j];

  /* --- build W, Winv (sparse upper triangular) and WtW = eta^2 (2 w w' - J)
   * --- */
  QOCOInt shift = 0;
  for (QOCOInt j = 0; j < qi; ++j) {
    for (QOCOInt k = 0; k <= j; ++k) {

      QOCOFloat val = (QOCOFloat)2.0 * zbar[idx + k] * zbar[idx + j];
      QOCOFloat winv_val = val;

      if (j != 0 && k == 0)
        winv_val = -val;

      if (j == 0 && k == 0) {
        val -= (QOCOFloat)1.0;
        winv_val -= (QOCOFloat)1.0;
      }
      else if (j == k) {
        val += (QOCOFloat)1.0;
        winv_val += (QOCOFloat)1.0;
      }

      val *= eta;
      winv_val *= finv;

      W[nt_idx + shift] = val;
      Winv[nt_idx + shift] = winv_val;

      QOCOFloat wtw = eta2 * (QOCOFloat)2.0 * sbar[idx + j] * sbar[idx + k];
      if (j == k && j == 0)
        wtw -= eta2;
      else if (j == k)
        wtw += eta2;
      WtW[nt_idx + shift] = wtw;

      shift++;
    }
  }
}

