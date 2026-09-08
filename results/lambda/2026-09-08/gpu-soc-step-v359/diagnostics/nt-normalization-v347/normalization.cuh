
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

