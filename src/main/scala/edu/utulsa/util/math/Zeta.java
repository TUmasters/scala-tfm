package edu.utulsa.util.math;

/** Basically a copy-paste of the CEPHA implementation of zeta function
 * https://github.com/jeremybarnes/cephes/blob/master/misc/zeta.c
 */
public class Zeta {
  public static final double[] A = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12, /*1.067062284288e16/3617*/
      1.1646782814350067249e14, /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17, /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18 /*1.6938241367317436694528e27/236364091*/
  };

  public static double zeta(double x, double q) {
    if(x == 1)
      return Double.POSITIVE_INFINITY;
    else if(x < 1)
      throw new IllegalArgumentException("Domain error: x cannot be less than 1.0");
    else if(q < 0) {
      if(q == Math.floor(q)) return Double.POSITIVE_INFINITY;
      if(x != Math.floor(x)) throw new IllegalArgumentException("Domain error: q^-x undefined");
    }

    double s = Math.pow(q, -x);
    double a = q;
    int i = 0;
    double b = 0;

    while(i < 9 || a < 9.0) {
      i += 1;
      a += 1.0;
      b = Math.pow(a, -x);
      s += b;
      if(Math.abs(b/s) < 2e-16) return s;
    }

    double w = a;
    s += b*w/(x-1);
    s -= 0.5 * b;
    a = 1.0;
    double k = 0;
    for(i=0; i < 12; i++) {
      a *= x + k;
      b /= w;
      double t = a*b/A[i];
      s = s + t;
      t = Math.abs(t/s);
      if( t < 2e-16 ) return s;
      k += 1.0;
      a *= x + k;
      b /= w;
      k += 1.0;
    }

    return s;
  }
}
