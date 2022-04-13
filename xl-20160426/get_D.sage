import argparse

def ps(q, m, n):
  R.<X> = PowerSeriesRing(ZZ)

  if(q == 2):
    return (1+X)^n/((1-X)*(1+X^2)^m)
  else:
    return (1-X)^(m-n-1)*(1+X)^m


def deg_info(q, m, n):
  f_XL = ps(q, m, n)

  for i, coef in enumerate(f_XL):
    if coef <= 0:
      return i

  return -1


parser = argparse.ArgumentParser(description='',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-q', dest='q',
                   type=int, required=True,
                   help='field')

parser.add_argument('-m', dest='m',
                   type=int, required=True,
                   help='M')

parser.add_argument('-n', dest='n',
                   type=int, required=True,
                   help='N')

parser.add_argument("-s", "--stat", help="print stat", action="store_true")

options = parser.parse_args()

deg = deg_info(q = options.q, m = options.m, n = options.n)


if not options.stat:
  print(deg)
else:

  def num_mons_sum(q, n, d):
      if(q == 2):
          sum = 0;
          for i in range(d + 1):
              sum += binomial(n, i);
      else:
          sum = binomial(n + d, d);
  
      return sum;

  if options.q == 2:
    BW_N = 512
    BW_M = 512
  else:
    BW_N = 128
    BW_M = 128
  
  n = options.n
  m = options.m
  
  Mac_width = num_mons_sum(options.q, n, deg)
  orig_sys_m = num_mons_sum(options.q, n, 2)
  
  num_it = Mac_width / BW_M + Mac_width / BW_N + 8

  bm = (num_it*num_it) / 4.0 * (float)(2*BW_N*BW_N*BW_N)
  
  bw1 = (Mac_width * orig_sys_m + (BW_N * BW_N)) * BW_N * num_it
  
  print("%i %.0f %.0f %.02f %i %i" % (m, bw1, bm, bw1/bm, orig_sys_m, deg) )

