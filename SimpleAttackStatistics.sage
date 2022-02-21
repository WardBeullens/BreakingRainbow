
# This script generates a key pair, and makes many guesses for D_x, and counts how many are good.
# The secret key is used to efficiently check if a D_x is good or not

load('Rainbow.sage')

q = 16
K = GF(q)

n  = 96
m  = 64
o2 = 32 

attempts = 0
successes = 0

basis_Fn = (K**n).basis()

def Guess(PK, O2):
    global attempts, successes

    print("successes/attempts: %d / %d" % (successes,attempts))

    # pick a random vector x
    x = vector([K.random_element() for i in range(n)])
    while Eval(PK,x)[0] == 0:
        x = vector([K.random_element() for i in range(n)])

    # compute linear map D_x = P'(x,.)
    D_x = Matrix(K, [ Differential(PK,x,b) for b in basis_Fn ] )

    D_x_ker = Matrix(D_x.kernel().basis())

    if D_x_ker.rank() != n-m:
            print("Kernel too big, resample x")
            return Attack(PK,O2)

    attempts += 1
    
    # check if ker(D_x) intersects O2
    V = K**n
    I = V.span(D_x_ker).intersection(V.span(O2.transpose()))
    if I.dimension() > 0:
        successes += 1

PK, O2, O1, W = Keygen(q,n,m,o2)
while True:
    Guess(PK, O2)
