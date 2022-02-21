import itertools

load('Rainbow.sage')

# Table 2
#q,n,m,o2 = 31, 30, 20, 10
#q,n,m,o2 = 31, 45, 30, 15
#q,n,m,o2 = 31, 60, 40, 20

# Table 3
#q,n,m,o2 = 16, 30, 20, 10
#q,n,m,o2 = 16, 36, 24, 12
q,n,m,o2 = 16, 42, 28, 14


K = GF(q)

attempts = 0

basis_Fn = (K**n).basis()
basis_Fm = (K**m).basis()

# if you run without O2, the system is not guaranteed to have a solution
def Attack(PK, O2 = None):
    global attempts

    # pick a random vector x
    x = vector([K.random_element() for i in range(n)])
    while Eval(PK,x)[0] == 0:
        x = vector([K.random_element() for i in range(n)])

    # compute linear map D_x = P'(x,.)
    D_x = Matrix(K, [ Differential(PK,x,b) for b in basis_Fn ] )
    D_x_ker = Matrix(D_x.kernel().basis())

    if q%2 == 0:
        D_x_ker[0] = x

    if D_x_ker.rank() != n-m:
            return Attack(PK,O2)

    attempts += 1

    Sol = None
    if not O2 is None:
        V = K**n
        I = V.span(D_x_ker).intersection(V.span(O2.transpose()))
        if I.dimension() == 0:
            print("Attack would fail. resample x")
            return Attack(PK,O2)

        print("Intersection has dimension:", I.dimension())
        Sol = I.basis()[0]

        Sol = D_x_ker.transpose().solve_right(Sol)

        if Sol[-1] == 0:
            print("last entry is zero, resample x")
            return Attack(PK,O2)

        Sol = Sol/Sol[-1]

        print("Good D_x found after %d attempts." % attempts)

        print("The expected solution is:")
        print(Sol)

    # Compose smaller system D_x(o)= 0 and P(o) = 0
    SS = [ D_x_ker*M*D_x_ker.transpose() for M in PK ]
    for s in SS:
        Make_UD(s)

    if not Sol is None:
        print("Sanity check: evaluation of sol")
        print(Eval(SS, Sol))

    if q % 2 == 0:
        Px = Eval(PK,x)
        SSS = [ (SS[i]*Px[0] + SS[0]*Px[i])[1:,1:] for i in range(1,len(SS)) ]

        if not Sol is None:
            print("Sanity check: evaluation of sol[1:]")
            print(Eval(SSS, Sol[1:]))
        
        SS = SSS

    return SS

PK, O2, O1, W = Keygen(q,n,m,o2)
print('O2')
print(O2)
print('O1')
print(O1)
print('W')
print(W)
tP = Attack(PK,O2)

N = tP[0].ncols()
M = len(tP)

#tP = [ Matrix(N,N,[ K.random_element() for _ in range(N*N) ])  for _ in tP]

print("M = %d, N = %d" % (M,N))
PR = PolynomialRing(K,N,'x')
PR.inject_variables()

x_vec = vector(PR,[PR.gens()])
tP = [ x_vec*M*x_vec for M in tP ]

# Compute all monomials of a certain degree
def Monomials(vars,degree):
    if degree < 0:
        return
    
    for comb in itertools.combinations_with_replacement(vars,degree):
        u = 1
        for var in comb:
            u *= var
        yield u
    return


L = 10
Expected_Ranks = [0] * L
Expected_Ranks[0] = 1

for _ in range(N):
    for i in range(1,L):
        Expected_Ranks[i] += Expected_Ranks[i-1]

NumberOfMonomials = [x for x in Expected_Ranks]

for _ in range(M):
    for i in range(L-1,1,-1):
        Expected_Ranks[i] -= Expected_Ranks[i-2]

Expected_Ranks = [ NumberOfMonomials[i] - Expected_Ranks[i] for i in range(L) ]

print("Cols and Expected_Ranks:")
print(NumberOfMonomials)
print(Expected_Ranks)

for D in range(2,5):
    eqns = []
    for p in tP:
        for Mon in Monomials(PR.gens(), D-2):
            eqns.append(p*Mon)
    
    s = Sequence(eqns)
    M, Mon = s.coefficient_matrix()
    Rank = M.rank()
    rows = M.nrows()
    cols = M.ncols()
    print("D = ", D)
    if Rank == cols-1:
        print("rank: %d, rows: %d, cols: %d, SOLVED"%(Rank,rows,cols))
    else:
        print("rank: %d, rows: %d, cols: %d"%(Rank,rows,cols))
    print("")