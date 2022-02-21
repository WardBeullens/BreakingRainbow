from itertools import combinations, combinations_with_replacement

load("Rainbow.sage")

# Table 4
q, n, m, o2, m_prime = 31, 15, 10, 5, 8
# q, n, m, o2, m_prime = 16, 15, 10, 5, 8
# q, n, m, o2, m_prime = 31, 15, 10, 5, 7
# q, n, m, o2, m_prime = 16, 15, 10, 5, 7
# q, n, m, o2, m_prime = 31, 14,  6, 4, 6
# q, n, m, o2, m_prime = 16, 14,  6, 4, 6


print("q,n,m,o2,m_prime:",q,n,m,o2, m_prime)

columns_to_keep = m_prime
rank = o2
K = GF(q)
attempts = 0

# generates a MinRank instance based on PK
def Attack(PK, O2):
    global attempts
    attempts += 1

    # pick a random vector x
    x = vector([K.random_element() for i in range(n)])

    # compute Differential D_x
    D_x = Matrix(K, [ Differential(PK,x,b) for b in (K**n).basis() ] )
    
    D_x_ker = Matrix(D_x.kernel().basis())

    V = K**n
    I = D_x.kernel().intersection(V.span(O2.transpose()))
    if I.dimension() != 1:
        return Attack(PK,O2)

    print("Good guess after %d attempts." % attempts)

    # Compose MinRank instance 
    Matrices = []
    for b in D_x.kernel().basis():
        Li = Matrix([ Differential(PK, e_i, b) for e_i in (K**n).basis()])
        Matrices.append(Li) 

    return Matrices

PK,O2,O1,W = Keygen(q,n,m,o2)
matrices = Attack(PK,O2)

#remove top row of matrices
matrices = [ M[1:,:] for M in matrices ]

nmatrices = len(matrices)
rows = matrices[0].nrows()

var_names = [ "x"+str(i) for i in range(nmatrices) ] + [ "m"+str(i) for i in range(binomial(columns_to_keep,rank)) ]
PR = PolynomialRing(K, var_names )

x_vars = list(PR.gens()[:nmatrices])

# Dictionary for m_variables
m_vars = {}
ctr = 0
for comb in combinations(list(range(columns_to_keep)), rank):
    m_vars[comb] = PR.gens()[nmatrices+ctr]
    ctr += 1

# construct the support-minors modelling equations
bilinear_eqns = []
for comb in combinations(list(range(columns_to_keep)), rank+1):
    for row in range(rows):
        eqn = 0
        sign = 1
        for i in range(rank+1):
            new_tuple = comb[:i] + comb[(i+1):]
            eqn += sign*m_vars[new_tuple]*sum( [ x_vars[j]*matrices[j][row][comb[i]] for j in range(nmatrices) ])
            sign *= -1
        bilinear_eqns.append(eqn)

# computes expected rank based on formula of Bardet et al
def ExpectedRank(rows,cols,nmatrices,rank,max_b):
    max_b += 1

    Monomials = [ [0, 0 ] for i in range(max_b) ]
    Monomials[0][0] = 1

    for _ in range(nmatrices):
        for i in range(1,max_b):
            Monomials[i][0] += Monomials[i-1][0]

    for i in range(max_b):
        Monomials[i][1] = Monomials[i][0]*binomial(cols,rank)

    GF = [x[:] for x in Monomials]

    #Formula from bardet et al.
    for b in range(max_b):
        GF[b][1] = 0
        for i in range(b+1):
            GF[b][1] += (-1)**i * binomial(cols,rank+i) * binomial(rows+i-1,i) * binomial(nmatrices +b -i -1, b-i) 

    return [Monomials[i][1]-GF[i][1] for i in range(max_b)], [Monomials[i][1] for i in range(max_b)]

# Compute all monomials of a certain degree
def Monomials(vars,degree):
    if degree < 0:
        return
    
    for comb in combinations_with_replacement(vars,degree):
        u = 1
        for var in comb:
            u *= var
        yield u
    return

# Compute the ranks of an XL system
def computeXLRanks(bil,x_vars,v_vars,b):
    eqns = []
    for p in bil:
        for mon in Monomials(x_vars,b-1):
            eqns.append(p*mon)
    
    s = Sequence(eqns)
    if len(eqns)>0:
        M, Mon = s.coefficient_matrix()
        Rank = M.rank()
        rows = M.nrows()
        cols = M.ncols()
        if Rank == cols-1:
            print("rank: %d, rows: %d, cols: %d, SOLVED"%(Rank,rows,cols))
        else:
            print("rank: %d, rows: %d, cols: %d"%(Rank,rows,cols))
        return Rank, cols
    else:
        print("no equations")
    return 0,0


print("MinRank instance has %d matrices of %d-by-%d and solution has rank %d" % (nmatrices, rows, columns_to_keep, rank))

ranks, mons = ExpectedRank(rows,columns_to_keep,nmatrices,rank,10) 

print()
print("--- RANK EXPERIMENTS: ---")

for b in range(1,4):
    print("b: %d"%b)
    print("predicted: %d, %d"%(ranks[b],mons[b]))
    rank, mon  = computeXLRanks(bilinear_eqns, x_vars, list(m_vars.values()), b)
    print(" ")

