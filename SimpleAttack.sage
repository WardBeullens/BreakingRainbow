# this script generates a public key, make a *good* guess (we 'cheat' by using the sk). And writes the P-tilde system to a file 
# in a format that is readable by the Block Wiedemann XL algorihm of Niederhagen


def elt_to_str(a):
    if q == 16:
        return str(hex(sum([2**i * a.polynomial()[i].lift() for i in range(4)])))[2:]
    return str(a)

D = { elt_to_str(a): a for a in K }

def str_to_elt(str):
    return D[str]   

basis_Fn = (K**n).basis()
basis_Fm = (K**m).basis()

def UD_to_string(M):
    S = ""
    for i in range(M.ncols()):
        for j in range(i+1):
            S += elt_to_str(M[j,i]) + ' '
    S += ';\n'
    return S

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

    fname = 'system'+str(len(SS))+'-'+str(SS[0].ncols()-1)+'.txt'
    f = open(fname,'w')
    for s in SS:
        f.write(UD_to_string(s))
    f.close()
    print("System written to: " + fname )
    print("Use block Wiedemann XL algorithm of Niederhagen to find a solution:")
    print("http://polycephaly.org/projects/xl")
    return x,fname,D_x_ker

#load('Rainbow.sage')
#q = 16
#K = GF(q)
#n = 96
#m = 64
#o2 = 32
#attempts = 0
#PK, O2, O1, W = Keygen(q,n,m,o2)
#Attack(PK,O2)
