# implements Rainbow Key generation

# makes a matrix upper diagonal
def Make_UD(M):
    n = M.ncols()
    for i in range(n):
        for j in range(i+1,n):
            M[i,j] += M[j,i]
            M[j,i] = K(0) 

# generates a Rainbow public key
def Keygen(q,n,m,o2):
    K = GF(q)

    Central_Map = [];
    # first layer
    for k in range(m-o2):
        P  = Matrix(K,n,n)
        for i in range(n-m):
            for j in range(i,n-o2):
                P[i,j] = K.random_element()
        Central_Map.append(P)
    # second layer
    for k in range(o2):
        P  = Matrix(K,n,n)
        for i in range(n-o2):
            for j in range(i,n):
                P[i,j] = K.random_element()
        Central_Map.append(P)

    T = Matrix(K,m,m)
    while not T.is_invertible():
        T = Matrix(K,[ [K.random_element() for j in range(m)] for i in range(m)] )

    S = Matrix(K,n,n)
    while not S.is_invertible():
        S = Matrix(K,[ [K.random_element() for j in range(n)] for i in range(n)] )

    Pre_Public_Key = [ S.transpose()*M*S for M in Central_Map ]

    Public_Key = []
    for i in range(m):
        P = Matrix(K,n,n)
        for j in range(m):
            P += T[i,j]*Pre_Public_Key[j]
        Make_UD(P)
        Public_Key.append(P)

    
    basis_Fn = (K**n).basis()
    basis_Fm = (K**m).basis()
    
    O1 = S.inverse() * Matrix(K, [ basis_Fn[i] for i in range(n-m,n) ] ).transpose()
    O2 = S.inverse() * Matrix(K, [ basis_Fn[i] for i in range(n-o2,n) ] ).transpose()
    W =  T * Matrix(K, [ basis_Fm[i] for i in range(m-o2,m) ] ).transpose()

    return Public_Key, O2, O1, W

# evaluate Multivariate map and Differential
def Eval(F,x):
    return vector([ x*M*x for M in F])

def Differential(F,x,y):
    return vector([ (x*M*y) + (y*M*x)  for M in F ])

