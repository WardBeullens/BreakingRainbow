import pickle
load('../Rainbow.sage')

q = 16
# toy parameters
n = 66 
m = 44
o2 = 22

# Round 2 SL 1 parameters
# n = 96
# m = 64
# o2 = 32

K = GF(q)

print("Do Keygen")
PK, O2, O1, W = Keygen(q,n,m,o2)

print("Write PK and SK")
pickle.dump( [q,n,m,o2,PK] , open( "pk.p", "wb" ) )
pickle.dump( [O2,O1,W] , open( "sk.p", "wb" ) )

