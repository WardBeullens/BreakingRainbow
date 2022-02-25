# BreakingRainbow
Scripts for the 'Breaking Rainbow takes a Weekend on a Laptop' paper.

We have the following scripts: 

- SimpleAttack.sage: Contains an implementation of the simple attack. Only constructs the \hat{P} system. You can use the block wiedemann XL implementation of Niederhagen to find a solution. http://polycephaly.org/projects/xl
- SimpleAttackRankExp.sage: Script to reproduce tables 2 and 3.
- CombinedAttackRankExp.sage: Script to reproduce table 4.
- SimpleAttackStatistics.sage: Counts how many guesses are good.
- Rainbow.sage: Used by the other scripts to generate a Rainbow public key


## using the wiedemann XL implmentation of Niederhagen et al.
The SimpleAttack.sage script creates a file named 'systemN-M.txt' for some values of N, M. You need to compile the wiedemann XL implementation with these values of N,M.
For example, the NIST SL 1 parameters result in 'system63-30.txt'. (N is one less than the value N in the paper, because the homogenous system with N variables is converted to a inhomogenous system by putting x_N = 1.) 

So in the XL Makefile you should set 

    Q = 16
    M = 63
    N = 30

You can then run the XL algorithm with:

    ./xl --challenge system63-30.txt --all

 
