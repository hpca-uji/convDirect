import tvm
from   tvm import te
import tvm.testing

import math
import numpy
import timeit
import os
import sys

digit=["0","1","2","3","4","5","6","7","8","9"]

def get_value(line, tag):
    _sp = line.split("#define")[1]
    value = ""
    for i in range(0, len(_sp) - 1):
        if (_sp[i] == tag[0]) and (_sp[i + 1] == tag[1]):
            j = i + 3
            while _sp[j] not in digit: j += 1
            while _sp[j] in digit:
                value += _sp[j]
                j += 1
            break

    return int(value)

def get_mr_nr():
    curr_path = os.path.abspath(os.getcwd())
    blis_file = os.path.join(curr_path, "src/qblis.h")
    blis_fd   = open(blis_file, "r")
    while True:
        line=blis_fd.readline()
        if not line: break
        if line.find("TVM") != -1:
            while line.find("#define MR") == -1:
                line=blis_fd.readline()
            mr = get_value(line, "MR")
            
            while line.find("#define NR") == -1:
                line=blis_fd.readline()

            nr = get_value(line, "NR")
    
    return (mr,nr)


    
def microkernel_direct(mr,nr,kc,target,dtype,dev):
    M=mr
    N=nr
    K=kc
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    Caux = te.placeholder((M, N), name="Caux")
    Ar = te.compute((K,M),lambda ka,ma:A[ma,ka], name='Ar')
    Br = te.compute((K,N),lambda ka,ma:B[ka,ma], name='Br')
    
    C = te.compute(
            (M, N),
            lambda m, n: te.sum(
                Ar[k,m]
                *
                Br[k,n]
                , axis=k, init=Caux[m,n]),
            name='C')

    blis = te.create_schedule(C.op)
    (k,),(m, n) = C.op.reduce_axis, C.op.axis

    blis[Ar].compute_at( blis[C], k )
    blis[Br].compute_at( blis[C], k )
    x, y = blis[Ar].op.axis
    blis[Ar].vectorize(y)
    z, l = blis[Br].op.axis
    blis[Br].vectorize(l)

    blis[C].reorder(k,m,n)  #n = jr, m=ir
    n,ni = blis[C].split(n,factor=4)
    #blis[C].unroll(k)
    blis[C].unroll(m)
    blis[C].unroll(n)
    blis[C].vectorize(ni)

    curr_path=os.path.abspath(os.getcwd())
    #curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    base_path=os.path.join(curr_path, "build/")
    # Compile library in system library mode
    func = tvm.build(blis, [A, B, Caux, C], target=target+" --system-lib", name="microkernel_mult".format(mr,nr))
    syslib_path = os.path.join(base_path, "microkernel_gen.o".format(mr,nr))
    func.save(syslib_path)    
    assert func

    return func



if __name__ == "__main__":
    if len(sys.argv) == 1:
        mr, nr = get_mr_nr()
    else:
        mr, nr = (int(sys.argv[1]), int(sys.argv[2]))

    kc = nr
    dtype  = "float32"
    target = "llvm -device=arm_cpu -mattr=+v8.2a,+fp-armv8,+neon" #"llvm"
    dev = tvm.device(target, 0)    
    
    ukernel = microkernel_direct(mr,nr,kc,target,dtype,dev)
