#!/bin/bash

# First column values
key="fp_ret_sse_avx_ops.all:u" ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="page-faults:u"            ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="cycles:u"                 ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="stalled-cycles-frontend:u";  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="stalled-cycles-backend:u" ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="instructions:u"           ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="branches:u"               ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="branch-misses:u"          ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="cache-misses:u"           ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="cache-references:u"       ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="L1-dcache-loads:u"        ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="L1-dcache-load-misses:u"  ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="L1-icache-loads:u"        ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="L1-icache-load-misses:u"  ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="dTLB-loads:u"             ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="dTLB-load-misses:u"       ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="iTLB-loads:u"             ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="iTLB-load-misses:u"       ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val
key="L1-dcache-prefetches:u"   ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%-30.30s %s\n' $key $val

# Fourth column values
key="frontend cycles idle"      ;  val=`grep 'frontend cycles idle'       $1 | awk '{print $4}'`;  printf '%-30.30s %s\n' "$key" "$val"
key="backend cycles idle"       ;  val=`grep 'backend cycles idle'        $1 | awk '{print $4}'`;  printf '%-30.30s %s\n' "$key" "$val"
key="insn per cycle"            ;  val=`grep 'insn per cycle'             $1 | awk '{print $4}'`;  printf '%-30.30s %s\n' "$key" "$val"
key="stalled cycles per insn"   ;  val=`grep 'stalled cycles per insn'    $1 | awk '{print $2}'`;  printf '%-30.30s %s\n' "$key" "$val"
key="of all branches"           ;  val=`grep 'of all branches'            $1 | awk '{print $4}'`;  printf '%-30.30s %s\n' "$key" "$val"
key="% of all cache refs"       ;  val=`grep '% of all cache refs'        $1 | awk '{print $4}'`;  printf '%-30.30s %s\n' "$key" "$val"
key="of all L1-dcache accesses" ;  val=`grep 'of all L1-dcache accesses'  $1 | awk '{print $4}'`;  printf '%-30.30s %s\n' "$key" "$val"
key="of all L1-icache accesses" ;  val=`grep 'of all L1-icache accesses'  $1 | awk '{print $4}'`;  printf '%-30.30s %s\n' "$key" "$val"
key="of all dTLB cache accesses";  val=`grep 'of all dTLB cache accesses' $1 | awk '{print $4}'`;  printf '%-30.30s %s\n' "$key" "$val"
key="of all iTLB cache accesses";  val=`grep 'of all iTLB cache accesses' $1 | awk '{print $4}'`;  printf '%-30.30s %s\n' "$key" "$val"


#       140,515,869      stalled-cycles-frontend:u #    0.48%   
#    21,733,877,988      stalled-cycles-backend:u  #   74.92%   
#    44,879,903,102      instructions:u            #    1.55    
#                                                  #    0.48    
#         7,875,853      branch-misses:u           #    0.22%   
#        72,008,854      cache-misses:u            #    4.709   
#       876,061,373      L1-dcache-load-misses:u   #    3.82%    
#         2,125,678      L1-icache-load-misses:u   #    0.81%    
#        12,638,002      dTLB-load-misses:u        #   32.14%    
#           419,041      iTLB-load-misses:u        #  164.64%    
