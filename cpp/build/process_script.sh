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



# First column values
key="fp_ret_sse_avx_ops.all:u" ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="page-faults:u"            ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="cycles:u"                 ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="stalled-cycles-frontend:u";  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="stalled-cycles-backend:u" ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="instructions:u"           ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="branches:u"               ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="branch-misses:u"          ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="cache-misses:u"           ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="cache-references:u"       ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="L1-dcache-loads:u"        ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="L1-dcache-load-misses:u"  ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="L1-icache-loads:u"        ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="L1-icache-load-misses:u"  ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="dTLB-loads:u"             ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="dTLB-load-misses:u"       ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="iTLB-loads:u"             ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="iTLB-load-misses:u"       ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val
key="L1-dcache-prefetches:u"   ;  val=`grep $key $1 | awk '{print $1}'`;  printf '%s\n' $val

# Fourth column values
key="frontend cycles idle"      ;  val=`grep 'frontend cycles idle'       $1 | awk '{print $4}'`;  printf '%s\n' "$val"
key="backend cycles idle"       ;  val=`grep 'backend cycles idle'        $1 | awk '{print $4}'`;  printf '%s\n' "$val"
key="insn per cycle"            ;  val=`grep 'insn per cycle'             $1 | awk '{print $4}'`;  printf '%s\n' "$val"
key="stalled cycles per insn"   ;  val=`grep 'stalled cycles per insn'    $1 | awk '{print $2}'`;  printf '%s\n' "$val"
key="of all branches"           ;  val=`grep 'of all branches'            $1 | awk '{print $4}'`;  printf '%s\n' "$val"
key="% of all cache refs"       ;  val=`grep '% of all cache refs'        $1 | awk '{print $4}'`;  printf '%s\n' "$val"
key="of all L1-dcache accesses" ;  val=`grep 'of all L1-dcache accesses'  $1 | awk '{print $4}'`;  printf '%s\n' "$val"
key="of all L1-icache accesses" ;  val=`grep 'of all L1-icache accesses'  $1 | awk '{print $4}'`;  printf '%s\n' "$val"
key="of all dTLB cache accesses";  val=`grep 'of all dTLB cache accesses' $1 | awk '{print $4}'`;  printf '%s\n' "$val"
key="of all iTLB cache accesses";  val=`grep 'of all iTLB cache accesses' $1 | awk '{print $4}'`;  printf '%s\n' "$val"
