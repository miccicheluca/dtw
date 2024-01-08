# cython: language_level=3

import numpy as np 
cimport numpy as np

from libc.stdlib cimport malloc, free
from libcpp.algorithm cimport sort
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf

from cython.parallel cimport prange
from cython.operator cimport dereference, postincrement

from dtaidistance cimport dtw


from cython cimport boundscheck, wraparound, infer_types

@boundscheck(False)
@wraparound(False)
@infer_types(False)
cdef double compute_median(double* arr_diff, int n_elem) nogil:
    cdef:
        int idx = n_elem//2
        double res = 0

    sort(arr_diff, arr_diff+n_elem)
    if n_elem%2==1:
        res = arr_diff[idx]
    else:
        res = (arr_diff[idx]+arr_diff[idx-1])/2
    return res

@boundscheck(False)
@wraparound(False)
@infer_types(False)
cdef double find_key_max_freq_um(unordered_map[double, int] freq_map) nogil:
    cdef:
        cdef unordered_map[double,int].iterator it = freq_map.begin()
        double res = dereference(it).first
        int max = 0
    while(it != freq_map.end()):
        if dereference(it).second > max:
            max = dereference(it).second
            res = dereference(it).first
        postincrement(it)
    return res

@boundscheck(False)
@wraparound(False)
@infer_types(False)
cdef double compute_mode(double* arr_diff, int n_elem) nogil:
    cdef:
        unordered_map[double, int] freq_map
        int i

    for i in range(n_elem):
        freq_map[arr_diff[i]]+=1
    
    return find_key_max_freq_um(freq_map=freq_map)
    

@boundscheck(False)
@wraparound(False)
cdef double compute_ll_over_array(double[:,:] arr_of_tuple, str lag_method) nogil:
    cdef:
        int i
        int n_elem = len(arr_of_tuple)
        double* arr_diff = <double*> malloc(n_elem*sizeof(double))
        double res = 0
        double check = 0
    
    for i in range(n_elem):
        arr_diff[i] = arr_of_tuple[i][0] - arr_of_tuple[i][1]
    
    
    if lag_method == "median":
        res = compute_median(arr_diff=arr_diff, n_elem=n_elem)
    else:
        res = compute_mode(arr_diff=arr_diff, n_elem=n_elem)
    free(arr_diff)
    return res


@boundscheck(False)
@wraparound(False)
def compute_ll_from_arr_tuple(double[:,:,:] arr, str lag_method):
    cdef:
        int i
        int n = len(arr)
        double[:] res = np.array(np.zeros(n), dtype=np.double)
    for i in prange(n, nogil=True):
        res[i] = compute_ll_over_array(arr_of_tuple=arr[i], lag_method=lag_method)

    return np.array(res, dtype=np.double)

@boundscheck(False)
@wraparound(False)
def gen_lead_lag_matrix(int n_assets,int[:] median_from_paths, double[:] clusters):
    cdef:
        int[:,:] lead_lag_matrix = np.array(np.zeros((n_assets, n_assets)), dtype=int)
        int i, j
        int ll_counter = 0
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            if clusters[i] == clusters[j]:
                    lead_lag_matrix[i][j] = median_from_paths[ll_counter]
                    lead_lag_matrix[j][i] = -median_from_paths[ll_counter]
                    ll_counter+=1
    return np.array(lead_lag_matrix, dtype=np.double)


