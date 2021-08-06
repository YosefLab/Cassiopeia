# distutils: language = c++

from ._iid_exponential_bayesian cimport DP
from libcpp.vector cimport vector
from libcpp.map cimport map

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.
cdef class PyDP:
    cdef DP* c_dp

    def __cinit__(self):
        self.c_dp = new DP();

    def run(
        self,
        int N,
        vector[vector[int]] children,
        int root,
        vector[int] is_internal_node,
        vector[int] get_number_of_mutated_characters_in_node,
        vector[int] non_root_internal_nodes,
        vector[int] leaves,
        vector[int] parent,
        int K,
        vector[int] Ks,
        int T,
        double r,
        double lam,
        double sampling_probability,
        vector[int] is_leaf,
    ):
        self.c_dp.run(
            N,
            children,
            root,
            is_internal_node,
            get_number_of_mutated_characters_in_node,
            non_root_internal_nodes,
            leaves,
            parent,
            K,
            Ks,
            T,
            r,
            lam,
            sampling_probability,
            is_leaf,
        )
        
    def get_down_res(self):
        return self.c_dp.get_down_res()
    
    def get_up_res(self):
        return self.c_dp.get_up_res()
    
    def get_posterior_means_res(self):
        return self.c_dp.get_posterior_means_res()
    
    def get_posteriors_res(self):
        return self.c_dp.get_posteriors_res()
    
    def get_log_joints_res(self):
        return self.c_dp.get_log_joints_res()
    
    def get_log_likelihood_res(self):
        return self.c_dp.get_log_likelihood_res()

    def __dealloc__(self):
        del self.c_dp

