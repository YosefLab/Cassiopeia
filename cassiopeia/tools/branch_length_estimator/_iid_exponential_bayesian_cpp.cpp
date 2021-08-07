#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <math.h>
#include "_iid_exponential_bayesian_cpp.h"

#define forn(i, n) for(int i = 0; i < int(n); i++)
#define forall(i,c) for(typeof((c).begin()) i = (c).begin();i != (c).end();i++)

using namespace std;

const double INF = 1e16;

int _compatible_with_observed_data(int x, int observed_cuts){
    return x <= observed_cuts;
}

double logsumexp(const vector<double> & lls){
    double mx = -INF;
    for(double ll: lls){
        mx = max(mx, ll);
    }
    double res = 0.0;
    for(double ll: lls){
        res += exp(ll - mx);
    }
    res = log(res) + mx;
    return res;
}

DP::DP(){}

void DP::precompute_p_unsampled(){
    double dt = 1.0 / T;
    if(1 - lam * dt <= 0){
        cerr << "1 - lam * dt = " << 1 - lam * dt << " should be positive!" << endl;
        throw std::invalid_argument("Discretization level too small. Please increase it.");
    }
    forn(i, T + 1) p_unsampled[i] = -100000000.0;
    if(sampling_probability < 1.0){
        p_unsampled[T] = log(1.0 - sampling_probability);
        for(int t = T - 1; t >= 0; t--){
            vector<double> log_likelihoods_cases;
            log_likelihoods_cases.push_back(log(1 - lam * dt) + p_unsampled[t + 1]);
            log_likelihoods_cases.push_back(log(lam * dt) + 2 * p_unsampled[t + 1]);
            p_unsampled[t] = logsumexp(log_likelihoods_cases);
        }
    }
}

pair<int, int> DP::valid_cuts_range(int v){
    if(v == root)
        return pair<int, int> (0, 0);
    int p = parent[v];
    int cuts_v = get_number_of_mutated_characters_in_node[v];
    int cuts_p = get_number_of_mutated_characters_in_node[p];
    return pair<int, int> (cuts_p, cuts_v);
}

bool DP::state_is_valid(int v, int x){
    if(v == root)
        return x == 0;
    int p = parent[v];
    int cuts_v = get_number_of_mutated_characters_in_node[v];
    int cuts_p = get_number_of_mutated_characters_in_node[p];
    return cuts_p <= x && x <= cuts_v;
}

double DP::down(int v, int t, int x){
    // Avoid doing anything at all for invalid states.
    if(!state_is_valid(v, x)){
        return -INF;
    }
    if(down_cache[v][t][x] != INF){
        return down_cache[v][t][x];
    }
    // Pull out params
    double dt = 1.0 / T;
    int Kv = Ks[v];
    assert(v != root);
    assert(0 <= t && t <= T);
    assert(0 <= x && x <= Kv);
    if(!(1.0 - lam * dt - Kv * r * dt > 0)){
        throw std::invalid_argument("Discretization level too small. Please increase it.");
    }
    double log_likelihood = 0.0;
    if(t == T){
        // Base case
        if (
            is_leaf[v]
            && (x == get_number_of_mutated_characters_in_node[v])
        ){
            log_likelihood = log(sampling_probability);
        } else {
            log_likelihood = -INF;
        }
    }
    else{
        // Recursion.
        vector<double> log_likelihoods_cases;
        // Case 1: Nothing happens
        log_likelihoods_cases.push_back(
            log(1.0 - lam * dt - (Kv - x) * r * dt)
            + down(v, t + 1, x)
        );
        // Case 2: One character mutates.
        if(x + 1 <= Kv){
            log_likelihoods_cases.push_back(
                log((Kv - x) * r * dt) + down(v, t + 1, x + 1)
            );
        }
        // Case 3: Cell divides
        // The number of cuts at this state must match the ground truth.
        if (
            _compatible_with_observed_data(
                x, get_number_of_mutated_characters_in_node[v]
            )
            && (!is_leaf[v])
        ){
            double ll = 0.0;
            forn(i, children[v].size()){
                int child = children[v][i];
                ll += down(child, t + 1, x);
            }
            ll += log(lam * dt);
            log_likelihoods_cases.push_back(ll);
        }
        // Case 4: There is a cell division event, but one of the two
        // lineages is not sampled
        double ll = log(2 * lam * dt) + p_unsampled[t + 1] + down(v, t + 1, x);
        log_likelihoods_cases.push_back(ll);
        log_likelihood = logsumexp(log_likelihoods_cases);
    }
    down_cache[v][t][x] = log_likelihood;
    assert(log_likelihood < INF);
    return log_likelihood;
}

double DP::up(int v, int t, int x){
    // Avoid doing anything at all for invalid states.
    if(!state_is_valid(v, x))
        return -INF;
    if(up_cache[v][t][x] != INF){
        return up_cache[v][t][x];
    }
    // Pull out params
    double dt = 1.0 / T;
    int Kv = Ks[v];
    assert(0 <= v  && v < N);
    assert(0 <= t && t <= T);
    assert(0 <= x && x <= Kv);
    if(!(1.0 - lam * dt - Kv * r * dt > 0)){
        throw std::invalid_argument("Discretization level too small. Please increase it.");
    }
    double log_likelihood = 0.0;
    if(v == root){
        // Base case: we reached the root of the tree.
        if((t == 0) && (x == get_number_of_mutated_characters_in_node[v]))
            log_likelihood = 0.0;
        else
            log_likelihood = -INF;
    } else if(t == 0){
        // Base case: we reached the start of the process, but we're not yet
        // at the root.
        assert(v != root);
        log_likelihood = -INF;
    } else {
        // Recursion.
        vector<double> log_likelihoods_cases;
        // Case 1: Nothing happened
        log_likelihoods_cases.push_back(
            log(1.0 - lam * dt - (Kv - x) * r * dt) + up(v, t - 1, x)
        );
        // Case 2: Mutation happened
        if(x - 1 >= 0){
            log_likelihoods_cases.push_back(
                log((Kv - (x - 1)) * r * dt) + up(v, t - 1, x - 1)
            );
        }
        // Case 3: A cell division happened
        if(v != root){
            int p = parent[v];
            if(_compatible_with_observed_data(
                x, get_number_of_mutated_characters_in_node[p]
            )){
                vector<int> siblings;
                for(auto u: children[p])
                    if(u != v)
                        siblings.push_back(u);
                double ll = log(lam * dt) + up(p, t - 1, x);
                for(auto u: siblings){
                    ll += down(u, t, x);
                }
                log_likelihoods_cases.push_back(ll);
            }
        }
        // Case 4: There is a cell division event, but one of the two
        // lineages is not sampled
        double ll = log(2 * lam * dt) + p_unsampled[t - 1] + up(v, t - 1, x);
        log_likelihoods_cases.push_back(ll);
        log_likelihood = logsumexp(log_likelihoods_cases);
    }
    up_cache[v][t][x] = log_likelihood;
    assert(log_likelihood < INF);
    return log_likelihood;
}

void DP::populate_down_res(){
    forn(v, N){
        if(v == root) continue;
        forn(t, T + 1){
            pair<int, int> x_range = valid_cuts_range(v);
            for(int x = x_range.first; x <= x_range.second; x++){
                double ll = down(v, t, x);
                down_res.push_back(pair<vector<int>, double>(vector<int> {v, t, x}, ll));
            }
        }
    }
}

void DP::populate_up_res(){
    forn(v, N){
        forn(t, T + 1){
            pair<int, int> x_range = valid_cuts_range(v);
            for(int x = x_range.first; x <= x_range.second; x++){
                double ll = up(v, t, x);
                up_res.push_back(pair<vector<int>, double>(vector<int> {v, t, x}, ll));
            }
        }
    }
}

void DP::populate_log_likelihood_res(){
    log_likelihood_res = 0.0;
    for(auto child_of_root: children[root]){
        log_likelihood_res += down(child_of_root, 0, 0);
    }
}

double DP::compute_log_joint(int v, int t){
    assert(is_internal_node[v] and v != root);
    vector<int> valid_num_cuts;
    valid_num_cuts.push_back(get_number_of_mutated_characters_in_node[v]);
    vector<double> ll_for_xs;
    for(auto x: valid_num_cuts){
        double ll_for_x = up(v, t, x);
        for(auto u: children[v]){
            ll_for_x += down(u, t, x);
        }
        ll_for_xs.push_back(ll_for_x);
    }
    return logsumexp(ll_for_xs);
}

void DP::populate_log_joints_res(){
    for(auto v: non_root_internal_nodes){
        vector<double> vec;
        for(int t = 0; t <= T; t++){
            vec.push_back(log_joints[v][t]);
        }
        log_joints_res.push_back(pair<int, vector<double> >(v, vec));
    }
}

void DP::populate_posteriors_res(){
    for(auto v: non_root_internal_nodes){
        vector<double> vec;
        for(int t = 0; t <= T; t++){
            vec.push_back(posteriors[v][t]);
        }
        posteriors_res.push_back(pair<int, vector<double> >(v, vec));
    }
}

void DP::populate_posterior_means_res(){
    for(auto v: non_root_internal_nodes){
        posterior_means_res.push_back(pair<int, double>(v, posterior_means[v]));
    }
}

void DP::populate_posterior_results(){
    // mimmicks _compute_posteriors of the python implementation.
    for(auto v: non_root_internal_nodes){
        // Compute the log_joints.
        for(int t = 0; t <= T; t++){
            log_joints[v][t] = compute_log_joint(v, t);
        }

        // Compute the posteriors
        double mx = -INF;
        for(int t = 0; t <= T; t++){
            mx = max(mx, log_joints[v][t]);
        }
        for(int t = 0; t <= T; t++){
            posteriors[v][t] = exp(log_joints[v][t] - mx);
        }
        // Normalize posteriors
        double tot_sum = 0.0;
        for(int t = 0; t <= T; t++){
            tot_sum += posteriors[v][t];
        }
        for(int t = 0; t <= T; t++){
            posteriors[v][t] /= tot_sum;
        }

        // Compute the posterior means.
        posterior_means[v] = 0.0;
        for(int t = 0; t <= T; t++){
            posterior_means[v] += posteriors[v][t] * t;
        }
        posterior_means[v] /= double(T);
    }

    // Write out the log_joints, posteriors, and posterior_means
    populate_log_joints_res();
    populate_posteriors_res();
    populate_posterior_means_res();
}

void DP::run(
    int N,
    vector<vector<int> > children,
    int root,
    vector<int> is_internal_node,
    vector<int> get_number_of_mutated_characters_in_node,
    vector<int> non_root_internal_nodes,
    vector<int> leaves,
    vector<int> parent,
    int K,
    vector<int> Ks,
    int T,
    double r,
    double lam,
    double sampling_probability,
    vector<int> is_leaf
){
    if((N > maxN) || (T > maxT) || (K > maxK)){
        throw std::invalid_argument("N or T or K too large");
    }
    this->N = N;
    this->children = children;
    this->root = root;
    this->is_internal_node = is_internal_node;
    this->get_number_of_mutated_characters_in_node = get_number_of_mutated_characters_in_node;
    this->non_root_internal_nodes = non_root_internal_nodes;
    this->leaves = leaves;
    this->parent = parent;
    this->K = K;
    this->Ks = Ks;
    this->T = T;
    this->r = r;
    this->lam = lam;
    this->sampling_probability = sampling_probability;
    this->is_leaf = is_leaf;

    this->dt = 1.0 / T;

    forn(v, N)
        forn(t, T + 1)
            forn(k, K + 1)
                down_cache[v][t][k] = up_cache[v][t][k] = INF;
    
    precompute_p_unsampled();

    populate_down_res();
    populate_up_res();
    populate_log_likelihood_res();
    populate_posterior_results();
}

DP::~DP(){}

vector<pair<vector<int>, double> > DP::get_down_res(){
    return down_res;
}

vector<pair<vector<int>, double> > DP::get_up_res(){
    return up_res;
}

vector<pair<int, double> > DP::get_posterior_means_res(){
    return posterior_means_res;
}

vector<pair<int, vector<double> > > DP::get_posteriors_res(){
    return posteriors_res;
}

vector<pair<int, vector<double> > > DP::get_log_joints_res(){
    return log_joints_res;
}

double DP::get_log_likelihood_res(){
    return log_likelihood_res;
}

int main(){
    return 0;
}