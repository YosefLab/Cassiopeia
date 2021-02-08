#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <assert.h>
#include <math.h>

#define forn(i, n) for(int i = 0; i < int(n); i++)
#define forall(i,c) for(typeof((c).begin()) i = (c).begin();i != (c).end();i++)

using namespace std;
const int maxN = 10000;
const int maxK = 100;
const int maxT = 512;
const float INF = 1e16;
float _down_cache[maxN][maxT + 1][maxK + 1];
float _up_cache[maxN][maxT + 1][maxK + 1];

string input_dir = "";
string output_dir = "";
int N = -1;
vector<int> children[maxN];
int root = -1;
int is_internal_node[maxN];
int get_number_of_mutated_characters_in_node[maxN];
vector<int> non_root_internal_nodes;
vector<int> leaves;
int parent[maxN];
int is_leaf[maxN];
int K = -1;
int T = -1;
int enforce_parsimony = -1;
float r;
float lam;

float logsumexp(const vector<float> & lls){
    float mx = -INF;
    for(auto ll: lls){
        mx = max(mx, ll);
    }
    float res = 0.0;
    for(auto ll: lls){
        res += exp(ll - mx);
    }
    res = log(res) + mx;
    return res;
}

int _compatible_with_observed_data(int x, int observed_cuts){
    if(enforce_parsimony)
        return x == observed_cuts;
    else
        return x <= observed_cuts;
}

bool _state_is_valid(int v, int t, int x){
    if(v == root)
        return x == 0;
    int p = parent[v];
    int cuts_v = get_number_of_mutated_characters_in_node[v];
    int cuts_p = get_number_of_mutated_characters_in_node[p];
    if(enforce_parsimony){
        return cuts_p <= x && x <= cuts_v;
    } else {
        return x <= cuts_v;
    }
}


void read_N(){
    ifstream fin(input_dir + "/N.txt");
    if(!fin.good()){
        cerr << "N input file not found" << endl;
        exit(1);
    }
    fin >> N;
    if(N == -1){
        cerr << "N input corrupted" << endl;
        exit(1); 
    }
    if(N > maxN){
        cerr << "N larger than maxN" << endl;
        exit(1);
    }
}

void read_children(){
    ifstream fin(input_dir + "/children.txt");
    if(!fin.good()){
        cerr << "children input file not found" << endl;
        exit(1);
    }
    int lines_read = 0;
    int v;
    while(fin >> v){
        lines_read++;
        int n_children;
        fin >> n_children;
        for(int i = 0; i < n_children; i++){
            int c;
            fin >> c;
            children[v].push_back(c);
        }
    }
    if(lines_read != N){
        cerr << "children input corrupted" << endl;
        exit(1);
    }
}

void read_root(){
    ifstream fin(input_dir + "/root.txt");
    fin >> root;
    if(root == -1){
        cerr << "N input corrupted" << endl;
        exit(1); 
    }
}

void read_is_internal_node(){
    ifstream fin(input_dir + "/is_internal_node.txt");
    int lines_read = 0;
    int v;
    while(fin >> v){
        lines_read++;
        fin >> is_internal_node[v];
    }
    if(lines_read != N){
        cerr << "is_internal_node input corrupted" << endl;
        exit(1);
    }
}

void read_get_number_of_mutated_characters_in_node(){
    ifstream fin(input_dir + "/get_number_of_mutated_characters_in_node.txt");
    int lines_read = 0;
    int v;
    while(fin >> v){
        lines_read++;
        fin >> get_number_of_mutated_characters_in_node[v];
    }
    if(lines_read != N){
        cerr << "get_number_of_mutated_characters_in_node input corrupted" << endl;
        exit(1);
    }
}

void read_non_root_internal_nodes(){
    ifstream fin(input_dir + "/non_root_internal_nodes.txt");
    int v;
    while(fin >> v){
        non_root_internal_nodes.push_back(v);
    }
}

void read_leaves(){
    ifstream fin(input_dir + "/leaves.txt");
    int v;
    while(fin >> v){
        leaves.push_back(v);
    }
    if(leaves.size() == 0){
        cerr << "leaves input corrupted" << endl;
        exit(1); 
    }
}

void read_parent(){
    ifstream fin(input_dir + "/parent.txt");
    int lines_read = 0;
    int v;
    while(fin >> v){
        lines_read++;
        fin >> parent[v];
    }
    if(lines_read != N - 1){
        cerr << "parent input corrupted" << endl;
        exit(1);
    }
}

void read_is_leaf(){
    ifstream fin(input_dir + "/is_leaf.txt");
    int lines_read = 0;
    int v;
    while(fin >> v){
        lines_read++;
        fin >> is_leaf[v];
    }
    if(lines_read != N){
        cerr << "is_leaf input corrupted" << endl;
        exit(1);
    }
}

void read_K(){
    ifstream fin(input_dir + "/K.txt");
    fin >> K;
    if(K == -1){
        cerr << "K input corrupted" << endl;
        exit(1); 
    }
    if(K > maxK){
        cerr << "K larger than maxK" << endl;
        exit(1);
    }
}

void read_T(){
    // T is the discretization level.
    ifstream fin(input_dir + "/T.txt");
    fin >> T;
    if(T == -1){
        cerr << "T input corrupted" << endl;
        exit(1); 
    }
    if(T > maxT){
        cerr << "T larger than maxT" << endl;
        exit(1);
    }
}

void read_enforce_parsimony(){
    ifstream fin(input_dir + "/enforce_parsimony.txt");
    fin >> enforce_parsimony;
    if(enforce_parsimony == -1){
        cerr << "enforce_parsimony input corrupted" << endl;
        exit(1); 
    }
}

void read_r(){
    ifstream fin(input_dir + "/r.txt");
    fin >> r;
    if(r == -1){
        cerr << "r input corrupted" << endl;
        exit(1); 
    }
}

void read_lam(){
    ifstream fin(input_dir + "/lam.txt");
    fin >> lam;
    if(lam == -1){
        cerr << "lam input corrupted" << endl;
        exit(1); 
    }
}

float down(int v, int t, int x){
    // Avoid doing anything at all for invalid states.
    if(!_state_is_valid(v, t, x)){
        return -INF;
    }
    if(_down_cache[v][t][x] < 1.0){
        return _down_cache[v][t][x];
    }
    // Pull out params
    float dt = 1.0 / T;
    assert(v != root);
    assert(0 <= t && t <= T);
    assert(0 <= x && x <= K);
    if(!(1.0 - lam * dt - K * r * dt > 0)){
        cerr << "Please choose a bigger discretization_level." << endl;
        exit(1);
    }
    float log_likelihood = 0.0;
    if(t == T){
        // Base case
        if (
            is_leaf[v]
            && (x == get_number_of_mutated_characters_in_node[v])
        ){
            log_likelihood = 0.0;
        } else {
            log_likelihood = -INF;
        }
    }
    else{
        // Recursion.
        vector<float> log_likelihoods_cases;
        // Case 1: Nothing happens
        log_likelihoods_cases.push_back(
            log(1.0 - lam * dt - (K - x) * r * dt)
            + down(v, t + 1, x)
        );
        // Case 2: One character mutates.
        if(x + 1 <= K){
            log_likelihoods_cases.push_back(
                log((K - x) * r * dt) + down(v, t + 1, x + 1)
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
            float ll = 0.0;
            forn(i, children[v].size()){
                int child = children[v][i];
                ll += down(child, t + 1, x);// If we want to ignore missing data, we just have to replace x by x+gone_missing(p->v). I.e. dropped out characters become free mutations.
            }
            ll += log(lam * dt);
            log_likelihoods_cases.push_back(ll);
        }
        log_likelihood = logsumexp(log_likelihoods_cases);
    }
    _down_cache[v][t][x] = log_likelihood;
    return log_likelihood;
}

float up(int v, int t, int x){
    // Avoid doing anything at all for invalid states.
    if(!_state_is_valid(v, t, x))
        return -INF;
    if(_up_cache[v][t][x] < 1.0){
        return _up_cache[v][t][x];
    }
    // Pull out params
    float dt = 1.0 / T;
    assert(0 <= v  && v < N);
    assert(0 <= t && t <= T);
    assert(0 <= x && x <= K);
    if(!(1.0 - lam * dt - K * r * dt > 0)){
        cerr << "Please choose a bigger discretization_level." << endl;
        exit(1);
    }
    float log_likelihood = 0.0;
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
        vector<float> log_likelihoods_cases;
        // Case 1: Nothing happened
        log_likelihoods_cases.push_back(
            log(1.0 - lam * dt - (K - x) * r * dt) + up(v, t - 1, x)
        );
        // Case 2: Mutation happened
        if(x - 1 >= 0){
            log_likelihoods_cases.push_back(
                log((K - (x - 1)) * r * dt) + up(v, t - 1, x - 1)
            );
        }
        // Case 3: A cell division happened
        if(v != root){
            int p = parent[v];
            if(_compatible_with_observed_data(
                x, get_number_of_mutated_characters_in_node[p] // If we want to ignore missing data, we just have to replace x by x-gone_missing(p->v). I.e. dropped out characters become free mutations.
            )){
                vector<int> siblings;
                for(auto u: children[p])
                    if(u != v)
                        siblings.push_back(u);
                float ll = log(lam * dt) + up(p, t - 1, x);  // If we want to ignore missing data, we just have to replace x by x-gone_missing(p->v). I.e. dropped out characters become free mutations.
                for(auto u: siblings){
                    ll += down(u, t, x); // If we want to ignore missing data, we just have to replace x by cuts(p)+gone_missing(p->u). I.e. dropped out characters become free mutations.
                }
                log_likelihoods_cases.push_back(ll);
            }
        }
        log_likelihood = logsumexp(log_likelihoods_cases);
    }
    _up_cache[v][t][x] = log_likelihood;
    return log_likelihood;
}

void write_down(){
    ofstream fout(output_dir + "/down.txt");
    string res = "";
    forn(v, N){
        if(v == root) continue;
        forn(t, T + 1){
            forn(x, K + 1){
                if(_state_is_valid(v, t, x)){
                    res += to_string(v) + " " + to_string(t) + " " + to_string(x) + " " + to_string(down(v, t, x)) + "\n";
                }
            }
        }
    }
    fout << res;
}

void write_up(){
    ofstream fout(output_dir + "/up.txt");
    string res = "";
    forn(v, N){
        forn(t, T + 1){
            forn(x, K + 1){
                if(_state_is_valid(v, t, x)){
                    res += to_string(v) + " " + to_string(t) + " " + to_string(x) + " " + to_string(up(v, t, x)) + "\n";
                }
            }
        }
    }
    fout << res;
}


int main(int argc, char *argv[]){
    if(argc != 3){
        cerr << "Need intput_dir and output_dir arguments, " << argc - 1 << " provided." << endl;
        exit(1);
    }
    input_dir = string(argv[1]);
    output_dir = string(argv[2]);
    read_N();
    read_children();
    read_root();
    read_is_internal_node();
    read_get_number_of_mutated_characters_in_node();
    read_non_root_internal_nodes();
    read_leaves();
    read_parent();
    read_is_leaf();
    read_K();
    read_T();
    read_enforce_parsimony();
    read_r();
    read_lam();

    forn(v, N) forn(t, T + 1) forn(k, K + 1) _down_cache[v][t][k] = _up_cache[v][t][k] = 1.0;
    write_down();
    write_up();
    return 0;
}
