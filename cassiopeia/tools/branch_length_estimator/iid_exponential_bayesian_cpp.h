#ifndef IID_EXPONENTIAL_BAYESIAN_CPP_H
#define IID_EXPONENTIAL_BAYESIAN_CPP_H

#include <vector>

using namespace std;

class DP{
    public:
        DP();
        ~DP();
        void run(
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
        );
        // The following methods access the results of the run() method.
        vector<pair<vector<int>, double> > get_down_res();
        vector<pair<vector<int>, double> > get_up_res();
        vector<pair<int, double> > get_posterior_means_res();
        vector<pair<int, vector<double> > > get_posteriors_res();
        vector<pair<int, vector<double> > > get_log_joints_res();
        double get_log_likelihood_res();

        private:
            static const int maxN = 4096;
            static const int maxK = 151;
            static const int maxT = 601;

            // These are the parameters to the run() call.
            int N;
            vector<vector<int> > children;
            int root;
            vector<int> is_internal_node;
            vector<int> get_number_of_mutated_characters_in_node;
            vector<int> non_root_internal_nodes;
            vector<int> leaves;
            vector<int> parent;
            int K;
            vector<int> Ks;
            int T;
            double r;
            double lam;
            double sampling_probability;
            vector<int> is_leaf;

            // These are computed internally.
            double dt;
            double down_cache[maxN][maxT + 1][maxK + 1];
            double up_cache[maxN][maxT + 1][maxK + 1];
            double p_unsampled[maxT + 1];
            double log_joints[maxN][maxT + 1];
            double posteriors[maxN][maxT + 1];
            double posterior_means[maxN];

            void precompute_p_unsampled();
            bool state_is_valid(int v, int x);
            double down(int v, int t, int x);
            double up(int v, int t, int x);
            void populate_down_res();
            void populate_up_res();
            void populate_log_likelihood_res();
            double compute_log_joint(int v, int t);
            void populate_log_joints_res();
            void populate_posteriors_res();
            void populate_posterior_means_res();
            void populate_posterior_results();

            vector<pair<vector<int>, double> > down_res;
            vector<pair<vector<int>, double> > up_res;
            vector<pair<int, double> > posterior_means_res;
            vector<pair<int, vector<double> > > posteriors_res;
            vector<pair<int, vector<double> > > log_joints_res;
            double log_likelihood_res;
};

#endif
