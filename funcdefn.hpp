#ifndef __STENCILDEFN_HPP__
#define __STENCILDEFN_HPP__
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <deque>
#include <map>
#include <sstream>
#include <cassert>
#include <tuple>
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include "sort.hpp"
#include "vardecl.hpp"

enum PRINT_OPTION {
	INIT_ASSIGN=0,
	INIT_EMBED,
};

/* A class representing a statement of the form a = b + c;
   The lhs is a string, and rhs is an expression node. */
class stmtnode {
	private:
		expr_node *lhs_node, *rhs_node;
		std::vector<std::string> lhs_labels, lhs_names;
		std::vector<std::string> rhs_labels, rhs_names;
		STMT_OP op_type;
		int stmt_num, orig_stmt_num,live_count=0, nonlive_count;
		bool executed=false;
	public:
		stmtnode (STMT_OP, expr_node *, expr_node *);
		stmtnode (STMT_OP, expr_node *, expr_node *, int);
		stmtnode (STMT_OP, expr_node *, expr_node *, int, int);
		~stmtnode ();
		STMT_OP get_op_type (void);
		void set_op_type (STMT_OP);
		void set_expr_data_types (void);
		expr_node *get_lhs_expr (void);
		expr_node *get_rhs_expr (void);
		void set_lhs_expr (expr_node *);
		void set_rhs_expr (expr_node *);
		std::vector <std::string>& get_lhs_labels (void);
		std::vector <std::string>& get_rhs_labels (void);
		std::vector <std::string>& get_lhs_names (void);
		std::vector <std::string>& get_rhs_names (void);
		void set_lhs_labels (std::vector <std::string>);
		void set_rhs_labels (std::vector <std::string>);
		void set_lhs_names (std::vector <std::string>);
		void set_rhs_names (std::vector <std::string>);
		void set_nonlive_count (void);
		void set_nonlive_count (int);
		int get_live_count (void);
		int get_nonlive_count (void);
		int get_stmt_num (void);
		void set_stmt_num (int);
		int get_orig_stmt_num (void);
		void set_orig_stmt_num (int);
		bool is_executed (void);
		void set_executed (void);
		bool is_label_present (std::string);
		bool is_label_present (std::string, int &);
		void print_statement (std::stringstream &);
		std::string print_statement (std::stringstream &, std::vector<std::string> &, std::vector<std::string>);
		void print_statement (std::map<std::string, std::string> &, std::map<std::string, expr_node*> &, std::map<std::string, int> &, std::map<std::string, int> &, std::vector<std::string> &, std::stringstream &);
};

inline stmtnode::stmtnode (STMT_OP op, expr_node *lhs, expr_node *rhs) {
	op_type = op;
	lhs_node = lhs;
	rhs_node = rhs;
	set_expr_data_types ();
}

inline stmtnode::stmtnode (STMT_OP op, expr_node *lhs, expr_node *rhs, int num) {
	op_type = op;
	lhs_node = lhs;
	rhs_node = rhs;
	stmt_num = num;
	set_expr_data_types ();
}

inline stmtnode::stmtnode (STMT_OP op, expr_node *lhs, expr_node *rhs, int num, int orig_num) {
	op_type = op;
	lhs_node = lhs;
	rhs_node = rhs;
	stmt_num = num;
	orig_stmt_num = orig_num;
	set_expr_data_types ();
}

inline stmtnode::~stmtnode (void) {
	delete lhs_node;
	delete rhs_node;
}

inline STMT_OP stmtnode::get_op_type (void) {
	return op_type;
}


inline void stmtnode::set_op_type (STMT_OP op) {
	op_type = op;
}

inline expr_node* stmtnode::get_lhs_expr (void) {
	return lhs_node;
}

inline expr_node* stmtnode::get_rhs_expr (void) {
	return rhs_node;
}

inline void stmtnode::set_lhs_expr (expr_node *lhs) {
	lhs_node = lhs;
}

inline void stmtnode::set_rhs_expr (expr_node *rhs) {
	rhs_node = rhs;
}

inline std::vector<std::string>& stmtnode::get_lhs_labels (void) {
	return lhs_labels;
}

inline std::vector<std::string>& stmtnode::get_rhs_labels (void) {
	return rhs_labels;
}

inline std::vector<std::string>& stmtnode::get_lhs_names (void) {
	return lhs_names;
}

inline std::vector<std::string>& stmtnode::get_rhs_names (void) {
	return rhs_names;
}

inline void stmtnode::set_lhs_labels (std::vector<std::string> a) {
	lhs_labels = a;
}

inline void stmtnode::set_rhs_labels (std::vector<std::string> a) {
	rhs_labels = a;
}

inline void stmtnode::set_lhs_names (std::vector<std::string> a) {
	lhs_names = a;
}

inline void stmtnode::set_rhs_names (std::vector<std::string> a) {
	rhs_names = a;
}

inline int stmtnode::get_live_count (void) {
	return live_count;
}

inline int stmtnode::get_nonlive_count (void) {
	return nonlive_count;
}

inline int stmtnode::get_stmt_num (void) {
	return stmt_num;
}

inline void stmtnode::set_stmt_num (int val) {
	stmt_num = val;
}

inline int stmtnode::get_orig_stmt_num (void) {
	return orig_stmt_num;
}

inline void stmtnode::set_orig_stmt_num (int val) {
	orig_stmt_num = val;
}

inline bool stmtnode::is_executed (void) {
	return executed;
}

inline void stmtnode::set_executed (void) {
	executed = true;
}

class stmtlist {
	private:
		std::vector<stmtnode*> stmt_list;
	public:
		void push_stmt (stmtnode *);
		void push_stmt (std::vector<stmtnode *>);
		std::vector<stmtnode*> get_stmt_list (void);
		void set_stmt_list (std::vector<stmtnode*>);
};

inline void stmtlist::push_stmt (stmtnode *stmt) {
	stmt_list.push_back (stmt);
}

inline void stmtlist::push_stmt (std::vector<stmtnode *> stmt_vec) {
	for (std::vector<stmtnode *>::const_iterator i=stmt_vec.begin(); i!=stmt_vec.end(); i++)  
		stmt_list.push_back (*i);
}

inline std::vector<stmtnode*> stmtlist::get_stmt_list (void) {
	return stmt_list;
}

inline void stmtlist::set_stmt_list (std::vector<stmtnode*> sl) {
	stmt_list = sl;
}

class funcdefn {
	private:
		int dim=0, max_reg=255, reg_count=0, total_stmts=0, total_orig_stmts=0, ACC_SIZE=2, INTERLOCK_SIZE=2;
		bool SPLIT_ACCS=false, TOPOLOGICAL_SORT=false;
		DATA_TYPE gdata_type;
		std::vector<std::string> iters;
		std::vector<std::string> coefficients;
		stmtlist *stmt_list;
		string_list *arg_list;
		std::map<stmtnode*, std::vector<stmtnode*>> substmt_dependence_graph;
		std::map<int, std::vector<int>> cluster_dependence_graph;
		std::vector<stmtnode*> schedulable_stmts;
		std::deque<stmtnode*> fireable_stmts;
		std::deque<stmtnode*> fireable_ilp_stmts;
		std::vector<std::string> nonlive_labels;
		std::vector<std::string> live_labels;
		std::deque<std::string> register_pool;
		std::map<int, int> clusterwise_stmts_executed;
		std::map<int, boost::dynamic_bitset<>> labels_per_stmt; 
		std::map<std::string, std::vector<int>> stmts_per_label;
		std::map<int, int> clusterwise_stmt_count;
		std::map<std::string, std::string> register_mapping;
		std::map<std::string, expr_node*> label_to_node_map;
		std::map<std::string, int> label_reuse;
		std::map<std::string, std::vector<std::string>> gather_contributions;
		std::map<std::string, std::vector<std::string>> scatter_contributions;
		std::map<std::string, std::map<std::string, int>> primary_affinity;
		std::map<std::string, std::map<std::string, int>> secondary_affinity;
		std::vector<std::tuple<std::string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,std::vector<int>>> order_metric;
		std::vector<stmtnode*> initial_assignments;
		std::vector<expr_node*> temp_vars;
		std::map<int,int> initial_priority;
		std::deque<std::tuple<std::string,int>> interlock_lhs;
		// The first field of the tuple indicates the maximum number of
		// copies, the second indicates the current copy number	
		std::map<std::string, std::tuple<int,int>> acc_vars;
		// Data structures for tree-based register allocation
		int label_count = 0;
		std::map<std::string, int> label_bitset_index;
		unsigned int *label_frequency;
		// The forest of disjoint trees. In reality, instead of a tree, we can have a 
		// collection of trees that need to be interleaved. 
		std::map<int,std::vector<treenode*>> stmt_forest;
		// For any two forests, it stores their intersection
		std::vector<int> disjoint_forests;
	public:
		funcdefn (stmtlist *, string_list *);
		funcdefn (stmtlist *);
		void push_arg_list (char *);
		void set_iters (std::vector<std::string>);
		void set_coefficients (std::vector<std::string>);
		bool is_live (std::string);
		bool is_nonlive (std::string);
		bool single_use (std::string);
		bool limited_use (std::string, int);
		void set_dim (int);
		void set_gdata_type (DATA_TYPE);
		int get_dim (void);
		void set_split_accs (bool);
		void set_topological_sort (bool);
		int live_index (std::string);
		stmtnode *split_accumulations (stmtnode *);
		void split_input_summation (stmtnode *);
		void split_output_summation (stmtnode *);
		void fire_non_interlock_executable_stmts (void);
		void set_reg_limit (int);
		void print_func_defn (std::string);
		bool valid_permutation (std::vector<int>);
		void create_tree_clusters (void);
		void print_forests (std::stringstream &, std::vector<int>, boost::dynamic_bitset<>, boost::dynamic_bitset<>);
		void print_forests (std::stringstream &, std::vector<int>);
		void fixed_order_inter_forest_optimizations (std::vector<int>, std::map<int, std::tuple<boost::dynamic_bitset<>,unsigned int*>>, boost::dynamic_bitset<>);
		void fixed_order_inter_forest_optimizations (std::vector<int>, std::map<int, std::tuple<boost::dynamic_bitset<>,unsigned int*>>, boost::dynamic_bitset<>, boost::dynamic_bitset<>);
		void fixed_order_intra_type_inter_forest_optimizations (std::vector<int>, std::map<int, std::tuple<boost::dynamic_bitset<>,unsigned int*>>, boost::dynamic_bitset<>, boost::dynamic_bitset<>);
		void intra_forest_optimizations (std::vector<int>, std::map<int, std::tuple<boost::dynamic_bitset<>,unsigned int *>> &, boost::dynamic_bitset<>, boost::dynamic_bitset<>);
		void fixed_order_intra_tree_optimizations (int, boost::dynamic_bitset<> &, unsigned int *, boost::dynamic_bitset<>);
		bool determine_intra_opt_profitability (std::vector<std::tuple<treenode*,accnode*>> &, std::vector<int> &, boost::dynamic_bitset<> &, unsigned int *, std::map<std::string,std::tuple<std::string,expr_node*>> &, boost::dynamic_bitset<>, boost::dynamic_bitset<>);
		std::tuple<int,int> compute_register_optimal_schedule (std::vector<int>, map<treenode*, vector<int>> &, map<accnode*, vector<int>> &);
		std::tuple<int,int> compute_register_optimal_schedule (int, map<treenode*, vector<int>> &, map<accnode*, vector<int>> &);
		void retrace_register_optimal_schedule (std::vector<int>, map<treenode*, vector<int>> &, map<accnode*, vector<int>> &);
		void retrace_register_optimal_schedule (int, map<treenode*, vector<int>> &, map<accnode*, vector<int>> &);
		void retrace_register_optimal_schedule (std::vector<int>);
		void compute_liveness_map (std::map<int, std::tuple<boost::dynamic_bitset<>,unsigned int*>> &,std::vector<int>);
		void print_reuse_graph (std::map<std::tuple<int,int>, boost::dynamic_bitset<>>);
		void create_topological_sort (void);
		void compute_transitive_dependences (std::map<int, std::vector<int>> &, std::map<int, std::vector<int>>);
		bool transitive_dependence_exists (std::map<int, std::vector<int>>, int, int);
		void print_dependence_graph (std::string);
		bool verify_dependence (std::map<int, std::vector<int>> &, int, int);
		bool dependence_exists_in_dependence_graph (std::map<int, std::vector<int>> &, int, int);
		void print_cluster_dependence_graph (std::string);
		void print_dependence_graph (std::map<int, std::vector<int>>);
		void merge_nodes_in_dependence_graph (std::map<int, std::vector<int>> &, int, int);
		void merge_nodes_in_topological_clustering (std::map<int, std::vector<int>> &, int, int);
		void print_transitive_dependence_graph (std::map<int, std::vector<int>>);
		void print_schedulable_stmts (std::string);
		void print_stmt_label_map (std::string);
		void print_scatter_gather_contributions (std::string);
		void print_affinities (std::string);
		void compute_nonlive_labels (void);
		std::vector<stmtnode*> get_schedulable_stmts (void);
		std::map<stmtnode*, std::vector<stmtnode*>> get_substmt_dependence_graph (void);
		std::map<std::string, std::vector<std::string>> get_gather_contributions (void);
		std::map<std::string, std::vector<std::string>> get_scatter_contributions (void);
		std::map<std::string, std::map<std::string,int>> get_primary_affinity (void);
		std::map<std::string, std::map<std::string,int>> get_secondary_affinity (void);
		std::tuple<std::string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,std::vector<int>> fill_first_level_metric (std::string, std::map<int, std::vector<int>>);
		void fill_second_level_metric (std::vector<std::tuple<std::string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,std::vector<int>>> &);
		int get_primary_affinity_to_live_labels (std::string);
		int get_secondary_affinity_to_live_labels (std::string);
		int get_non_interlock_value (std::string);
		float get_primary_depth_to_live_labels (std::string);
		float get_secondary_depth_to_live_labels (std::string);
		int get_first_level_fire_potential (std::string);
		int get_first_level_non_interlock_fire_potential (std::string);
		int get_leading_stmt_fire_potential (std::string, std::map<int, std::vector<int>>);
		int assign_label_tree_priority (std::string);
		void get_second_level_fire_potential (std::string, std::vector<std::string>, int &, int &);
		int get_first_level_release_potential (std::string);
		int get_first_level_non_interlock_release_potential (std::string);
		void get_second_level_release_potential (std::string, std::vector<std::string>, int &, int &);
		int get_nonlive_values_touched (std::string);
		std::map<std::string, int> get_label_reuse (void);
		std::vector<std::tuple<std::string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,std::vector<int>>> get_order_metric (void);
		bool imminently_fireable (void);
		void fire_single_schedulable_statement (std::vector<std::tuple<std::string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,std::vector<int>>>); 
		void optimize_available_expressions (void);
		void distribute_rhs (void);
		void optimize_available_expressions (int, std::vector<int>, std::map<int,std::vector<std::tuple<treenode*,accnode*>>> &, std::map<int, std::tuple<boost::dynamic_bitset<>,unsigned int*>> &, std::map<std::string, std::tuple<std::string,expr_node*>> &, boost::dynamic_bitset<>);
		void optimize_intra_type_available_expressions (int, std::vector<int>, std::map<int,std::vector<std::tuple<treenode*,accnode*>>> &, std::map<int, std::tuple<boost::dynamic_bitset<>,unsigned int*>> &, std::map<std::string, std::tuple<std::string,expr_node*>> &, boost::dynamic_bitset<>);
		void update_forests (int, std::map<int, std::vector<std::tuple<treenode*,accnode*>>> &, std::map<std::string, std::tuple<std::string,expr_node*>> &);
		void get_lowest_cost_configuration (std::stringstream &, std::stringstream &);
		void copy_propagation (std::vector<std::tuple<expr_node*,expr_node*,STMT_OP>> &);
		void copy_propagation (std::vector<int> &);
		void simplify_accumulations (std::vector<std::tuple<expr_node*,expr_node*,STMT_OP>> &, int &);
		void decompose_statements (DATA_TYPE, int);
		void remove_redundant_stmts (void);
		void unroll_stmts (std::map<std::string,int>);
		void create_labels (void);
		void compute_stmt_label_map (void);
		void compute_decomposed_stmts_per_label (std::map<std::string,std::vector<int>> &);
		void compute_leading_stmt (std::map<int, std::vector<int>> &);
		void compute_participating_labels (void);
		void compute_dependences (void);
		void compute_cluster_dependences (void);
		void compute_reuse_graph (std::map<std::tuple<int,int>, boost::dynamic_bitset<>> &, std::map<int, boost::dynamic_bitset<>>);
		void update_dependences_and_schedulable_list (stmtnode *);
		void update_label_reuse (stmtnode *);
		void compute_schedulable_stmts (void);
		void compute_fireable_stmts (void);
		void add_fireable_stmt (stmtnode *);
		void make_label_live (std::string, int);
		void make_label_dead (void);
		void update_stmt_nonlive_count (std::string);
		void update_stmt_live_count (std::string);
		void compute_scatter_gather_contributions (void);
		void compute_label_reuse (void);
		void compute_primary_affinity (void);
		void compute_secondary_affinity (void);
		void compute_order_metric (void);
		void analyze_statements (std::stringstream &);
		void register_pressure_stats (void);
		void linear_scan_spill (void);
		void linear_scan_containment_spill (void);
		void linear_scan_split (void);
		void linear_scan_containment_split (void);
		void reorder_statements (std::stringstream &);
		void print_order_metric (void);
		void print_spill_metric (std::vector<std::tuple<std::string, int, int, int, int, int>>);
		void print_reordered_stmts (std::stringstream &, std::map<std::string, stmtnode*> &, std::deque<stmtnode*> &, PRINT_OPTION);
};

inline funcdefn::funcdefn (stmtlist *stmts, string_list *args) {
	stmt_list = stmts;
	arg_list = args;
}

inline funcdefn::funcdefn (stmtlist *stmts) {
	stmt_list = stmts;
}

inline void funcdefn::push_arg_list (char *str) {
	arg_list->push_back (std::string(str));
}

inline bool funcdefn::is_live (std::string s) {
	if (std::find (live_labels.begin(), live_labels.end(), s) == live_labels.end ())
		return false;
	return true;
}

inline bool funcdefn::is_nonlive (std::string s) {
	if (std::find (nonlive_labels.begin(), nonlive_labels.end(), s) == nonlive_labels.end ())
		return false;
	return true;
}

inline bool funcdefn::single_use (std::string s) {
	if (DEBUG) assert (label_reuse.find (s) != label_reuse.end () && "Label not present in reuse map");
	return (label_reuse[s] == 1);
}

inline bool funcdefn::limited_use (std::string s, int height) {
	if (DEBUG) assert (label_reuse.find (s) != label_reuse.end () && "Label not present in reuse map");
	return (label_reuse[s] <= height);
}

inline void funcdefn::set_dim (int d) {
	dim = d;
}

inline void funcdefn::set_gdata_type (DATA_TYPE g) {
	gdata_type = g;
}

inline int funcdefn::get_dim (void) {
	return dim;
}

inline void funcdefn::set_split_accs (bool a) {
	SPLIT_ACCS = a;
}

inline void funcdefn::set_topological_sort (bool a) {
	TOPOLOGICAL_SORT = a;
}

inline void funcdefn::set_reg_limit (int val) {
	max_reg = val;
}

inline void funcdefn::set_iters (std::vector<std::string> it) {
	iters = it;
}

inline void funcdefn::set_coefficients (std::vector<std::string> coef) {
	coefficients = coef;
}

inline int funcdefn::live_index (std::string s) {
	int ret = std::find (live_labels.begin(), live_labels.end(), s) - live_labels.begin ();
	return ret;
}

inline std::map<stmtnode*, std::vector<stmtnode*>> funcdefn::get_substmt_dependence_graph (void) {
	return substmt_dependence_graph;
}

inline std::map<std::string, std::vector<std::string>> funcdefn::get_gather_contributions (void) {
	return gather_contributions;
}

inline std::map<std::string, std::vector<std::string>> funcdefn::get_scatter_contributions (void) {
	return scatter_contributions;
}

inline std::map<std::string, int> funcdefn::get_label_reuse (void) {
	return label_reuse;
}

inline std::map<std::string, std::map<std::string,int>> funcdefn::get_primary_affinity (void) {
	return primary_affinity;
}

inline std::map<std::string, std::map<std::string,int>> funcdefn::get_secondary_affinity (void) {
	return secondary_affinity;
}

inline std::vector<std::tuple<std::string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,std::vector<int>>> funcdefn::get_order_metric (void) {
	return order_metric;
}

inline std::vector<stmtnode*> funcdefn::get_schedulable_stmts (void) {
	return schedulable_stmts;
}

class start_node {
	private:
		int max_reg=255;
		// Stencil definitions
		symtab <funcdefn*> *func_defns;
		// Parameters
		std::vector <std::string> parameters;
		// Var declarations
		symtab <DATA_TYPE> *var_decls;
		// Array declarations
		std::vector<array_decl *> array_decls;
		// Temporary declarations
		std::vector<std::string> temp_decls;
		// Unroll factor declarations
		std::map<std::string, int> unroll_decls;
		// Iterators
		std::vector<std::string> iters;
		// Coefficients
		std::vector<std::string> coefficients;
		// Stencil calls
		std::vector<func_call *> func_calls;
	public:
		start_node ();
		void push_func_defn (char *, funcdefn *);
		void push_func_defn (funcdefn *);
		void push_parameter (char *);
		void push_var_decl (char *, DATA_TYPE);
		void push_array_decl (array_decl *);
		void push_func_call (func_call *);
		void push_temp_decl (string_list *);
		void push_unroll_decl (char *, int);
		void push_unroll_decl (std::map<std::string, int> &);
		void push_iterator (char *);
		std::vector<std::string> get_iters (void);
		std::vector<std::string> get_coefficients (void);
		void push_coefficient (char *);
		void set_reg_limit (int);
		int get_reg_limit (void);
		bool is_array_decl (std::string);
		bool is_temp_decl (std::string);
		bool is_incoming_decl (std::string);
		DATA_TYPE get_array_type (std::string);
		DATA_TYPE get_var_type (std::string);
		std::vector<std::string> get_temp_decl (void);
		std::map<std::string, int> get_unroll_decl (void);
		std::vector<std::string> get_parameters (void);
		symtab <funcdefn *> *get_funcdefns (void);
		funcdefn *get_func_defn (std::string); 
		std::vector<array_decl *> get_array_decl (void);
		std::vector<array_range *> get_array_range (std::string s);
		int get_array_dimensionality (std::string s);
		int get_max_dimensionality (void);
		range_list *get_range_list (std::string s);
		std::vector<func_call *> get_func_calls (void);
		void compute_intermediate_arrays (std::vector<std::string>&);
		symtab <DATA_TYPE> *get_var_decls (void);
};

inline start_node::start_node () {
	func_defns = new symtab <funcdefn*>;
	var_decls = new symtab <DATA_TYPE>;
}

inline symtab <DATA_TYPE> *start_node::get_var_decls (void) {
	return var_decls;
}

inline std::vector<std::string> start_node::get_parameters (void) {
	return parameters;
}

inline std::vector<array_decl *> start_node::get_array_decl (void) {
	return array_decls;
}

inline std::vector<func_call *> start_node::get_func_calls (void) {
	return func_calls;
}

inline std::vector<std::string> start_node::get_iters (void) {
	return iters;
}

inline std::vector<std::string> start_node::get_coefficients (void) { 
	return coefficients;
}

inline std::vector<std::string> start_node::get_temp_decl (void) {
	return temp_decls;
}

inline std::map<std::string, int> start_node::get_unroll_decl (void) {
	return unroll_decls;
}

inline symtab <funcdefn *> *start_node::get_funcdefns (void) {
	return func_defns;
} 

inline void start_node::push_func_defn (char *s, funcdefn *def) {
	func_defns->push_symbol (s, def);
}

inline void start_node::push_func_defn (funcdefn *def) {
	char *s = (char*)"stencil";
	func_defns->push_symbol (s, def);
}

inline void start_node::push_array_decl (array_decl *a) {
	array_decls.push_back (a);
}

inline void start_node::push_parameter (char *s) {
	parameters.push_back (std::string (s));
}

inline void start_node::push_var_decl (char *s, DATA_TYPE t) {
	var_decls->push_symbol (s, t);
}

inline void start_node::push_func_call (func_call *st_call) {
	func_calls.push_back (st_call);
}

#endif
