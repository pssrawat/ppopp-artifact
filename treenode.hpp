#ifndef __TREENODE_HPP__
#define __TREENODE_HPP__
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <tuple>
#include <queue>
#include <map>
#include <iostream>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/algorithm/string/replace.hpp>
#include "utils.hpp" 
#include "sort.hpp"

class expr_node;
class stmtnode;
// The hierarchy is a bit convoluted. 
// A node of the cluster : {LHS name, vector<stmt assignment op, RHS operator, RHS operands>}
// where each RHS operand is in turn a node of the cluster.
// If size (RHS operands) == 2, then it is a binary node.
// If size (vector<stmt assignment op, RHS operator, RHS operands>) == 1, then it is not accumulation 
class treenode;
class accnode {
	private:
		STMT_OP asgn_op;
		OP_TYPE rhs_operator;
		std::vector <treenode*> rhs_operands;
		std::string expr_string;
		std::string rhs_printing;
		boost::dynamic_bitset<> used_labels;
		unsigned int *use_frequency;
		boost::dynamic_bitset<> appended_labels;
		unsigned int *appended_frequency;
		//std::stringstream code_string;
		vector<treenode*> spliced_treenodes;
		// The memoization map
		std::map<std::tuple<boost::dynamic_bitset<>,boost::dynamic_bitset<>>, std::tuple<int,int,std::vector<int>,map<treenode*,vector<int>>,map<accnode*,vector<int>>>> optimal_reg_cache;
		//std::map<std::tuple<boost::dynamic_bitset<>,boost::dynamic_bitset<>>, std::tuple<int,int,std::vector<int>>> retrace_reg_cache;
	public:
		accnode (int);
		accnode (int, std::string, treenode*, STMT_OP);
		void set_assignment_op (STMT_OP);
		void set_rhs_operator (OP_TYPE);
		STMT_OP get_assignment_op (void);
		OP_TYPE get_rhs_operator (void);
		bool is_binary_op (void);
		bool is_unary_op (void);
		bool is_asgn_eq_op (void);
		void append_to_expr_string (std::string);
		void append_to_rhs_printing (std::string);
		//std::string get_code_string (void);
		std::string get_expr_string (void);
		std::string get_rhs_printing_string (void);
		void set_expr_string (std::string);
		boost::dynamic_bitset<> get_appended_labels (void);
		unsigned int *get_appended_frequency (void);
		void add_operand (treenode *, std::map<std::string, int>, int);
		std::vector<treenode*> &get_operands (void);
		boost::dynamic_bitset<> get_used_labels (void);
		void reset_rhs_operand ();
		void reset_rhs_operand (std::vector<treenode*> &);
		unsigned int *get_use_frequency (void);
		bool subtree_has_only_leafs (void);
		void add_spliced_treenode (treenode *);
		std::vector<treenode*> &get_spliced_treenodes (void);
		void compute_leaf_nodes (boost::dynamic_bitset<> &, std::map<std::string, int>);
		void recompute_rhs_expr (std::map<std::string, int>, std::vector<std::string> &, int);
		bool recompute_rhs_expr (std::map<std::string, int>, std::vector<std::tuple<treenode*,accnode*>> &, std::vector<std::string> &, int);
		std::tuple<int, int> compute_register_optimal_schedule (boost::dynamic_bitset<> &, unsigned int *, std::map<std::string, int>, unsigned int *, int, map<treenode*, vector<int>> &, map<accnode*, vector<int>> &);
		//std::tuple<int, int> retrace_register_optimal_schedule (boost::dynamic_bitset<> &, unsigned int *, std::map<std::string, int>, unsigned int *, int);
		void retrace_register_optimal_schedule (map<treenode*, vector<int>> &, map<accnode*, vector<int>> &);
		void append_to_code (treenode*);
		void update_appended_info (treenode*, std::map<std::string, int>, int);
};

inline accnode::accnode (int label_count) {
	used_labels.resize (label_count);
	use_frequency = new unsigned int[label_count] ();
	appended_labels.resize (label_count);
	appended_frequency = new unsigned int[label_count] ();
}

inline accnode::accnode (int label_count, std::string s, treenode *t, STMT_OP op) {
	used_labels.resize (label_count);
	use_frequency = new unsigned int[label_count] ();
	expr_string = s;
	rhs_printing = s; 
	rhs_operands.push_back (t);
	asgn_op = op;
	appended_labels.resize (label_count);
	appended_frequency = new unsigned int[label_count] ();
}

inline void accnode::reset_rhs_operand (std::vector<treenode *> &new_rhs_operands) {
	rhs_operands.clear ();
	rhs_operands = new_rhs_operands;
}

inline void accnode::reset_rhs_operand (void) {
	rhs_operands.clear ();
	expr_string = "";
	rhs_printing = "";
}

//inline std::string accnode::get_code_string (void) {
//	return code_string.str();
//}

inline boost::dynamic_bitset<> accnode::get_used_labels (void) {
	return used_labels;
}

inline unsigned int *accnode::get_use_frequency (void) {
	return use_frequency;
}

inline void accnode::set_assignment_op (STMT_OP op) {
	asgn_op = op;
}

inline void accnode::set_rhs_operator (OP_TYPE op) {
	rhs_operator = op;
}

inline STMT_OP accnode::get_assignment_op (void) {
	return asgn_op;
}

inline bool accnode::is_asgn_eq_op (void) {
	return asgn_op == ST_EQ;
}

inline vector<treenode*> &accnode::get_spliced_treenodes (void) {
	return spliced_treenodes;
}

inline OP_TYPE accnode::get_rhs_operator (void) {
	return rhs_operator;
}

inline void accnode::append_to_expr_string (std::string s) {
	expr_string = expr_string + s;
}

inline void accnode::append_to_rhs_printing (std::string s) {
	rhs_printing = rhs_printing + s;
}

inline void accnode::set_expr_string (std::string s) {
	expr_string = s;
	rhs_printing = s;
}

inline std::string accnode::get_expr_string (void) {
	return expr_string;
}

inline std::string accnode::get_rhs_printing_string (void) {
	return rhs_printing;
}

inline bool accnode::is_binary_op (void) {
	return (rhs_operands.size () == 2);
}

inline bool accnode::is_unary_op (void) {
	return (rhs_operands.size () == 1);
}

inline boost::dynamic_bitset<> accnode::get_appended_labels (void) {
	return appended_labels;
}

inline unsigned int *accnode::get_appended_frequency (void) {
	return appended_frequency;
}

inline std::vector<treenode*>& accnode::get_operands (void) {
	return rhs_operands;
}

// If rhs_accs is 1, then the node is a simple assignment.
// Otherwise, it is an accumulation.
class treenode {
	private:
		expr_node *lhs;
		std::string lhs_label;
		std::string lhs_printing;
		std::vector<accnode*> rhs_accs;
		bool is_uminus=false;
		// These are independent of cluster
		boost::dynamic_bitset<> used_labels;
		unsigned int *use_frequency;
		boost::dynamic_bitset<> appended_labels;
		unsigned int *appended_frequency;
		//std::stringstream code_string;
		// The memoization map
		std::map<std::tuple<boost::dynamic_bitset<>,boost::dynamic_bitset<>>, std::tuple<int,int,std::vector<int>,map<treenode*,vector<int>>,map<accnode*,vector<int>>>> optimal_reg_cache;
		//std::map<std::tuple<boost::dynamic_bitset<>,boost::dynamic_bitset<>>, std::tuple<int,int,std::vector<int>>> retrace_reg_cache;
		vector<treenode*> spliced_treenodes;
	public:
		treenode (expr_node *, std::string, std::string, std::map<std::string, int>, int, bool);
		void set_lhs (expr_node *);
		void reset_rhs_accs (accnode *);
		void reset_rhs_accs (void);
		void reset_rhs_accs (std::vector<accnode *> &);
		bool is_accumulation_node (void);
		bool is_leaf_node (void);
		bool is_data_node (void);
		bool is_uminus_node (void);
		void set_uminus_val (bool);
		expr_node *get_lhs (void);
		void append_to_code (treenode *);
		void add_rhs_expr (accnode *, std::map<std::string, int>, int);
		boost::dynamic_bitset<> get_used_labels (void);
		boost::dynamic_bitset<> get_appended_labels (void);
		unsigned int *get_use_frequency (void);
		unsigned int *get_appended_frequency (void);
		std::string get_lhs_label (void);
		void set_lhs_label (std::string);
		std::vector<accnode*> &get_rhs_operands (void);
		std::vector<treenode*> &get_spliced_treenodes (void);
		void add_spliced_treenode (treenode *);
		void compute_leaf_nodes (boost::dynamic_bitset<> &, std::map<std::string, int>);
		std::tuple<boost::dynamic_bitset<>,unsigned int *> compute_liveness (boost::dynamic_bitset<> &, unsigned int *, unsigned int *, int); 
		std::tuple<int, int> compute_register_optimal_schedule (boost::dynamic_bitset<> &, unsigned int *, std::map<std::string, int>, unsigned int *, int, map<treenode*, vector<int>> &, map<accnode*, vector<int>> &);
		//std::tuple<int, int> retrace_register_optimal_schedule (boost::dynamic_bitset<> &, unsigned int *, std::map<std::string, int>, unsigned int *, int);
		void retrace_register_optimal_schedule (map<treenode*, vector<int>> &, map<accnode*, vector<int>> &);
		void identify_optimizable_subtrees (std::vector<std::tuple<treenode*,accnode*>>&, boost::dynamic_bitset<>, boost::dynamic_bitset<>, int);
		void identify_optimizable_subtrees (std::vector<std::tuple<treenode*,accnode*>>&, boost::dynamic_bitset<>, boost::dynamic_bitset<>, boost::dynamic_bitset<>, int);
		void identify_optimizable_subtrees (std::vector<std::tuple<treenode*,accnode*>>&, boost::dynamic_bitset<>, boost::dynamic_bitset<>, unsigned int *, int);
		void identify_optimizable_subtrees (std::vector<std::tuple<treenode*,accnode*>>&, boost::dynamic_bitset<>, boost::dynamic_bitset<>, boost::dynamic_bitset<>, unsigned int *, int);
		void identify_leafy_computation_subtrees (std::vector<std::tuple<treenode*,accnode*>>&, boost::dynamic_bitset<>, std::map<std::string, int>, int);	
		void identify_optimizable_leafy_subtrees (std::vector<std::tuple<treenode*,accnode*>>&, std::vector<std::tuple<treenode*,accnode*>>&, boost::dynamic_bitset<>&, std::map<std::string, int>, int);
		void recompute_tree (std::map<std::string, int>, std::vector<std::tuple<treenode*,accnode*>> &, std::vector<std::string> &, int);
		void recompute_tree (std::map<std::string, int>, std::vector<std::string> &, int);
		void update_host_tree (treenode *, int);
		void create_expr_lhs_map (std::map<std::string, std::tuple<std::string,expr_node*>> &);
		std::string print_tree (std::vector<treenode*> &, boost::dynamic_bitset<>, boost::dynamic_bitset<>, int);
		//std::string print_tree (void);
		void print_finalized_tree (std::stringstream &, std::vector<expr_node*> &, std::vector<expr_node*> &, std::map<std::string,std::string> &, DATA_TYPE);
		void print_temp_type (std::vector<expr_node*>&, expr_node *, std::stringstream &, DATA_TYPE);
		void copy_propagation (std::map<std::string, treenode*> &, unsigned int *, std::map<std::string, int>);
		void allocate_registers (std::stringstream &, int &, std::queue<int> &, std::map<std::string,int> &, unsigned int *, unsigned int *, std::map<std::string, int>);
		bool append_subtree_to_code (treenode*, boost::dynamic_bitset<>, boost::dynamic_bitset<>, std::map<std::string, int>, int);
		void update_appended_info (treenode*, std::map<std::string, int>, int);
};

inline treenode::treenode (expr_node *node, std::string p, std::string s, std::map<std::string, int> label_bitset_index, int label_count, bool is_um) {
	lhs = node;
	lhs_label = s;
	lhs_printing = p;
	is_uminus = is_um;
	used_labels.resize (label_count);
	if (!is_data_node ()) {
		if (DEBUG) assert (label_bitset_index.find (lhs_label) != label_bitset_index.end() && "Entry not found in label_bitset_index (treenode contructor)");
		used_labels[label_bitset_index[lhs_label]] = true;
	}
	use_frequency = new unsigned int[label_count] ();
	appended_labels.resize (label_count);
	appended_frequency = new unsigned int[label_count] ();
}

inline boost::dynamic_bitset<> treenode::get_appended_labels (void) {
	return appended_labels;
}

inline unsigned int *treenode::get_appended_frequency (void) {
	return appended_frequency;
}

inline vector<treenode*> &treenode::get_spliced_treenodes (void) {
	return spliced_treenodes;
}

inline void treenode::reset_rhs_accs (std::vector<accnode *> &new_rhs_accs) {
	rhs_accs.clear ();
	rhs_accs = new_rhs_accs;
}

inline void treenode::reset_rhs_accs (void) {
	rhs_accs.clear ();
}

inline void treenode::reset_rhs_accs (accnode *t_rhs) {
	rhs_accs.clear ();
	rhs_accs.push_back (t_rhs);
}

inline std::vector<accnode*>& treenode::get_rhs_operands (void) {
	return rhs_accs;
}

inline expr_node* treenode::get_lhs (void) {
	return lhs;
}

inline void treenode::set_lhs (expr_node *node) {
	lhs = node;
}

inline std::string treenode::get_lhs_label (void) {
	return lhs_label;
}

inline void treenode::set_lhs_label (std::string s) {
	lhs_label = s;
}

inline boost::dynamic_bitset<> treenode::get_used_labels (void) {
	return used_labels;
}

inline unsigned int *treenode::get_use_frequency (void) {
	return use_frequency;
}

inline bool treenode::is_accumulation_node (void) {
	return (rhs_accs.size () > 1);
}

inline void treenode::set_uminus_val (bool is_um) {
	is_uminus = is_um;
}

inline bool treenode::is_leaf_node (void) {
	return (rhs_accs.size () == 0);
}

inline bool treenode::is_data_node (void) {
	return (lhs_label.compare ("") == 0);
}

inline bool treenode::is_uminus_node (void) {
	return is_uminus;
}

#endif 
