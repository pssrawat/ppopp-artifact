#include "codegen.hpp"
using namespace std;

void codegen::print_parameters (void) {
	vector <string> parameters = start->get_parameters ();
	cout << "Parameters : ";
	for (vector<string>::const_iterator i=parameters.begin(); i!=parameters.end(); ++i)
		cout << *i << " ";
	cout << endl;
}

void codegen::print_temp_decls (void) {
	vector <string> temp_vec = start->get_temp_decl ();
	cout << "Temporary arrays : ";
	for (vector<string>::const_iterator i=temp_vec.begin(); i!=temp_vec.end(); ++i)
		cout << *i << " ";
	cout << endl;
}

void codegen::print_unroll_decls (void) {
	map <string, int> temp_map = start->get_unroll_decl ();
	cout << "Unroll factors : ";
	for (map<string, int>::const_iterator i=temp_map.begin(); i!=temp_map.end(); ++i)
		cout << "(" << i->first << " : " << i->second << ") ";
	cout << endl;
}

void codegen::print_var_decls (void) {
	symtab <DATA_TYPE> *var_decls = start->get_var_decls ();
	cout << "Variable declaration : ";
	map<string, DATA_TYPE> list = var_decls->get_symbol_list (); 
	for (map<string, DATA_TYPE>::const_iterator i=list.begin(); i!=list.end(); i++) {
		cout << "(" << print_data_type (i->second) << " " << i->first << ") ";
	}
	cout << endl;
}

void codegen::print_array_decls (void) {
	vector<array_decl *> array_decls = start->get_array_decl ();
	for (vector<array_decl *>::const_iterator i=array_decls.begin(); i!=array_decls.end(); i++) {
		cout << "Array declaration : ";
		string name = (*i)->get_array_name ();
		DATA_TYPE t = (*i)->get_array_type ();
		cout << print_data_type (t) << " " << name;
		vector<array_range *> r = (*i)->get_array_range ();
		for (vector<array_range *>::const_iterator j = r.begin (); j!=r.end (); j++) {
			expr_node *lo = (*j)->get_lo_range ();
			stringstream lo_out;
			lo->print_node (lo_out);
			expr_node *hi = (*j)->get_hi_range ();
			stringstream hi_out;
			hi->print_node (hi_out);
			cout << "[" << lo_out.str() << ".." << hi_out.str() << "]";
		}
		cout << endl;
	}
}

void codegen::print_func_calls (void) {
	vector<func_call *> func_calls = start->get_func_calls ();
	for (vector<func_call*>::const_iterator i=func_calls.begin(); i!=func_calls.end(); i++) {
		vector<string> arg_list = (*i)->get_arg_list ();
		vector<string> out_list = (*i)->get_out_list ();
		string name = (*i)->get_name ();
		cout << "func call : ( ";
		for (vector<string>::const_iterator j=out_list.begin(); j!=out_list.end(); j++) 
			cout << (*j) << " ";
		cout << ") = " << name << " ( ";
		for (vector<string>::const_iterator k=arg_list.begin(); k!=arg_list.end(); k++) 
			cout << (*k) << " ";
		cout << ")" << endl;
	}
}

// The main driver routine for the register reordering
void codegen::generate_code (stringstream &reorder, stringstream &original, map<string,int> &ud, DATA_TYPE gdata_type, int HEURISTIC, bool DISTRIBUTE_RHS, bool SPLIT_ACCS, bool TOPOLOGICAL_SORT) {
	start->push_unroll_decl (ud);
	// Compile the list of known types 
	map <string, DATA_TYPE> known_types;
	// 1. iterate over var decls 
	symtab <DATA_TYPE> *var_decls = start->get_var_decls ();
	map<string, DATA_TYPE> list = var_decls->get_symbol_list (); 
	for (map<string, DATA_TYPE>::const_iterator i=list.begin(); i!=list.end(); i++) 
		known_types.insert (make_pair (i->first, i->second));
	// 2. iterate over declared arrays 
	vector<array_decl *> array_decls = start->get_array_decl ();
	for (vector<array_decl *>::const_iterator i=array_decls.begin(); i!=array_decls.end(); i++) 
		known_types.insert (make_pair ((*i)->get_array_name (), (*i)->get_array_type ()));

	symtab <funcdefn *> *func_defns = start->get_funcdefns ();
	map<string, funcdefn*> func_defn = func_defns->get_symbol_list ();
	// Iterate over each function and generate a reordered version
	int max_dim = start->get_max_dimensionality ();
	for (map<string, funcdefn*>::const_iterator i=func_defn.begin(); i!=func_defn.end(); i++) {
		string name = i->first;
		funcdefn *defn = i->second;
		defn->set_gdata_type (gdata_type);
		defn->set_split_accs (SPLIT_ACCS);
		defn->set_topological_sort (TOPOLOGICAL_SORT);
		defn->set_dim (max_dim);
		defn->set_iters (start->get_iters ());
		defn->set_coefficients (start->get_coefficients ());
		defn->set_reg_limit (start->get_reg_limit ());
		// Commenting the remove_redundant_stmts function, since this may be incorrect. We need to 
		// know whether the value written was reused again.
		//defn->remove_redundant_stmts ();
		// Now unroll the statement
		defn->unroll_stmts (start->get_unroll_decl ());
		// Decompose statement into accumulations	
		defn->decompose_statements (gdata_type, HEURISTIC);
		if (AVAIL_EXPR_OPT) defn->optimize_available_expressions ();
		if (DISTRIBUTE_RHS) defn->distribute_rhs ();
		// Create labels for accesses
		defn->create_labels ();
		defn->compute_participating_labels ();
		defn->compute_label_reuse ();
		if (DEBUG) defn->print_func_defn (name);
		if (HEURISTIC == 0) {
			// Create clusters of trees
			defn->create_tree_clusters ();
			defn->compute_cluster_dependences ();
			if (DEBUG) defn->print_cluster_dependence_graph (name);
			defn->get_lowest_cost_configuration (original, reorder);
		}
		if (HEURISTIC == 1) {
			// Compute dependences, and initial schedulable statements
			defn->compute_stmt_label_map ();
			if (DEBUG) defn->print_stmt_label_map (name);
			defn->compute_dependences ();
			if (DEBUG) defn->print_dependence_graph (name);
			defn->compute_schedulable_stmts ();
			defn->compute_fireable_stmts ();
			if (DEBUG) defn->print_schedulable_stmts (name);
			// Compute the affinities and label reuse count
			defn->compute_nonlive_labels ();
			defn->compute_scatter_gather_contributions ();
			defn->analyze_statements (original);
			if (DEBUG) defn->print_scatter_gather_contributions (name);
			defn->compute_primary_affinity ();
			defn->compute_secondary_affinity ();
			if (DEBUG) defn->print_affinities (name);
			// Finally run the reordering algorithm
			defn->reorder_statements (reorder);
		}
	}
}
