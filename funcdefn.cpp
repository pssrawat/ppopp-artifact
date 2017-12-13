#include "funcdefn.hpp"
#include "tree-reg-funcdefn.cpp"

void stmtnode::set_nonlive_count (void) {
	nonlive_count = (int)lhs_labels.size() + (int)rhs_labels.size();
}

void stmtnode::set_nonlive_count (int val) {
	nonlive_count = val;
}

void stmtnode::print_statement (stringstream &output) {
	lhs_node->print_node (output);
	output << print_stmt_op (op_type);
	rhs_node->print_node (output);
	output << ";\n";
}

void stmtnode::set_expr_data_types (void) {
	DATA_TYPE d_type = infer_data_type (lhs_node->get_type (), rhs_node->get_type ());
	lhs_node->set_type (d_type);
	rhs_node->set_type (d_type);	
}

string stmtnode::print_statement (stringstream &output, vector<string> &initialized_labels, vector<string> iters) {
	stringstream header_output;
	if (!PRINT_INTRINSICS) {
		rhs_node->print_initializations (header_output, initialized_labels, iters, EXPLICIT_LOADS, false);
		stringstream rhs_output;
		rhs_node->print_node (rhs_output, initialized_labels, iters, EXPLICIT_LOADS, false);
		if (get_op_type() == ST_EQ)
			lhs_node->print_initializations (header_output, initialized_labels, iters, false, true);
		else
			lhs_node->print_initializations (header_output, initialized_labels, iters, true, true);
		lhs_node->print_node (output, initialized_labels, iters, false, true);
		output << print_stmt_op (op_type);
		output << rhs_output.str();
		output << ";\n";
	}
	else {
		rhs_node->print_initializations (header_output, initialized_labels, iters, EXPLICIT_LOADS, false);
		if (get_op_type() == ST_EQ)
			lhs_node->print_initializations (header_output, initialized_labels, iters, false, true);
		else
			lhs_node->print_initializations (header_output, initialized_labels, iters, true, true);
		// Convert the statement back to assignment
		bool is_binary = (rhs_node->get_expr_type () == T_BINARY);
		OP_TYPE rhs_op = is_binary ? dynamic_cast<binary_node*>(rhs_node)->get_operator () : T_EQ;
		OP_TYPE fma_op = T_EQ; 
		if (op_type != ST_EQ) {
			fma_op = convert_stmt_op_to_op (op_type);
			rhs_node = new binary_node (fma_op, lhs_node, rhs_node);
			set_expr_data_types ();
			op_type = ST_EQ;
		}
		// Now check if we can emit an FMA
		if (GEN_FMA && is_binary && rhs_op == T_MULT && (fma_op == T_PLUS || fma_op == T_MINUS)) {
			lhs_node->print_node (output, initialized_labels, iters, false, true);
			output << print_stmt_op (op_type);
			expr_node *t_lhs = dynamic_cast<binary_node*>(rhs_node)->get_lhs ();
			expr_node *t_rhs = dynamic_cast<binary_node*>(rhs_node)->get_rhs ();
			expr_node *rhs_m1 = dynamic_cast<binary_node*>(t_rhs)->get_lhs();
			expr_node *rhs_m2 = dynamic_cast<binary_node*>(t_rhs)->get_rhs();
			string tail = (lhs_node->get_type () == DOUBLE) ? "pd" : "ps";
			if (fma_op == T_PLUS)  output << "_mm256_fmadd_" << tail << " (";
			else output << "_mm256_fnmadd_" << tail << " (";
			rhs_m1->print_node (output, initialized_labels, iters, EXPLICIT_LOADS, false);
			output << ", ";
			rhs_m2->print_node (output, initialized_labels, iters, EXPLICIT_LOADS, false);
			output << ", ";
			t_lhs->print_node (output, initialized_labels, iters, EXPLICIT_LOADS, false);
			output << ")";  	
		}
		else { 
			lhs_node->print_node (output, initialized_labels, iters, false, true);
			output << print_stmt_op (op_type);
			rhs_node->print_node (output, initialized_labels, iters, EXPLICIT_LOADS, false);
		}
		output << ";\n";
	}
	return header_output.str ();
}

void stmtnode::print_statement (map<string, string> &reg_map, map<string, expr_node*> &label_to_node_map, map<string,int> &first_load, map<string,int> &last_write, vector<string> &printed_regs, stringstream &output) {
	vector<string> l_labels = get_lhs_labels ();
	vector<string> r_labels = get_rhs_labels ();
	// For each r_label, print out the first load.
	for (vector<string>::iterator i=r_labels.begin(); i!=r_labels.end(); i++) {
		if (first_load.find (*i) == first_load.end ()) continue;
		if (first_load[*i] == stmt_num) {
			if (DEBUG) assert (label_to_node_map.find (*i) != label_to_node_map.end ());
			if (DEBUG) assert (reg_map.find (*i) != reg_map.end ());
			expr_node *node = label_to_node_map[*i];
			if (find (printed_regs.begin(), printed_regs.end(), reg_map[*i]) == printed_regs.end ()) {
				output << print_data_type (node->get_type ());
				printed_regs.push_back (reg_map[*i]); 
			}
			output << reg_map[*i] << " = ";
			node->print_node (output);
			output << ";\n";
		}
	}
	// Print type of lhs if register is appearing for the first time
	for (vector<string>::iterator i=l_labels.begin(); i!=l_labels.end(); i++) {
		if (find (printed_regs.begin(), printed_regs.end(), reg_map[*i]) == printed_regs.end ()) {
			output << print_data_type (lhs_node->get_type ());
			printed_regs.push_back (reg_map[*i]); 
		}
	}
	// Print the statement
	lhs_node->print_node (reg_map, output);
	output << print_stmt_op (op_type);
	rhs_node->print_node (reg_map, output);
	output << ";\n";
	// For each l_label, print out the last write.
	for (vector<string>::iterator i=l_labels.begin(); i!=l_labels.end(); i++) {
		if (DEBUG) assert (last_write.find (*i) != last_write.end ());
		if (last_write[*i] == stmt_num) {
			if (DEBUG) assert (label_to_node_map.find (*i) != label_to_node_map.end ());
			expr_node *node = label_to_node_map[*i];
			if (DEBUG) assert (reg_map.find (*i) != reg_map.end ());
			node->print_node (output);
			output << " = " << reg_map[*i] << ";\n";
		}
	}
}

// Check if the label is present in the statement
bool stmtnode::is_label_present (string s) {
	vector<string> l_labels = get_lhs_labels ();
	vector<string> r_labels = get_rhs_labels ();
	for (vector<string>::iterator i=l_labels.begin(); i!=l_labels.end(); i++) 
		if (s.compare (*i) == 0) return true;
	for (vector<string>::iterator i=r_labels.begin(); i!=r_labels.end(); i++) 
		if (s.compare (*i) == 0) return true;
	return false;
}

bool stmtnode::is_label_present (string s, int &frequency) {
	vector<string> l_labels = get_lhs_labels ();
	vector<string> r_labels = get_rhs_labels ();
	bool ret = false;
	frequency = 0;
	for (vector<string>::iterator i=l_labels.begin(); i!=l_labels.end(); i++) { 
		if (s.compare (*i) == 0) {
			ret = true;
			frequency++;
		}
	}
	for (vector<string>::iterator i=r_labels.begin(); i!=r_labels.end(); i++) { 
		if (s.compare (*i) == 0) {
			ret = true;
			frequency++;
		}
	}
	return ret;
}

void funcdefn::print_func_defn (string name) {
	cout << "func definition : " << name << "{\n";
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator k=stmts.begin(); k!=stmts.end(); k++) {
		expr_node *lhs = (*k)->get_lhs_expr ();
		string op = print_stmt_op ((*k)->get_op_type());
		expr_node *rhs = (*k)->get_rhs_expr ();
		stringstream lhs_out;
		lhs->print_node (lhs_out);
		stringstream rhs_out;
		rhs->print_node (rhs_out);
		cout << "    <" << lhs->get_type () << "> " << lhs_out.str() << op << "<" << rhs->get_type() << "> " << rhs_out.str() << ";";
		cout <<  "\t{STMT_NO : " << (*k)->get_stmt_num() << ", ORIG_STMT_NO : " << (*k)->get_orig_stmt_num() << " | OUT_LABELS : ";
		vector<string> lhs_label = (*k)->get_lhs_labels ();
		for (vector<string>::const_iterator l=lhs_label.begin(); l!=lhs_label.end(); l++) {
			cout << *l << " ";
		}
		cout << "| IN_LABELS : ";
		vector<string> rhs_label = (*k)->get_rhs_labels ();
		for (vector<string>::const_iterator l=rhs_label.begin(); l!=rhs_label.end(); l++) {
			cout << *l << " ";
		}
		cout << "} (live: " << (*k)->get_live_count () << ", nonlive: " <<(*k)->get_nonlive_count() << ")" << endl;
	}
	cout << "}\n";
}

void funcdefn::print_transitive_dependence_graph (map<int, vector<int>> transitive_dependence_graph) { 
	cout << "\nTransitive dependence graph : " << endl;
	for (map<int, vector<int>>::iterator j=transitive_dependence_graph.begin(); j!=transitive_dependence_graph.end(); j++) {
		int lhs = j->first;
		printf ("%d - ", lhs);
		vector<int> rhs = j->second;
		for (vector<int>::iterator k=rhs.begin(); k!=rhs.end(); k++) {
			printf ("%d ", *k);
		}
		printf ("\n");
	}
} 

// Return true if source has to be executed before destination
bool funcdefn::dependence_exists_in_dependence_graph (map<int, vector<int>> &dependence_graph, int source, int dest) {
	if (dependence_graph.find (dest) != dependence_graph.end ()) {
		if (find (dependence_graph[dest].begin(), dependence_graph[dest].end (), source) != dependence_graph[dest].end ())
			return true;	
	}
	return false;;
}

// Return false if there is an edge from dest to source, i.e. source has dest in its rhs
bool funcdefn::verify_dependence (map<int, vector<int>> &dependence_graph, int source, int dest) {
	if (dependence_graph.find (source) != dependence_graph.end ()) {
		if (find (dependence_graph[source].begin(), dependence_graph[source].end (), dest) != dependence_graph[source].end ())
			return false;	
	}
	return true;
}

void funcdefn::merge_nodes_in_topological_clustering (map<int, vector<int>> &clustering, int source, int dest) {
	// First add just the dest to source cluster
	if (clustering.find (source) != clustering.end ()) {
		// First check if there is a dependence between the source cluster rhs and dest
		bool dependence_exists = false ;
		for (vector<int>::iterator it=clustering[source].begin(); it!=clustering[source].end(); it++) {
			dependence_exists |= dependence_exists_in_dependence_graph (cluster_dependence_graph, dest, *it);
		}
		if (dependence_exists) {
			bool inserted = false;
			for (vector<int>::iterator it=clustering[source].begin(); it!=clustering[source].end(); it++) {
				if (dependence_exists_in_dependence_graph (cluster_dependence_graph, dest, *it)) {
					clustering[source].insert (it, dest);
					inserted = true; 
					break;
				}
			}
			if (!inserted) clustering[source].push_back (dest);
		}
		else clustering[source].push_back (dest);
	}
	else {
		vector<int> dest_vec;
		dest_vec.push_back (dest);
		clustering[source] = dest_vec;
	}
	if (clustering.find (dest) != clustering.end ()) {
		for (vector<int>::iterator it=clustering[dest].begin(); it!=clustering[dest].end(); it++) {
			int val = *it;
			bool dependence_exists = false ;
			for (vector<int>::iterator jt=clustering[source].begin(); jt!=clustering[source].end(); jt++) {
				dependence_exists |= dependence_exists_in_dependence_graph (cluster_dependence_graph, val, *jt);
			}
			if (dependence_exists) {
				bool inserted = false;
				for (vector<int>::iterator jt=clustering[source].begin(); jt!=clustering[source].end(); jt++) {
					if (dependence_exists_in_dependence_graph (cluster_dependence_graph, dest, *jt)) {
						clustering[source].insert (jt, val);
						inserted = true;
						break;
					}
				}
				if (!inserted) clustering[source].push_back (val);
			}
			else clustering[source].push_back (val);
		}
		clustering.erase (dest);
	}
}

void funcdefn::merge_nodes_in_dependence_graph (map<int, vector<int>> &dependence_graph, int source, int dest) {
	// Merge dest dependences into source
	if (dependence_graph.find (source) != dependence_graph.end ()) {
		if (dependence_graph.find (dest) != dependence_graph.end ()) {
			vector<int> dest_vec = dependence_graph[dest];
			for (vector<int>::iterator it=dest_vec.begin(); it!=dest_vec.end (); it++) {
				if (*it != source && find (dependence_graph[source].begin(), dependence_graph[source].end(), *it) == dependence_graph[source].end ())
					dependence_graph[source].push_back (*it);
			}
			dependence_graph.erase (dest);
		}
	}
	else if (dependence_graph.find (dest) != dependence_graph.end ()) {
		dependence_graph[source] = dependence_graph[dest];
		dependence_graph.erase (dest);
	}
	// Remove self-dependence in source
	if (dependence_graph.find (source) != dependence_graph.end ()) {
		vector<int>::iterator jt = find (dependence_graph[source].begin(), dependence_graph[source].end(), source);
		if (jt != dependence_graph[source].end ()) 
			dependence_graph[source].erase (jt);
	}	
	// Iterate over the entire map, and replace dest with source in rhs of each entry
	for (map<int, vector<int>>::iterator it=dependence_graph.begin(); it!=dependence_graph.end(); it++) {
		vector<int>::iterator jt = find (it->second.begin(), it->second.end(), dest);
		if (jt != it->second.end ()) {
			it->second.erase (jt);
			if (it->first != source && find (it->second.begin(), it->second.end(), source) == it->second.end ()) 
				it->second.push_back (source);	
		}
	}
	// Iterate over the map, and remove entries with empty rhs
	for (map<int, vector<int>>::iterator it=dependence_graph.begin(); it!=dependence_graph.end();) {
		if (it->second.empty ()) 
			it = dependence_graph.erase (it);
		else ++it;
	}
}

void funcdefn::print_dependence_graph (map<int, vector<int>> dep_graph) {
	if (DEBUG) cout << "\nCluster dependence graph : " << endl;
	for (map<int, vector<int>>::iterator j=dep_graph.begin(); j!=dep_graph.end(); j++) {
		int lhs = j->first;
		printf ("%d - ", lhs);
		vector<int> rhs = j->second;
		for (vector<int>::iterator k=rhs.begin(); k!=rhs.end(); k++) {
			printf ("%d ", *k);
		}
		printf ("\n");
	}
}

void funcdefn::print_dependence_graph (string name) {
	// First print the statement dependence graph
	print_cluster_dependence_graph (name);
	// Then print the sub-statement dependence graph
	if (DEBUG) cout << "\nSub-statement dependence graph for function " << name << " : " << endl;
	for (map<stmtnode*, vector<stmtnode*>>::iterator j=substmt_dependence_graph.begin(); j!=substmt_dependence_graph.end(); j++) {
		stmtnode *lhs = (*j).first;
		printf ("%d - ", lhs->get_stmt_num ());	
		vector<stmtnode*> rhs = (*j).second;
		for (vector<stmtnode*>::iterator k=rhs.begin(); k!=rhs.end(); k++) {
			printf ("%d ", (*k)->get_stmt_num());
		}
		printf ("\n");
	}
}

void funcdefn::print_schedulable_stmts (string name) {
	vector<stmtnode*> schedulable_stmts = get_schedulable_stmts ();
	cout << "\nFor function " << name << ", schedulable stmts are: ( ";
	for (vector<stmtnode*>::iterator j=schedulable_stmts.begin(); j!=schedulable_stmts.end(); j++) {
		cout << (*j)->get_stmt_num() << " ";
	}
	cout << ")" << endl;
}

void funcdefn::print_stmt_label_map (string name) {
	// First print the stmt -> labels map
	cout << "\nFor function " << name << ", stmt -> label map:\n";
	for (map<int, boost::dynamic_bitset<>>::iterator it=labels_per_stmt.begin(); it!=labels_per_stmt.end(); it++) {
		cout << "stmt " << it->first << " -> ";
		print_bitset (label_bitset_index, it->second, label_count);
	}
	// Now print the labels -> stmt map
	cout << "\nFor function " << name << ", labels -> stmt map: ";
	for (map<string, vector<int>>::iterator it=stmts_per_label.begin(); it!=stmts_per_label.end (); it++) {
		cout << endl << it->first << " -> ( ";
		for (vector<int>::iterator jt=it->second.begin(); jt!=it->second.end(); jt++)
			cout << *jt << " ";
		cout << " ) "; 
	}
	cout << endl;
}

void funcdefn::print_scatter_gather_contributions (string name) {
	map<string, vector<string>> gather_contrib = get_gather_contributions ();
	// Print gather values for output
	cout << "\nFor function " << name << ", output's gather contributions are: { ";
	for (map<string, vector<string>>::iterator j=gather_contrib.begin(); j!=gather_contrib.end(); j++) {
		cout << j->first << " <- ( ";
		for (vector<string>::iterator k=(j->second).begin(); k!=(j->second).end(); k++)
			cout << *k << " ";
		cout << ") ";
	}
	cout << "}" << endl;
	// Print scatter values for input
	map<string, vector<string>> scatter_contrib = get_scatter_contributions ();
	cout << "\nFor function " << name << ", input's scatter contributions are: { ";
	for (map<string, vector<string>>::iterator j=scatter_contrib.begin(); j!=scatter_contrib.end(); j++) {
		cout << j->first << " -> ( ";
		for (vector<string>::iterator k=(j->second).begin(); k!=(j->second).end(); k++)
			cout << *k << " ";
		cout << ") ";
	}
	cout << "}" << endl;
	// Print the label reuse
	map<string, int> label_reuse = get_label_reuse ();
	cout << "\nFor function " << name << ", label reuse: { ";
	for (map<string, int>::iterator j=label_reuse.begin(); j!=label_reuse.end(); j++) {
		cout << "[" << j->first << " , " << j->second << "] ";
	}
	cout << "}" << endl;
}

void funcdefn::print_affinities (string name) {
	// Print primary affinity
	int p_count = 0;
	map<string,map<string,int>> primary_affinity = get_primary_affinity ();
	cout << "\nFor function " << name << ", primary affinity : ";
	for (map<string,map<string,int>>::iterator j=primary_affinity.begin(); j!=primary_affinity.end(); j++) {
		string lhs = j->first;
		map<string,int> &rhs = j->second;
		cout << endl << lhs << " -> ";
		for (map<string,int>::iterator k=rhs.begin(); k!=rhs.end(); k++) {
			cout << "[" << k->first << " , " << k->second << "] ";
			p_count++;
		}
	}
	cout << "\nTotal primary affinity count: " << p_count << endl;
	// Print secondary affinity
	int s_count = 0;
	map<string,map<string,int>> secondary_affinity = get_secondary_affinity ();
	cout << "\nFor function " << name << ", secondary affinity: ";
	for (map<string,map<string,int>>::iterator j=secondary_affinity.begin(); j!=secondary_affinity.end(); j++) {
		string lhs = j->first;
		map<string,int> &rhs = j->second;
		cout << endl << lhs << " -> ";
		for (map<string,int>::iterator k=rhs.begin(); k!=rhs.end(); k++) {
			cout << "[" << k->first << " , " << k->second << "] ";
			s_count++;
		}
	}
	cout << "\nTotal secondary affinity count: " << s_count << endl;	
}

void funcdefn::create_labels (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		expr_node *lhs = (*i)->get_lhs_expr ();
		expr_node *rhs = (*i)->get_rhs_expr ();
		lhs->create_labels (label_to_node_map);
		rhs->create_labels (label_to_node_map);
	}
}

//// TODO: Erroneous right now. Remove WAR dependences by creating new entry in lassign_map
//void funcdefn::create_labels (void) {
//	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
//	map<string, int> lassign_map;
//	vector<stmtnode*> init = initial_assignments;
//	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
//		expr_node *lhs = (*i)->get_lhs_expr ();
//		expr_node *rhs = (*i)->get_rhs_expr ();
//		// If the operator is an assignment, we need to make sure that the
//		// lhs is properly SSA'ed.
//		bool is_asgn = (*i)->get_op_type () == ST_EQ;
//		if (!is_asgn) {
//			for (vector<stmtnode*>::iterator it=init.begin(); it!=init.end();) {
//				if ((*it)->get_lhs_expr () == lhs) {
//					is_asgn = true;
//					it = init.erase (it);
//					break;
//				}
//				else ++it; 
//			}
//		}
//		lhs->create_labels (lassign_map, label_to_node_map, is_asgn);
//		rhs->create_labels (lassign_map, label_to_node_map, false);
//	}
//}

void funcdefn::compute_participating_labels (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		expr_node *lhs = (*i)->get_lhs_expr ();
		expr_node *rhs = (*i)->get_rhs_expr ();
		lhs->gather_participating_labels ((*i)->get_lhs_labels (), (*i)->get_lhs_names (), coefficients);
		rhs->gather_participating_labels ((*i)->get_rhs_labels (), (*i)->get_rhs_names (), coefficients);
		(*i)->set_nonlive_count ();
	}
}

void funcdefn::distribute_rhs (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::reverse_iterator i=stmts.rbegin(); i!=stmts.rend();) {
		expr_node *lhs = (*i)->get_lhs_expr ();
		expr_node *rhs = (*i)->get_rhs_expr ();
		bool is_asgn = (*i)->get_op_type () == ST_EQ;
		// rhs must be a binary expression with mult op
		if (rhs->get_expr_type () != T_BINARY) {
			i++;
			continue;
		}
		OP_TYPE binop = dynamic_cast<binary_node*>(rhs)->get_operator ();
		if (binop != T_MULT) {
			i++;
			continue;
		}
		expr_node *bin_lhs = dynamic_cast<binary_node*>(rhs)->get_lhs ();
		expr_node *bin_rhs = dynamic_cast<binary_node*>(rhs)->get_rhs ();
		if (bin_lhs->is_data_type () && bin_rhs->is_data_type ()) {
			i++;
			continue;
		}
		vector<string> bin_lhs_labels, bin_rhs_labels;
		bin_lhs->stringify_accesses (bin_lhs_labels);
		bin_rhs->stringify_accesses (bin_rhs_labels);
		if (bin_rhs_labels.size()>1 || bin_lhs_labels.size()>1) {
			i++;
			continue;
		}
		// Identify the temp and coef from the binary node
		vector<string> temp_val;
		expr_node *temp_node=NULL, *coef_node=NULL;
		// Trivial case: coef is datatype value
		if (bin_lhs->is_data_type ()) 
			coef_node = bin_lhs;
		else if (bin_rhs->is_data_type ())
			coef_node = bin_rhs;
		for (vector<string>::iterator it=bin_lhs_labels.begin(); it!=bin_lhs_labels.end(); it++) {
			if ((*it).find ("_t_") != string::npos) {
				temp_val.push_back (*it);
				temp_node = bin_lhs;
			}
			else {
				coef_node = bin_lhs;
			}
		}
		for (vector<string>::iterator it=bin_rhs_labels.begin(); it!=bin_rhs_labels.end(); it++) {
			if ((*it).find ("_t_") != string::npos) { 
				temp_val.push_back (*it);
				temp_node = bin_rhs;
			}
			else {
				coef_node = bin_rhs;
			}
		}
		if (temp_node == NULL) {
			i++;
			continue;
		}
		if (!(temp_node->get_expr_type () == T_ID || temp_node->get_expr_type () != T_UMINUS) | temp_val.size() != 1) {
			i++;
			continue;
		}
		string replace_temp = temp_val[0];
		bool replaced = false;
		for (vector<stmtnode*>::reverse_iterator j=next(i); j!=stmts.rend(); j++) {
			expr_node *j_lhs = (*j)->get_lhs_expr ();
			// Check that the stmt type is assignment
			vector<string> t_lhs;
			j_lhs->stringify_accesses (t_lhs);
			bool lhs_found = false;
			for (vector<string>::iterator lt=t_lhs.begin(); lt!=t_lhs.end(); lt++) {
				if (replace_temp.compare (*lt) == 0) 
					lhs_found = true;	
			}
			if (lhs_found) {
				stmtnode *new_stmt = new stmtnode (distributive_stmt_op ((*i)->get_op_type(), (*j)->get_op_type()), lhs, new binary_node (binop, coef_node, (*j)->get_rhs_expr()), (*j)->get_stmt_num(), (*j)->get_orig_stmt_num());
				if (DEBUG) {
					stringstream rhs_out;
					rhs_out << "Distribution: Replacing ";
					(*j)->print_statement (rhs_out);
					rhs_out << " with ";
					new_stmt->print_statement(rhs_out);
					cout << rhs_out.str() << "\n";
				}
				*j = new_stmt;
				replaced = true;
			}
		}
		if (replaced) {
			// Replace the initialization
			if (is_asgn) {
				for (vector<stmtnode*>::iterator kt=initial_assignments.begin(); kt!=initial_assignments.end(); kt++) {
					if ((*kt)->get_lhs_expr () == temp_node) {
						(*kt)->set_lhs_expr (lhs);	
					}	
				} 
			}
			i = vector<stmtnode*>::reverse_iterator (stmts.erase (i.base() - 1));
		}
		else i++;
	}
	stmt_list->set_stmt_list (stmts);
	total_stmts = (stmt_list->get_stmt_list()).size ();
}

void funcdefn::optimize_available_expressions (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	vector<stmtnode*> init = initial_assignments;
	// The first field in tuple is the label for lhs, the second is the labels for rhs, and the third is the rhs expression
	map<stmtnode*, tuple<vector<string>, vector<string>, string>> stmt_labels;
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		// Compute the labels for this one
		expr_node *lhs = (*i)->get_lhs_expr ();
		expr_node *rhs = (*i)->get_rhs_expr ();
		bool is_asgn = (*i)->get_op_type () == ST_EQ;
		if (!is_asgn) {
			for (vector<stmtnode*>::iterator it=init.begin(); it!=init.end();) {
				if ((*it)->get_lhs_expr () == lhs) {
					is_asgn = true;
					it = init.erase (it);
					break;
				}
				else ++it; 
			}
		}
		vector<string> lhs_labels;
		vector<string> rhs_labels;
		string rhs_expr;
		lhs->stringify_accesses (lhs_labels);
		rhs->stringify_accesses (rhs_labels, rhs_expr);
		stmt_labels[*i] = make_tuple (lhs_labels, rhs_labels, rhs_expr);
	}
	// Now find available expression starting from top going down
	map<string, string> replacement_map;
	map<int, string> av_optimized;
	int av_count = 0;
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		tuple<vector<string>, vector<string>, string> src_tuple = stmt_labels[*i];
		string src_expr_label = get<2>(src_tuple);
		if (src_expr_label.compare ("") == 0) continue;
		vector<string> src_lhs_labels = get<0>(src_tuple);
		vector<string> src_rhs_labels = get<1>(src_tuple);
		if (src_rhs_labels.size () <= 1) continue;
		bool expression_killed = false;
		// Make sure that lhs does not write any rhs label 
		for (vector<string>::iterator it=src_lhs_labels.begin(); it!=src_lhs_labels.end(); it++) {
			for (vector<string>::iterator jt=src_rhs_labels.begin(); jt!=src_rhs_labels.end(); jt++) {
				if ((*it).compare (*jt) == 0) {
					expression_killed = true;
					break;
				}
			}
		}
		if (expression_killed) continue; 
		for (vector<stmtnode*>::const_iterator j=i+1; j!=stmts.end(); j++) {
			tuple<vector<string>, vector<string>, string> dest_tuple = stmt_labels[*j];
			string dest_expr_label = get<2>(dest_tuple);
			vector<string> dest_lhs_labels = get<0>(dest_tuple);
			vector<string> dest_rhs_labels = get<1>(dest_tuple);
			// First check if the destinations's rhs is same as source's rhs. If so, perform available expression analysis
			if (dest_expr_label.compare (src_expr_label) == 0) {
				if (replacement_map.find (src_expr_label) == replacement_map.end ()) {
					string temp = "_v_" + to_string (av_count++) + "_";
					replacement_map[src_expr_label] = temp;
					av_optimized[(*i)->get_stmt_num()] = temp;
				}
				string replacement = replacement_map[src_expr_label];
				expr_node *temp = new id_node (replacement);
				temp->set_type (((*i)->get_lhs_expr ())->get_type ());
				(*j)->set_rhs_expr (temp);
				// Remove the expr_label of dest
				get<2>(dest_tuple) = "";
			}
			bool expression_killed = false;
			for (vector<string>::iterator it=dest_lhs_labels.begin(); it!=dest_lhs_labels.end(); it++) {
				for (vector<string>::iterator jt=src_rhs_labels.begin(); jt!=src_rhs_labels.end(); jt++) {
					if ((*it).compare (*jt) == 0) {
						expression_killed = true;
						break;
					}
				}
			}
			if (expression_killed) {
				// Remove expression to label mapping for source stmt
				if (replacement_map.find (src_expr_label) != replacement_map.end ())
					replacement_map.erase (src_expr_label);
				break;	
			}
		}
	}
	// Insert placeholders for optimization
	if (av_optimized.size () > 0) {
		stmtlist *new_stmts = new stmtlist ();
		int id = 0, stmt_num = 0;
		for (vector<stmtnode*>::const_iterator it=stmts.begin(); it!=stmts.end(); it++,stmt_num++) {
			// Find if we need to push in an optimization condition
			int st_no = (*it)->get_stmt_num ();
			if (av_optimized.find (st_no) != av_optimized.end ()) {
				string replacement = av_optimized[st_no];
				expr_node *temp = new id_node (replacement);
				expr_node *lhs_expr = (*it)->get_lhs_expr ();
				expr_node *rhs_expr = (*it)->get_rhs_expr ();
				temp->set_type (lhs_expr->get_type ());
				(*it)->set_rhs_expr (temp);
				temp_vars.push_back (temp);
				new_stmts->push_stmt (new stmtnode (ST_EQ, temp, rhs_expr, stmt_num++, (*it)->get_orig_stmt_num ()));
			}
			(*it)->set_stmt_num (stmt_num);
			new_stmts->push_stmt (*it);
		}
		stmt_list = new_stmts;
		total_stmts = (stmt_list->get_stmt_list()).size ();
	}
}

/* A simple copy propagation pass for reducing temporaries
   1. Convert a = b; c += a to c += b;
   2. Convert a = 1; a*=b; c+=a to c+=b;
   a and b, both have to be a label at present */
void funcdefn::copy_propagation (vector<tuple<expr_node*,expr_node*,STMT_OP>> &tstmt) {
	vector<stmtnode*> init = initial_assignments;
	for (vector<tuple<expr_node*,expr_node*,STMT_OP>>::iterator i=tstmt.begin(); i!=tstmt.end();) {
		if (get<0>(*i)->get_expr_type () == T_ID && get<1>(*i)->get_expr_type () == T_ID) {
			bool init_found = false;
			vector<stmtnode*>::iterator it = initial_assignments.begin();
			for (vector<stmtnode*>::iterator j=init.begin(); j!=init.end(); j++,it++) {
				if ((*j)->get_lhs_expr () == get<0>(*i)) {
					init_found = true;
					j = init.erase (j);
					break;
				}
			}
			if (get<2>(*i) == ST_EQ || init_found) {
				bool replaced_uses = false;
				for (vector<tuple<expr_node*,expr_node*,STMT_OP>>::iterator j=i+1; j!=tstmt.end(); j++) {
					// Change the RHS
					if (get<1>(*j) == get<0>(*i)) {
						get<1>(*j) = get<1>(*i);
						replaced_uses = true;
					}
					// If the node is rewritten, then exit 
					if (get<0>(*j) == get<0>(*i)) {
						if (init_found && get<2>(*j) != ST_EQ) {
							get<0>(*j) = get<1>(*i);
							replaced_uses = true;
						}
						else break;
					}
				}
				if (replaced_uses) {
					if (init_found) initial_assignments.erase (it);
					i = tstmt.erase (i);
				}
				else ++i;
			}
			else ++i;
		}
		else ++i;
	}
}

void funcdefn::decompose_statements (DATA_TYPE gdata_type, int HEURISTIC) {
	stmtlist *new_stmts = new stmtlist (); 
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	int id = 0, stmt_num = 0, orig_stmt_num = 0;
	int s_count = 0;
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++, orig_stmt_num++) {
		vector<tuple<expr_node*, expr_node*, STMT_OP>> tstmt;
		vector<tuple<expr_node*, expr_node*, STMT_OP>> init;
		expr_node *lhs = (*i)->get_lhs_expr ();
		expr_node *rhs = (*i)->get_rhs_expr ();
		STMT_OP stmt_type = (*i)->get_op_type ();
		bool local_assigned = false;
		bool global_assigned = false;
		bool flip = false;
		if (stmt_type == ST_EQ) 
			rhs->decompose_node (tstmt, init, temp_vars, lhs, ST_EQ, id, gdata_type, local_assigned, global_assigned, flip);
		else {
			if (rhs->simple_nondecomposable_expr ()) 
				tstmt.push_back (make_tuple (lhs, rhs, stmt_type));
			else {
				string name_t = "_t_" + to_string (id++) + "_";
				expr_node *temp = new id_node (name_t);
				rhs->decompose_node (tstmt, init, temp_vars, temp, ST_EQ, id, gdata_type, local_assigned, global_assigned, flip);
				// Now infer types
				lhs->set_type (gdata_type);
				tstmt.push_back (make_tuple (lhs, temp, stmt_type));	
				temp_vars.push_back (temp);
			}
		}
		for (vector<tuple<expr_node*,expr_node*,STMT_OP>>::const_iterator j=init.begin(); j!=init.end(); j++) {
			stmtnode *node = new stmtnode (get<2>(*j), get<0>(*j), get<1>(*j));
			initial_assignments.push_back (node);
		}
		// Run a pass of copy propagation 
		if (ASSOC_MULT) copy_propagation (tstmt);
		// Run a pass of explicit assignments for tree based register allocation
		if (HEURISTIC == 0) simplify_accumulations (tstmt, s_count);
		int local_stmt_cnt = 0; 
		for (vector<tuple<expr_node*,expr_node*,STMT_OP>>::const_iterator j=tstmt.begin(); j!=tstmt.end(); j++,stmt_num++,local_stmt_cnt++) {
			stmtnode *node = new stmtnode (get<2>(*j), get<0>(*j), get<1>(*j), stmt_num, orig_stmt_num);
			new_stmts->push_stmt (node);
		}
		clusterwise_stmt_count[orig_stmt_num] = local_stmt_cnt;
		clusterwise_stmts_executed[orig_stmt_num] = 0;
	}
	stmt_list = new_stmts;
	total_stmts = (stmt_list->get_stmt_list()).size ();
	total_orig_stmts = orig_stmt_num;
	// Set the initial priority
	for (int i=0; i<total_orig_stmts; i++) {
		initial_priority[i] = total_orig_stmts-1-i;
	}
}

//// Only preserves the last writes
//void funcdefn::remove_redundant_stmts (void) {
//	map<string, stmtnode*> write_labels;
//	vector<stmtnode*> redundant_stmts;
//	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
//	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
//		// Compute the labels for this one
//		expr_node *lhs = (*i)->get_lhs_expr ();
//		if (lhs->get_expr_type () == T_SHIFTVEC && (*i)->get_op_type () == ST_EQ) {
//			vector<string> lhs_labels;
//			lassign_map[lhs->get_name()] = 0;
//			lhs->stringify_accesses (lhs_labels);
//			for (vector<string>::iterator j=lhs_labels.begin(); j!=lhs_labels.end(); j++) {
//				if (write_labels.find (*j) == write_labels.end ())
//					write_labels[*j] = *i;
//				else { 
//					redundant_stmts.push_back (write_labels[*j]);
//					write_labels[*j] = *i;
//				}
//			}
//		}
//	}
//	if (redundant_stmts.size () != 0) {
//		stmtlist *new_stmts = new stmtlist ();
//		for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
//			if (find (redundant_stmts.begin(), redundant_stmts.end (), *i) == redundant_stmts.end()) 
//				new_stmts->push_stmt (*i);
//		}
//		stmt_list = new_stmts;
//		redundant_stmts.clear ();
//	}
//	write_labels.clear ();		
//}

void funcdefn::unroll_stmts (map<string, int> unroll_decls) {
	for (map<string, int>::const_iterator u=unroll_decls.begin(); u!=unroll_decls.end(); ++u) {
		string id = u->first;
		map<string, int> scalar_count;
		stmtlist *new_stmts = new stmtlist ();
		vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
		for (int val=1; val<u->second; val++) {
			for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
				vector<tuple<expr_node*, expr_node*, STMT_OP>> tstmt;
				vector<tuple<expr_node*, expr_node*, STMT_OP>> init;
				expr_node *lhs = (*i)->get_lhs_expr ();
				expr_node *rhs = (*i)->get_rhs_expr ();
				STMT_OP stmt_type = (*i)->get_op_type ();
				expr_node *unroll_lhs = lhs->unroll_expr (id, val, coefficients, scalar_count, true);
				expr_node *unroll_rhs = rhs->unroll_expr (id, val, coefficients, scalar_count, false);
				new_stmts->push_stmt (new stmtnode (stmt_type, unroll_lhs, unroll_rhs));
			}
		}
		stmt_list->push_stmt (new_stmts->get_stmt_list ());
	}
}

// Compute the initial schedulable stmts list
void funcdefn::compute_schedulable_stmts (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		if (substmt_dependence_graph.find (*i) == substmt_dependence_graph.end ()) 
			schedulable_stmts.push_back (*i);
	}
}

stmtnode *funcdefn::split_accumulations (stmtnode *stmt) {
	if (stmt->get_op_type () == ST_EQ)
		return stmt;
	// In case of an accumulation node, find the current counter
	stmtnode *new_stmt = new stmtnode (stmt->get_op_type (), (stmt->get_lhs_expr())->deep_copy(), (stmt->get_rhs_expr())->deep_copy());
	new_stmt->set_lhs_labels (stmt->get_lhs_labels ());
	new_stmt->set_rhs_labels (stmt->get_rhs_labels ());
	vector<string> &l_labels = new_stmt->get_lhs_labels ();
	bool split = false;
	for (vector<string>::iterator it=l_labels.begin(); it!=l_labels.end(); it++) {
		if (acc_vars.find (*it) == acc_vars.end ())
			acc_vars[*it] = make_tuple (0, 0);
		bool conflicts = true;
		int free_idx = 0;
		for (int i=0; i<=get<0>(acc_vars[*it]); i++) {
			tuple<string,int> index_tuple = make_tuple (*it, i);  
			if (find (interlock_lhs.begin (), interlock_lhs.end (), index_tuple) == interlock_lhs.end ()) {
				conflicts = false;
				free_idx = i;
				break;
			}
		}
		tuple<int,int> &index_tuple = acc_vars[*it];
		if (conflicts) {
			free_idx = (get<1>(index_tuple) + 1) % ACC_SIZE;
			if (free_idx > get<0>(index_tuple) && new_stmt->get_op_type () != ST_EQ)
				initial_assignments.push_back (new stmtnode (ST_EQ, new_stmt->get_lhs_expr(), new datatype_node<int> (get_init_val (new_stmt->get_op_type()), INT)));
		}
		get<0>(index_tuple) = max (get<0>(index_tuple), free_idx);
		get<1>(index_tuple) = free_idx;
		// Change the lhs label if free_idx is greater than 0
		if (free_idx > 0) {
			string new_label = *it;
			if (new_label.substr(new_label.length()-1).compare ("_") == 0)  
				new_label += to_string (free_idx) + "_";
			else new_label += "_" + to_string (free_idx) + "_";
			if (new_stmt->get_lhs_expr()->get_expr_type () == T_SHIFTVEC) 
				dynamic_cast<shiftvec_node*>(new_stmt->get_lhs_expr())->set_label (new_label);
			if (new_stmt->get_lhs_expr()->get_expr_type () == T_ID) 
				dynamic_cast<id_node*>(new_stmt->get_lhs_expr())->set_label (new_label);
			replace (l_labels.begin (), l_labels.end (), *it, new_label);
			// Create a new_label -> register mapping
			if (register_mapping.find (new_label) == register_mapping.end ()) {
				if (register_pool.size () == 0) {
					string s = "_r_" + to_string (reg_count++) + "_";
					register_mapping[new_label] = s;
				}
				else {
					register_mapping[new_label] = register_pool.front ();
					register_pool.pop_front ();
				}
			}
			split = true;
		}
	}
	return split ? new_stmt : stmt;
}

void funcdefn::split_output_summation (stmtnode *stmt) {
	vector<string> l_labels = stmt->get_lhs_labels ();
	for (vector<string>::iterator it=l_labels.begin(); it!=l_labels.end(); it++) {
		if (acc_vars.find (*it) != acc_vars.end ()) {
			// Generate the summation
			tuple<int,int> index_value = acc_vars[*it];
			if (get<0>(index_value) > 0) {
				for (int j=1; j<=get<0>(index_value); j++) {
					string new_label = *it;
					if (new_label.substr(new_label.length()-1).compare ("_") == 0)
						new_label += to_string (j) + "_";
					else new_label += "_" + to_string (j) + "_";
					stmtnode *new_stmt = new stmtnode (ST_PLUSEQ, stmt->get_lhs_expr()->deep_copy(), new id_node (new_label));
					new_stmt->set_lhs_labels (stmt->get_lhs_labels ());
					vector<string> rhs_label;
					rhs_label.push_back (new_label);
					new_stmt->set_rhs_labels (rhs_label); 
					fireable_ilp_stmts.push_back (new_stmt);
					register_pool.push_back (register_mapping[new_label]);
				}
			}
			// Clear up the values
			acc_vars[*it] = make_tuple (0,0);
		}
	}
}

/* For all the rhs nodes, generate the summation of all copies till now */
void funcdefn::split_input_summation (stmtnode *stmt) {
	vector<string> r_labels = stmt->get_rhs_labels ();
	for (vector<string>::iterator it=r_labels.begin(); it!=r_labels.end(); it++) {
		if (acc_vars.find (*it) != acc_vars.end ()) {
			// Generate the summation
			tuple<int,int> index_value = acc_vars[*it];
			if (get<0>(index_value) > 0) {
				expr_node *lhs = new id_node (*it);
				for (int j=1; j<=get<0>(index_value); j++) {
					string new_label = *it;
					if (new_label.substr(new_label.length()-1).compare ("_") == 0)  
						new_label += to_string (j) + "_";
					else new_label += "_" + to_string (j) + "_";
					stmtnode *new_stmt = new stmtnode (ST_PLUSEQ, lhs, new id_node (new_label));
					vector<string> lhs_label, rhs_label;
					lhs_label.push_back (*it); 
					rhs_label.push_back (new_label);
					new_stmt->set_lhs_labels (lhs_label);
					new_stmt->set_rhs_labels (rhs_label);
					fireable_ilp_stmts.push_back (new_stmt);
					register_pool.push_back (register_mapping[new_label]);
				}
			}
			// Clear up the values
			acc_vars[*it] = make_tuple (0,0);
		}
	}
}

/* Compute the initial fireable stmts list. This will generally be 0, since all
   the statements will need atleast some label to be live */
void funcdefn::compute_fireable_stmts (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		if ((*i)->get_nonlive_count() == 0) {
			// Check RHS, and put in statements summing up the copies of accumulations
			if (SPLIT_ACCS) split_input_summation (*i);
			// Modify fireable_node in case of accumulation, and the LHS already in interlock_lhs
			fireable_stmts.push_back (*i);
			if (SPLIT_ACCS) fireable_ilp_stmts.push_back (split_accumulations (*i));
			(*i)->set_executed ();
			// Push the lhs labels into the interlock vector
			vector<string> l_labels = (*i)->get_lhs_labels ();
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
				int curr_idx = 0;
				if (acc_vars.find (*j) != acc_vars.end ()) {
					tuple<int,int> index_tuple = acc_vars[*j];
					curr_idx = get<1> (index_tuple);
				}
				deque<tuple<string,int>>::iterator it = find (interlock_lhs.begin(), interlock_lhs.end (), make_tuple (*j,curr_idx));
				if (it == interlock_lhs.end ()) {
					if (interlock_lhs.size () == INTERLOCK_SIZE) 
						interlock_lhs.pop_front ();
					interlock_lhs.push_back (make_tuple (*j, curr_idx));	
				}
				else {
					interlock_lhs.erase (it);
					interlock_lhs.push_back (make_tuple (*j, curr_idx));
				}
			}
			int orig_stmt_num = (*i)->get_orig_stmt_num ();
			clusterwise_stmts_executed[orig_stmt_num] += 1;
			// Check if a cluster has finished execution
			if (clusterwise_stmts_executed[orig_stmt_num] == clusterwise_stmt_count[orig_stmt_num]) 
				clusterwise_stmts_executed[orig_stmt_num] = 0;
			update_label_reuse (*i);
			if (DEBUG) printf ("\nExecuting statement %d\n", (*i)->get_stmt_num ());
			update_dependences_and_schedulable_list (*i);
		}
	}
}

// Mark the statement fireable
void funcdefn::add_fireable_stmt (stmtnode *fireable_node) {
	if (DEBUG) assert (fireable_node->get_nonlive_count() == 0);
	// Check RHS, and put in statements summing up the copies of accumulations
	if (SPLIT_ACCS) split_input_summation (fireable_node);
	// Modify fireable_node in case of accumulation, and the LHS already in interlock_lhs
	fireable_stmts.push_back (fireable_node);
	if (SPLIT_ACCS) fireable_ilp_stmts.push_back (split_accumulations (fireable_node));
	fireable_node->set_executed ();
	// Push the lhs labels into the interlock vector
	vector<string> l_labels = fireable_node->get_lhs_labels ();
	for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
		int curr_idx = 0;
		if (acc_vars.find (*j) != acc_vars.end ()) {
			tuple<int,int> index_tuple = acc_vars[*j];
			curr_idx = get<1> (index_tuple);
		}
		deque<tuple<string,int>>::iterator it = find (interlock_lhs.begin(), interlock_lhs.end (), make_tuple (*j,curr_idx));
		if (it == interlock_lhs.end ()) {
			if (interlock_lhs.size () == INTERLOCK_SIZE) 
				interlock_lhs.pop_front ();
			interlock_lhs.push_back (make_tuple (*j, curr_idx));	
		}
		else {
			interlock_lhs.erase (it);
			interlock_lhs.push_back (make_tuple (*j, curr_idx));
		}
	}
	int orig_stmt_num = fireable_node->get_orig_stmt_num ();
	clusterwise_stmts_executed[orig_stmt_num] += 1;
	if (clusterwise_stmts_executed[orig_stmt_num] == clusterwise_stmt_count[orig_stmt_num]) 
		clusterwise_stmts_executed[orig_stmt_num] = 0;
	update_label_reuse (fireable_node);
	if (DEBUG) printf ("\nExecuting statement %d\n", fireable_node->get_stmt_num ());
	update_dependences_and_schedulable_list (fireable_node);
}

// Gather all the labels in the program 
void funcdefn::compute_nonlive_labels (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::iterator i=stmts.begin(); i!=stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
			if (find (nonlive_labels.begin(), nonlive_labels.end(), *s1) == nonlive_labels.end())
				nonlive_labels.push_back (*s1);
		}
		for (vector<string>::iterator s2=r_labels.begin(); s2!=r_labels.end(); s2++) {
			if (find (nonlive_labels.begin(), nonlive_labels.end(), *s2) == nonlive_labels.end())
				nonlive_labels.push_back (*s2);
		}
	}
	if (DEBUG) printf ("Total non-live labels = %d\n", (int)nonlive_labels.size ());
}

// Update the nonlive count of all statements after label is made live
void funcdefn::update_stmt_nonlive_count (string label) {
	// Visit all the non-executed statements, and reduce their live requirement by 1 if label s participates in them
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::iterator i=stmts.begin(); i!=stmts.end(); i++) {
		if ((*i)->is_executed ()) continue;
		int frequency = 0;
		if ((*i)->is_label_present (label, frequency)) {
			(*i)->set_nonlive_count ((*i)->get_nonlive_count ()-frequency);
		}
	}
}

// Update the live count of all statements after label is made dead 
void funcdefn::update_stmt_live_count (string label) {
	// Visit all the non-executed statements, and reduce their live requirement by 1 if label s participates in them
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::iterator i=stmts.begin(); i!=stmts.end(); i++) {
		if ((*i)->is_executed ()) continue;
		int frequency = 0;
		if ((*i)->is_label_present (label, frequency)) {
			(*i)->set_nonlive_count ((*i)->get_nonlive_count ()+frequency);
		}
	}
}

// Check if there is at least one schedulable statement that requires only 1 label to be live
bool funcdefn::imminently_fireable (void) {
	bool fireable = false;
	for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		vector<string> nl_labels;
		for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
			if ((!is_live (*j)) && (find (nl_labels.begin(), nl_labels.end(), *j) == nl_labels.end())) {
				nl_labels.push_back (*j); 
			}
		}
		// Check if all the rhs labels are live or single-use
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
			if ((!is_live (*j)) && (find (nl_labels.begin(), nl_labels.end(), *j) == nl_labels.end())) {
				nl_labels.push_back (*j); 
			}
		}
		fireable |= (nl_labels.size() <= 1);
	}
	return fireable;
}

// This is a statement-level view instread of a label-level view.
// We find statements in the schedulable list that comprise labels that are live or 
// labels that have single use, and fire all the labels and execute the statement.
void funcdefn::fire_non_interlock_executable_stmts (void) {
	bool found_executable_stmt = true;
	do {
		found_executable_stmt  = false;
		for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end() && found_executable_stmt==false;) {
			vector<string> l_labels = (*i)->get_lhs_labels ();
			vector<string> r_labels = (*i)->get_rhs_labels ();
			bool interlocks = false;
			for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
				if (find (interlock_lhs.begin (), interlock_lhs.end (), make_tuple (*j, 0)) != interlock_lhs.end ())
					interlocks = true; 
			}
			if (interlocks || (*i)->is_executed ()) {++i; continue;} 
			// Check if all the lhs labels are live or single-use
			bool executable = true;
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
				if (!(is_live (*j) || single_use (*j))) {
					executable = false;
					break;
				} 
			}
			// Check if all the rhs labels are live or single-use
			for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
				if (!(is_live (*j) || single_use (*j))) {
					executable = false;
					break;
				}
			}
			if (executable) {
				found_executable_stmt = true;
				// If executable, make the statement fireable, and make all the participating labels live
				for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
					if (!is_live (*j)) make_label_live (*j, 0);
				}
				for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
					if (!is_live (*j)) make_label_live (*j, 0);
				}
				stmtnode *fired_stmt = *i;
				i = schedulable_stmts.erase (i);
				add_fireable_stmt (fired_stmt);
				break;
			}
			else ++i;
		}
	} while (found_executable_stmt);
}

// Mark the label live, and if the firing potential is not 0, then fire all the fireable statements
void funcdefn::make_label_live (string label, int f_pot) {
	// Make label live
	live_labels.push_back (label);
	nonlive_labels.erase (remove (nonlive_labels.begin(), nonlive_labels.end (), label), nonlive_labels.end ());	
	// Create a label -> register mapping
	if (DEBUG) assert (register_mapping.find (label) == register_mapping.end ());
	if (register_pool.size () == 0) {
		string s = "_r_" + to_string (reg_count++) + "_";
		register_mapping[label] = s;
	}
	else {
		register_mapping[label] = register_pool.front ();
		register_pool.pop_front ();
	}
	// Update the nonlive count of all the statement in which label participates
	update_stmt_nonlive_count (label);
	deque<stmtnode*> fire_queue;
	// Check if this label made anything fireable. If it did, remove that statement from 
	// schedulable list and put it into fireable list.
	if (f_pot > 0) {
		for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end();) {
			if (!(*i)->is_executed() && (*i)->get_nonlive_count()==0 && (*i)->is_label_present(label)) {
				if (DEBUG) printf ("Erasing %d from schedulable statements\n", (*i)->get_stmt_num ());
				fire_queue.push_back (*i);
				i = schedulable_stmts.erase (i);
			}
			else ++i;
		}
	}
	map<int,deque<stmtnode*>> interlock_distance_map;
	// Choose an instruction such that there are no interlocks
	int fire_queue_size = fire_queue.size ();
	for (deque<stmtnode*>::iterator i=fire_queue.begin(); i!=fire_queue.end(); i++) {
		vector<string> r_labels = (*i)->get_rhs_labels ();
		int interlock_distance = 0;
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
			deque<tuple<string,int>>::iterator k = find (interlock_lhs.begin (), interlock_lhs.end (), make_tuple (*j, 0));
			if (k == interlock_lhs.end ()) continue;
			interlock_distance = max (interlock_distance, (int)(distance(interlock_lhs.begin(),k)+1));
		}
		if (DEBUG) assert (interlock_distance <= INTERLOCK_SIZE);
		if (interlock_distance_map.find (interlock_distance) == interlock_distance_map.end ()) {
			deque<stmtnode*> stmt_queue;
			stmt_queue.push_back (*i);
			interlock_distance_map[interlock_distance] = stmt_queue;
		}
		else interlock_distance_map[interlock_distance].push_back (*i);
	}
	for (map<int,deque<stmtnode*>>::iterator i=interlock_distance_map.begin(); i!=interlock_distance_map.end(); i++) {
		deque<stmtnode*> &stmt_queue = i->second;
		while (!stmt_queue.empty ()) {
			stmtnode *cur_node = stmt_queue.front ();
			stmt_queue.pop_front ();
			fire_queue_size--;
			add_fireable_stmt (cur_node);
		}
	}
	if (DEBUG) assert (fire_queue_size == 0);
}

// Decide which label to spill. This is done by finding the primary and secondary affinities to non-live label.
void funcdefn::make_label_dead (void) {
	vector<tuple<string, int, int, int, int, int>> spill_metric;
	vector<tuple<string, int, int, int, int>> label_data;
	for (vector<string>::const_iterator i=live_labels.begin(); i!=live_labels.end(); i++) {
		if (DEBUG) assert (label_reuse.find (*i) != label_reuse.end () && "Could not find the label reuse (make_label_dead)");
		if (label_reuse[*i] != 0) {
			// Iterate over all the non-live values, and compute the primary/secondary affinity
			int p_aff=0, s_aff=0;
			if (primary_affinity.find (*i) != primary_affinity.end ()) {
				map<string,int> rhs = primary_affinity[*i];
				for (map<string,int>::iterator j=rhs.begin(); j!=rhs.end(); j++) {
					if (is_nonlive (j->first)) 
						p_aff += j->second;
				}
			}
			if (secondary_affinity.find (*i) != secondary_affinity.end ()) {
				map<string,int> rhs = secondary_affinity[*i];
				for (map<string,int>::iterator j=rhs.begin(); j!=rhs.end(); j++) {
					if (is_nonlive (j->first))
						s_aff += j->second;
				}
			}
			int n_val = get_nonlive_values_touched (*i);
			int depth = live_index (*i);
			label_data.push_back (make_tuple (*i, p_aff, s_aff, n_val, depth));
		}
	}
	// Now sum up to form the spill metric
	for (vector<tuple<string, int, int, int, int>>::const_iterator i=label_data.begin(); i!=label_data.end(); i++) {
		int acc_p_aff=0, acc_s_aff=0, acc_n_val=0;
		for (vector<tuple<string, int, int, int, int>>::const_iterator j=label_data.begin(); j!=label_data.end(); j++) {
			if ((get<0>(*i)).compare(get<0>(*j)) == 0) continue;
			acc_p_aff += get<1>(*j);
			acc_s_aff += get<2>(*j);
			acc_n_val += get<3>(*j);
		}
		spill_metric.push_back (make_tuple (get<0>(*i), acc_p_aff, acc_s_aff, get<3>(*i), acc_n_val, get<4>(*i)));
	}
	// Sort the spill metric
	sort (spill_metric.begin(), spill_metric.end(), sort_spill_metric_a1);
	if (DEBUG) print_spill_metric (spill_metric);
	// Make the first label non-live
	for (vector<tuple<string,int,int,int,int,int>>::const_iterator i=spill_metric.begin(); i!=spill_metric.begin()+1; i++) {
		string s = get<0>(*i);
		if (DEBUG) printf ("Decided to spill label %s (reg_count = %d)\n", s.c_str(), reg_count);
		live_labels.erase (remove (live_labels.begin(), live_labels.end (), s), live_labels.end ());
		nonlive_labels.push_back (s);
		register_pool.push_back (register_mapping[s]);
		register_mapping.erase (s);
		// Increase the count for this label in all schedulable statements
		update_stmt_live_count (s);
	}
}

void funcdefn::compute_scatter_gather_contributions (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		// Gather contributions
		for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
			if (gather_contributions.find(*s1) == gather_contributions.end() && r_labels.size() > 0) {
				vector<string> contributing_labels;
				gather_contributions[*s1] = contributing_labels;	
			}
			vector<string> &tvec = gather_contributions[*s1];
			for (vector<string>::iterator s2=r_labels.begin(); s2!=r_labels.end(); s2++) {
				if (find (tvec.begin(), tvec.end(), *s2) == tvec.end())
					tvec.push_back (*s2);
			}
		}
		// Scatter contributions
		for (vector<string>::iterator s2=r_labels.begin(); s2!=r_labels.end(); s2++) {
			if (scatter_contributions.find(*s2) == scatter_contributions.end() && l_labels.size() > 0) {
				vector<string> scatter_labels;
				scatter_contributions[*s2] = scatter_labels;
			}
			vector<string> &tvec = scatter_contributions[*s2];
			for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
				if (find (tvec.begin(), tvec.end(), *s1) == tvec.end ())
					tvec.push_back (*s1);
			}
		}
	}
}

/* Count the total number of reuses for each label */
void funcdefn::compute_label_reuse (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	int pos = 0;
	for (vector<stmtnode*>::const_iterator it=stmts.begin(); it!=stmts.end(); it++) {
		vector<string> l_labels = (*it)->get_lhs_labels ();
		vector<string> r_labels = (*it)->get_rhs_labels ();
		// Count reuse of lhs 
		for (vector<string>::iterator i=l_labels.begin(); i!=l_labels.end(); i++) {
			if (label_reuse.find (*i) == label_reuse.end ()) { 
				label_reuse[*i] = 1;
				label_bitset_index[*i] = pos++;
				//if (DEBUG) printf ("label_bitset_index[%s] = %d\n", (*i).c_str(), label_bitset_index[*i]);
			}
			else 
				label_reuse[*i] = label_reuse[*i] + 1;
		}
		// Count reuse of rhs
		for (vector<string>::iterator i=r_labels.begin(); i!=r_labels.end(); i++) {
			if (label_reuse.find (*i) == label_reuse.end ()) { 
				label_reuse[*i] = 1;
				label_bitset_index[*i] = pos++;
				//if (DEBUG) printf ("label_bitset_index[%s] = %d\n", (*i).c_str(), label_bitset_index[*i]);
			}
			else 
				label_reuse[*i] = label_reuse[*i] + 1;
		}
	}
	// Now we have the ammunition to compute the bitset and frequency for label
	label_count = (int)label_bitset_index.size ();
	label_frequency = new unsigned int[label_count] ();
	for (map<string,int>::iterator it=label_bitset_index.begin(); it!=label_bitset_index.end(); it++)
		label_frequency[it->second] = label_reuse[it->first]; 
}

void funcdefn::update_label_reuse (stmtnode *fired_node) {
	vector<string> l_labels = fired_node->get_lhs_labels ();
	vector<string> r_labels = fired_node->get_rhs_labels ();

	// Iterate over l_labels, and decrement their use
	for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
		if (DEBUG) assert ((label_reuse.find (*s1) != label_reuse.end() && label_reuse[*s1] != 0) && "Reuse already 0 (update_label_reuse)");
		label_reuse[*s1] = label_reuse[*s1] - 1;
		if (label_reuse[*s1] == 0) { 
			// Assure that all the copies are summated
			if (SPLIT_ACCS) split_output_summation (fired_node);
			register_pool.push_back (register_mapping[*s1]);
		}
	}
	// Iterate over r_labels, and decrement their use
	for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++) {
		if (DEBUG) assert ((label_reuse.find (*s1) != label_reuse.end() && label_reuse[*s1] != 0) && "Reuse already 0 (update_label_reuse)");
		label_reuse[*s1] = label_reuse[*s1] - 1;
		if (label_reuse[*s1] == 0)
			register_pool.push_back (register_mapping[*s1]);
	}
}

/* For each original statement, find the labels that participate in it, and vice versa */
void funcdefn::compute_stmt_label_map (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		int st_num = (*i)->get_orig_stmt_num ();
		if (labels_per_stmt.find (st_num) == labels_per_stmt.end ()) {
			boost::dynamic_bitset<> label_bitset (label_count);
			labels_per_stmt[st_num] = label_bitset;		
		}
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
			// Make an entry into labels -> stmt map
			if (stmts_per_label.find (*s1) == stmts_per_label.end ()) {
				vector<int> stmt_vec;
				stmts_per_label[*s1] = stmt_vec;
			}
			if (find (stmts_per_label[*s1].begin(), stmts_per_label[*s1].end(), st_num) == stmts_per_label[*s1].end()) 
				stmts_per_label[*s1].push_back (st_num);
			// Make an entry into stmt -> labels map
			labels_per_stmt[st_num][label_bitset_index[*s1]] = true;
		}
		for (vector<string>::iterator s2=r_labels.begin(); s2!=r_labels.end(); s2++) {
			// Make an entry into labels -> stmt map
			if (stmts_per_label.find (*s2) == stmts_per_label.end ()) {
				vector<int> stmt_vec;
				stmts_per_label[*s2] = stmt_vec;
			}
			if (find (stmts_per_label[*s2].begin(), stmts_per_label[*s2].end(), st_num) == stmts_per_label[*s2].end()) 
				stmts_per_label[*s2].push_back (st_num);
			// Make an entry into stmt -> labels map
			labels_per_stmt[st_num][label_bitset_index[*s2]] = true;
		}
	}
}

/* Primary affinity is among labels involved in an op. The data structure is a map, 
   from label to all the labels that it has a primary affinity to, and the affinity value. */
void funcdefn::compute_primary_affinity (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		// 1. Assign primary affinity amongst all l_labels
		if (l_labels.size () > 1) {
			for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
				map <string,int> affinity;
				if (primary_affinity.find(*s1) != primary_affinity.end()) 
					affinity = primary_affinity[*s1];
				for (vector<string>::iterator s2=l_labels.begin(); s2!=l_labels.end(); s2++) {
					if ((*s1).compare(*s2) == 0) continue;
					if (affinity.find (*s2) == affinity.end ())
						affinity[*s2] = 1;
					else affinity[*s2] = affinity[*s2] + 1;
				}
				if (affinity.size() > 0) primary_affinity[*s1] = affinity;
			}
		}
		// 2. Assign primary affinity amongst all r_labels
		if (r_labels.size () > 1) {
			for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++) {
				map <string,int> affinity;
				if (primary_affinity.find(*s1) != primary_affinity.end()) 
					affinity = primary_affinity[*s1];
				for (vector<string>::iterator s2=r_labels.begin(); s2!=r_labels.end(); s2++) {
					if ((*s1).compare(*s2) == 0) continue;
					if (affinity.find (*s2) == affinity.end ())
						affinity[*s2] = 1;
					else affinity[*s2] = affinity[*s2] + 1;
				}
				if (affinity.size() > 0) primary_affinity[*s1] = affinity;
			}
		}
		// 3. Assign primary affinity from l_labels to r_labels
		if (r_labels.size () > 0) {
			for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
				map <string,int> affinity;
				if (primary_affinity.find(*s1) != primary_affinity.end()) 
					affinity = primary_affinity[*s1];
				for (vector<string>::iterator s2=r_labels.begin(); s2!=r_labels.end(); s2++) {
					if ((*s1).compare(*s2) == 0) continue;
					if (affinity.find (*s2) == affinity.end ())
						affinity[*s2] = 1;
					else affinity[*s2] = affinity[*s2] + 1;
				}
				if (affinity.size() > 0) primary_affinity[*s1] = affinity;
			}
		}
		// 4. Assign primary affinity from r_labels to l_labels
		if (l_labels.size () > 0) {
			for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++) {
				map <string,int> affinity;
				if (primary_affinity.find(*s1) != primary_affinity.end()) 
					affinity = primary_affinity[*s1];
				for (vector<string>::iterator s2=l_labels.begin(); s2!=l_labels.end(); s2++) {
					if ((*s1).compare(*s2) == 0) continue;
					if (affinity.find (*s2) == affinity.end ())
						affinity[*s2] = 1;
					else affinity[*s2] = affinity[*s2] + 1;
				}
				if (affinity.size() > 0) primary_affinity[*s1] = affinity;
			}
		}
	}
}

// The secondary affinities are between outputs or inputs
void funcdefn::compute_secondary_affinity (void) {
	// Create affinities from gather contributions
	for (map<string, vector<string>>::iterator i=gather_contributions.begin(); i!=gather_contributions.end(); i++) {
		vector<string> &inputs = i->second;
		if (inputs.size () > 1) {
			for (vector<string>::iterator s1=inputs.begin(); s1!=inputs.end(); s1++) {
				map <string,int> affinity;
				if (secondary_affinity.find(*s1) != secondary_affinity.end()) 
					affinity = secondary_affinity[*s1];
				for (vector<string>::iterator s2=inputs.begin(); s2!=inputs.end(); s2++) {
					if ((*s1).compare (*s2) == 0) continue;
					if (affinity.find (*s2) == affinity.end ())
						affinity[*s2] = 1;
					else affinity[*s2] = affinity[*s2] + 1;
				}
				if (affinity.size() > 0) secondary_affinity[*s1] = affinity;
			}
		}
	}
	// Create affinities from scatter contributions
	for (map<string, vector<string>>::iterator i=scatter_contributions.begin(); i!=scatter_contributions.end(); i++) {
		vector<string> &inputs = i->second;
		if (inputs.size () > 1) {
			for (vector<string>::iterator s1=inputs.begin(); s1!=inputs.end(); s1++) {
				map <string,int> affinity;
				if (secondary_affinity.find(*s1) != secondary_affinity.end()) 
					affinity = secondary_affinity[*s1];
				for (vector<string>::iterator s2=inputs.begin(); s2!=inputs.end(); s2++) {
					if ((*s1).compare (*s2) == 0) continue;
					if (affinity.find (*s2) == affinity.end ())
						affinity[*s2] = 1;
					else affinity[*s2] = affinity[*s2] + 1;
				}
				if (affinity.size() > 0) secondary_affinity[*s1] = affinity;
			}
		}
	}
}

void funcdefn::compute_dependences (void) {
	// First compute the substatement dependences
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	vector<string> i_label;
	vector<string> j_label;
	bool dependence;
	// Clear the dependence graph, start from scratch
	cluster_dependence_graph.clear ();
	// Compute the statement dependences
	if (DEBUG) cout << "\nCOMPUTING CLUSTER DEPENDENCE GRAPH\n";
	compute_cluster_dependences ();
	// Now compute the dependence between substatements
	substmt_dependence_graph.clear ();
	if (DEBUG) cout << "\nCOMPUTING SUB-STATEMENT DEPENDENCE GRAPH\n";
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		if ((*i)->is_executed()) continue;
		// Go over all previous statements to figure out dependences. No need to
		// compute dependences if the statement is already executed
		for (vector<stmtnode*>::const_iterator j=stmts.begin(); j!=i; j++) {
			if ((*j)->is_executed()) continue;
			// 1. WAW dependence is only if either of them is an assignment, or they both have different orig_stmt_num
			dependence = false;
			if ((*i)->get_op_type() == ST_EQ || (*j)->get_op_type() == ST_EQ || dependence_exists_in_dependence_graph (cluster_dependence_graph, (*j)->get_orig_stmt_num(), (*i)->get_orig_stmt_num())) {
				i_label = (*i)->get_lhs_labels ();
				j_label = (*j)->get_lhs_labels ();
				for (vector<string>::iterator s1=i_label.begin(); s1!=i_label.end(); s1++) {
					for (vector<string>::iterator s2=j_label.begin(); s2!=j_label.end(); s2++) {
						if ((*s1).compare (*s2) == 0) {
							dependence = true;
							break;
						}
					}
				}
				//i_label = (*i)->get_lhs_names ();
				//j_label = (*j)->get_lhs_names ();
				//for (vector<string>::iterator s1=i_label.begin(); s1!=i_label.end(); s1++) {
				//	for (vector<string>::iterator s2=j_label.begin(); s2!=j_label.end(); s2++) {
				//		if ((*s1).compare (*s2) == 0) {
				//			dependence = true;
				//			break;
				//		}
				//	}
				//}
				if (dependence) {
					if (DEBUG) printf ("Found WAW dependence between source sub-stmt %d and dest sub-stmt %d\n", (*j)->get_stmt_num (), (*i)->get_stmt_num ());
					if (substmt_dependence_graph.find (*i) == substmt_dependence_graph.end ()) {
						vector<stmtnode*> dep_stmts;
						dep_stmts.push_back (*j);
						substmt_dependence_graph[*i] = dep_stmts;
					}
					else {
						if (find ((substmt_dependence_graph[*i]).begin(), (substmt_dependence_graph[*i]).end(), *j) == (substmt_dependence_graph[*i]).end())
							(substmt_dependence_graph[*i]).push_back (*j); 
					}
				}
			}
			// 2. WAR dependence
			i_label = (*i)->get_lhs_labels ();
			j_label = (*j)->get_rhs_labels ();
			dependence = false;
			for (vector<string>::iterator s1=i_label.begin(); s1!=i_label.end(); s1++) {
				for (vector<string>::iterator s2=j_label.begin(); s2!=j_label.end(); s2++) {
					if ((*s1).compare (*s2) == 0) {
						dependence = true;
						break;
					}
				}
			}
			i_label = (*i)->get_lhs_names ();
			j_label = (*j)->get_rhs_names ();
			//for (vector<string>::iterator s1=i_label.begin(); s1!=i_label.end(); s1++) {
			//	for (vector<string>::iterator s2=j_label.begin(); s2!=j_label.end(); s2++) {
			//		if ((*s1).compare (*s2) == 0) {
			//			dependence = true;
			//			break;
			//		}
			//	}
			//}
			if (dependence) {
				if (DEBUG) printf ("Found WAR dependence between source sub-stmt %d and dest sub-stmt %d\n", (*j)->get_stmt_num (), (*i)->get_stmt_num ());
				if (substmt_dependence_graph.find (*i) == substmt_dependence_graph.end ()) {
					vector<stmtnode*> dep_stmts;
					dep_stmts.push_back (*j);
					substmt_dependence_graph[*i] = dep_stmts;
				}
				else {
					if (find ((substmt_dependence_graph[*i]).begin(), (substmt_dependence_graph[*i]).end(), *j) == (substmt_dependence_graph[*i]).end())
						(substmt_dependence_graph[*i]).push_back (*j); 
				}
			}
			// 2. RAW dependence
			i_label = (*i)->get_rhs_labels ();
			j_label = (*j)->get_lhs_labels ();
			dependence = false;
			for (vector<string>::iterator s2=j_label.begin(); s2!=j_label.end(); s2++) {
				for (vector<string>::iterator s1=i_label.begin(); s1!=i_label.end(); s1++) {
					if ((*s1).compare (*s2) == 0) {
						dependence = true;
						break;
					}
				}
			}
			//i_label = (*i)->get_rhs_names ();
			//j_label = (*j)->get_lhs_names ();
			//for (vector<string>::iterator s2=j_label.begin(); s2!=j_label.end(); s2++) {
			//	for (vector<string>::iterator s1=i_label.begin(); s1!=i_label.end(); s1++) {
			//		if ((*s1).compare (*s2) == 0) {
			//			dependence = true;
			//			break;
			//		}
			//	}
			//}
			if (dependence) {
				if (DEBUG) printf ("Found RAW dependence between source sub-stmt %d and dest sub-stmt %d\n", (*j)->get_stmt_num (), (*i)->get_stmt_num ());
				if (substmt_dependence_graph.find (*i) == substmt_dependence_graph.end ()) {
					vector<stmtnode*> dep_stmts;
					dep_stmts.push_back (*j);
					substmt_dependence_graph[*i] = dep_stmts;
				}
				else {
					if (find ((substmt_dependence_graph[*i]).begin(), (substmt_dependence_graph[*i]).end(), *j) == (substmt_dependence_graph[*i]).end())
						(substmt_dependence_graph[*i]).push_back (*j); 
				}
			}
		}
	}
}

// Print the reuse graph
void funcdefn::print_reuse_graph (map<tuple<int,int>, boost::dynamic_bitset<>> reuse_graph) {
	cout << "\nReuse graph : " << endl;
	for (map<tuple<int,int>, boost::dynamic_bitset<>>::iterator it=reuse_graph.begin(); it!=reuse_graph.end (); it++) {
		tuple<int,int> stmt_tuple = it->first;
		cout << "(" << get<0>(stmt_tuple) << "," << get<1>(stmt_tuple) << ") : ";
		print_bitset (label_bitset_index, it->second, label_count);
	}
}

// Compute the label intersection between different statements
void funcdefn::compute_reuse_graph (map<tuple<int,int>, boost::dynamic_bitset<>> &reuse_graph, map<int, boost::dynamic_bitset<>> labels_per_stmt) {
	for (map<int, boost::dynamic_bitset<>>::iterator it=labels_per_stmt.begin(); it!=prev(labels_per_stmt.end()); it++) {
		for (map<int, boost::dynamic_bitset<>>::iterator jt=it; jt!=labels_per_stmt.end(); jt++) {
			if (it->first == jt->first) continue;
			boost::dynamic_bitset<> intersection (label_count);
			intersection = it->second & jt->second;
			if (intersection.any ())
				reuse_graph[make_tuple (it->first,jt->first)] = intersection;
		}
	}
}

// Modify the dependence graph, refill schedulable list 
void funcdefn::update_dependences_and_schedulable_list (stmtnode *fired_node) {
	deque<stmtnode*> fire_queue;
	if (substmt_dependence_graph.find (fired_node) != substmt_dependence_graph.end ()) 
		substmt_dependence_graph.erase (fired_node);
	for (map<stmtnode*, vector<stmtnode*>>::iterator i=substmt_dependence_graph.begin(); i!=substmt_dependence_graph.end();) {
		vector<stmtnode*> &rhs = i->second;
		vector<stmtnode*>::iterator it = find (rhs.begin(), rhs.end (), fired_node);
		if (it != rhs.end ()) {
			//printf ("Erasing %d from %d", (*it)->get_stmt_num (), (i->first)->get_stmt_num ()); 
			rhs.erase (it);
		}
		// If rhs size becomes 0, put the lhs into either fireable or schedulable list
		if (rhs.size () == 0) {
			stmtnode *cur_node = i->first;
			if (cur_node->get_nonlive_count () == 0) {
				if (DEBUG) printf ("Pushing %d to fireable list\n", cur_node->get_stmt_num ());
				fire_queue.push_back (cur_node);
			}
			else {
				schedulable_stmts.push_back (cur_node);
				if (DEBUG) printf ("Pushing %d to schedulable list\n", cur_node->get_stmt_num ());
			}
			if (DEBUG) printf ("\n\nRemoving %d from dep graph\n", cur_node->get_stmt_num ());
			i = substmt_dependence_graph.erase (i);
		}
		else ++i;
	}
	map<int,deque<stmtnode*>> interlock_distance_map;
	// Choose an instruction such that there are no interlocks
	int fire_queue_size = fire_queue.size ();
	for (deque<stmtnode*>::iterator i=fire_queue.begin(); i!=fire_queue.end(); i++) {
		vector<string> r_labels = (*i)->get_rhs_labels ();
		int interlock_distance = 0;
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
			deque<tuple<string,int>>::iterator k = find (interlock_lhs.begin (), interlock_lhs.end (), make_tuple (*j, 0));
			if (k == interlock_lhs.end ()) continue;
			interlock_distance = max (interlock_distance, (int)(distance(interlock_lhs.begin(),k)+1));
		}
		if (DEBUG) assert (interlock_distance <= INTERLOCK_SIZE);
		if (interlock_distance_map.find (interlock_distance) == interlock_distance_map.end ()) {
			deque<stmtnode*> stmt_queue;
			stmt_queue.push_back (*i);
			interlock_distance_map[interlock_distance] = stmt_queue;
		}
		else interlock_distance_map[interlock_distance].push_back (*i);
	}
	for (map<int,deque<stmtnode*>>::iterator i=interlock_distance_map.begin(); i!=interlock_distance_map.end(); i++) {
		deque<stmtnode*> &stmt_queue = i->second;
		while (!stmt_queue.empty ()) {
			stmtnode *cur_node = stmt_queue.front ();
			stmt_queue.pop_front ();
			fire_queue_size--;
			add_fireable_stmt (cur_node);
		}
	}
	if (DEBUG) assert (fire_queue_size == 0);
}

void funcdefn::compute_leading_stmt (map<int, vector<int>> &leading_stmt_map) {
	for (map<int, int>::iterator it=clusterwise_stmts_executed.begin(); it!=clusterwise_stmts_executed.end(); it++) {
		int executed_count = it->second;
		int stmt_num = it->first;
		if (executed_count == 0) continue;
		if (leading_stmt_map.find (executed_count) == leading_stmt_map.end ()) {
			vector<int> stmt_vec;
			leading_stmt_map[executed_count] = stmt_vec;
		}
		leading_stmt_map[executed_count].push_back (stmt_num);
	}
	// Print the leading_stmt_map
	if (DEBUG) {
		cout << "\nPrinting the leading statement map ";
		for (map<int, vector<int>>::iterator it=leading_stmt_map.begin(); it!=leading_stmt_map.end(); it++) {
			cout << endl << "Executed " << it->first << " : orig_stmt_num ( ";
			for (vector<int>::iterator jt=it->second.begin(); jt!=it->second.end(); jt++) 
				cout << *jt << " ";
			cout << " )";	
		}
		cout << endl;
	}
}

// Compute the ordered 7-tuple
void funcdefn::compute_order_metric (void) {
	// Iterate over all the schedulable statements, and create a metric for labels that are not yet live.
	order_metric.clear ();
	vector<string> ordered_labels;
	// Create necessary maps to find the leading statement
	map<int, vector<int>> leading_stmt_map;
	compute_leading_stmt (leading_stmt_map);

	for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
			if (!is_live (*j) && find (ordered_labels.begin(),ordered_labels.end(),*j) == ordered_labels.end()) {
				tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> t = fill_first_level_metric (*j, leading_stmt_map);
				order_metric.push_back (t);
				ordered_labels.push_back (*j);
			}
		}
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
			if (!is_live (*j) && find (ordered_labels.begin(),ordered_labels.end(),*j) == ordered_labels.end()) {
				tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> t = fill_first_level_metric (*j, leading_stmt_map);
				order_metric.push_back (t);
				ordered_labels.push_back (*j);
			}
		}
	}
	if (!ordered_labels.empty ()) {
		if (SECOND_LEVEL) 
			fill_second_level_metric (order_metric);
		// Now sort the metric
		if (FIRST_LEVEL && !SECOND_LEVEL)
			sort (order_metric.begin(), order_metric.end(), sort_order_metric_a0);
		else if (SECOND_LEVEL)
			sort (order_metric.begin(), order_metric.end(), sort_order_metric_b5);
	}
}

void funcdefn::register_pressure_stats (void) {
	if (DEBUG) printf ("\nREGISTER PRESSURE PER PROGRAM POINT\n");
	register_pool.clear ();
	register_mapping.clear ();
	live_labels.clear ();
	int regs_used = 0;
	map<string,int> reuse = label_reuse;
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		int max_rp = (int) register_mapping.size ();
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
			if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
				live_labels.push_back (*j);
				if (register_pool.size () == 0) {
					string s = "_r_" + to_string (regs_used++) + "_";
					register_mapping[*j] = s;
				}
				else {
					register_mapping[*j] = register_pool.front ();
					register_pool.pop_front ();
				}
			}
		}
		max_rp = max (max_rp, (int) register_mapping.size ());
		// Iterate over r_labels, and decrement their use
		for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++) {
			if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (register_pressure_stats)");
			reuse[*s1] = reuse[*s1] - 1;
			if (reuse[*s1] == 0) { 
				register_pool.push_back (register_mapping[*s1]);
				register_mapping.erase (*s1);
			}
		}
		max_rp = max (max_rp, (int) register_mapping.size ());
		for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
			if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
				live_labels.push_back (*j);
				if (register_pool.size () == 0) {
					string s = "_r_" + to_string (regs_used++) + "_";
					register_mapping[*j] = s;
				}
				else {
					register_mapping[*j] = register_pool.front ();
					register_pool.pop_front ();
				}
			}
		}
		max_rp = max (max_rp, (int) register_mapping.size ());
		// Iterate over l_labels, and decrement their use
		for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
			if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (register_pressure_stats)");
			reuse[*s1] = reuse[*s1] - 1;
			if (reuse[*s1] == 0) { 
				register_pool.push_back (register_mapping[*s1]);
				register_mapping.erase (*s1);
			}
		}
		max_rp = max (max_rp, (int) register_mapping.size ());
		printf ("Program point %4d\t-\t Max RP %4d\t Final RP%4d\n", (*i)->get_stmt_num (), max_rp, (int)register_mapping.size ());
	}
}


// Implement a linear-scan, generate spills
void funcdefn::linear_scan_spill (void) {
	if (DEBUG) printf ("\nPERFORMING LINEAR SCAN WITH SPILLS\n");
	// Create the last use (live range end) to decide spills
	map<string,int> last_use, last_write;
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
			last_use[*j] = (*i)->get_stmt_num ();
			last_write[*j] = (*i)->get_stmt_num ();
		}
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) 
			last_use[*j] = (*i)->get_stmt_num ();
	}
	for (int cnt=2; cnt<=reg_count; cnt++) {
		// Clear the data structures
		register_pool.clear ();
		register_mapping.clear ();
		live_labels.clear ();
		map<string,int> reuse = label_reuse;
		vector<string> spilled_labels;
		int spill_decisions=0, spill_loads=0, spill_stores=0, regs_used=0;

		for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
			vector<string> l_labels = (*i)->get_lhs_labels ();
			vector<string> r_labels = (*i)->get_rhs_labels ();
			// First process rhs
			for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
				// If the value is already spilled, it is a spill load
				if (find (spilled_labels.begin(), spilled_labels.end(), *j) != spilled_labels.end()) 
					spill_loads++;
				if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
					live_labels.push_back (*j);
					// If we have not reached the spill count yet
					if (register_pool.size()>0 || regs_used<cnt) {
						if (register_pool.size () == 0) {
							string s = "_r_" + to_string (regs_used++) + "_";
							register_mapping[*j] = s;
						}
						else {
							register_mapping[*j] = register_pool.front ();
							register_pool.pop_front ();
						}
					}
					// SpillAtInterval
					else {
						vector<string>::iterator spill_it = prev (live_labels.end ());
						int max_range = -1;
						// Find a spill candidate
						for (vector<string>::iterator it=live_labels.begin(); it!=live_labels.end(); it++) {
							// Do not consider already spilled labels, or labels used in this statement for spilling
							bool label_in_stmt = false; 
							for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							if (label_in_stmt || find (spilled_labels.begin(), spilled_labels.end(), *it) != spilled_labels.end()) continue;
							if (DEBUG) assert (reuse[*it] != 0 && "Reuse 0, but not removed from live labels (linear_scan_spills ())");
							if (last_use[*it] >= max_range) {
								spill_it = it;
								max_range = last_use[*it];
							}
						}
						// Could not find anything to spill? Perhaps we need more than the provided number of registers after all. 
						if (spill_it != live_labels.end () && find (spilled_labels.begin(), spilled_labels.end(), *spill_it) == spilled_labels.end()) {
							// Check if the current range added is to be spilled
							if (spill_it != prev(live_labels.end())) {
								if (last_write.find (*spill_it) != last_write.end ()) spill_stores++;
								register_mapping[*j] = register_mapping[*spill_it];
							}
							else spill_loads++;
							spill_decisions++;
							if (find (spilled_labels.begin(), spilled_labels.end(), *spill_it) == spilled_labels.end())
								spilled_labels.push_back (*spill_it);
							if (register_mapping.find (*spill_it) != register_mapping.end ()) register_mapping.erase (*spill_it);
							live_labels.erase (spill_it);
						}
					}
				}
				// Remove the spilled label from live_labels, and free the assigned register
				if (find (spilled_labels.begin(), spilled_labels.end(), *j) != spilled_labels.end()) {
					if (find (live_labels.begin(), live_labels.end(), *j) != live_labels.end()) {
						live_labels.erase (remove (live_labels.begin(), live_labels.end(), *j), live_labels.end());
						if (register_mapping.find (*j) != register_mapping.end ()) {
							if (find (register_pool.begin(), register_pool.end(), register_mapping[*j]) == register_pool.end ())
								register_pool.push_back (register_mapping[*j]);
							register_mapping.erase (*j);
						}
					}
				}
			}
			// Iterate over r_labels, and decrement their use (ExpireOldIntervals)
			for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++) {
				if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (linear_scan)");
				reuse[*s1] = reuse[*s1] - 1;
				if (reuse[*s1]==0) {
					live_labels.erase (remove (live_labels.begin(), live_labels.end(), *s1), live_labels.end());		 
					if (register_mapping.find (*s1) != register_mapping.end ()) {
						if (find (register_pool.begin(), register_pool.end(), register_mapping[*s1]) == register_pool.end ())
							register_pool.push_back (register_mapping[*s1]);
						register_mapping.erase (*s1);
					}
				}
			}
			// Then process lhs
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
				// If the value is already spilled, it is a spill load (store)
				if (find (spilled_labels.begin(), spilled_labels.end(), *j) != spilled_labels.end()) {
					spill_stores++;
					if ((*i)->get_op_type () != ST_EQ) spill_loads++;
				}
				if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
					live_labels.push_back (*j);
					// If we have not reached the spill count yet
					if (register_pool.size()>0 || regs_used<cnt) {
						if (register_pool.size () == 0) {
							string s = "_r_" + to_string (regs_used++) + "_";
							register_mapping[*j] = s;
						}
						else {
							register_mapping[*j] = register_pool.front ();
							register_pool.pop_front ();
						}
					}
					// SpillAtInterval
					else {
						vector<string>::iterator spill_it = prev (live_labels.end ());
						int max_range = -1;
						for (vector<string>::iterator it=live_labels.begin(); it!=live_labels.end(); it++) {
							// Do not consider already spilled labels, or labels used in this statement for spilling
							bool label_in_stmt = false;
							for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							if (label_in_stmt || find (spilled_labels.begin(), spilled_labels.end(), *it) != spilled_labels.end()) continue;
							if (DEBUG) assert (reuse[*it] != 0 && "Reuse 0, but not removed from live labels (linear_scan_spills ())");
							if (last_use[*it] >= max_range) {
								spill_it = it;
								max_range = last_use[*it];
							}
						}
						// Could not find anything to spill? Just assume memory op
						if (spill_it != live_labels.end () && find (spilled_labels.begin(), spilled_labels.end(), *spill_it) == spilled_labels.end()) {
							// Check if the current range added is to be spilled
							if (spill_it != prev(live_labels.end())) {
								if (last_write.find (*spill_it) != last_write.end ()) spill_stores++;
								register_mapping[*j] = register_mapping[*spill_it];
							}
							else spill_stores++;
							spill_decisions++;
							if (find (spilled_labels.begin(), spilled_labels.end(), *spill_it) == spilled_labels.end())
								spilled_labels.push_back (*spill_it);
							if (register_mapping.find (*spill_it) != register_mapping.end ()) register_mapping.erase (*spill_it);
							live_labels.erase (spill_it);
						}
					}
				}
				// Remove the spilled label from live_labels, and free the assigned register
				if (find (spilled_labels.begin(), spilled_labels.end(), *j) != spilled_labels.end()) {
					if (find (live_labels.begin(), live_labels.end(), *j) != live_labels.end()) {
						live_labels.erase (remove (live_labels.begin(), live_labels.end(), *j), live_labels.end());
						if (register_mapping.find (*j) != register_mapping.end ()) {
							if (find (register_pool.begin(), register_pool.end(), register_mapping[*j]) == register_pool.end ())
								register_pool.push_back (register_mapping[*j]);
							register_mapping.erase (*j);
						}
					}
				}
			}
			// Iterate over l_labels, and decrement their use (ExpireOldIntervals)
			for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
				if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (linear_scan)");
				reuse[*s1] = reuse[*s1] - 1;
				if (reuse[*s1]==0) {
					live_labels.erase (remove (live_labels.begin(), live_labels.end(), *s1), live_labels.end());
					if (register_mapping.find (*s1) != register_mapping.end ()) {
						if (find (register_pool.begin(), register_pool.end(), register_mapping[*s1]) == register_pool.end ())
							register_pool.push_back (register_mapping[*s1]);
						register_mapping.erase (*s1);
					}
				}
			}
		}
		if (DEBUG) printf ("For %d registers, spill decisions = %d spill loads = %d spill_stores = %d\n", cnt, spill_decisions, spill_loads, spill_stores);
	}
}


/* For each decomposed statement, find the labels that participate in it, and vice versa */
void funcdefn::compute_decomposed_stmts_per_label (map<string,vector<int>> &dstmts_per_label) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		int st_num = (*i)->get_stmt_num ();
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
			if (dstmts_per_label.find (*s1) == dstmts_per_label.end ()) {
				vector<int> stmt_vec;
				dstmts_per_label[*s1] = stmt_vec;
			}
			if (find (dstmts_per_label[*s1].begin(), dstmts_per_label[*s1].end(), st_num) == dstmts_per_label[*s1].end()) 
				dstmts_per_label[*s1].push_back (st_num);
		}
		for (vector<string>::iterator s2=r_labels.begin(); s2!=r_labels.end(); s2++) {
			if (dstmts_per_label.find (*s2) == dstmts_per_label.end ()) {
				vector<int> stmt_vec;
				dstmts_per_label[*s2] = stmt_vec;
			}
			if (find (dstmts_per_label[*s2].begin(), dstmts_per_label[*s2].end(), st_num) == dstmts_per_label[*s2].end()) 
				dstmts_per_label[*s2].push_back (st_num);
		}
	}
}

// Implement a linear-scan with containment holes, generate spills
void funcdefn::linear_scan_containment_spill (void) {
	if (DEBUG) printf ("\nPERFORMING LINEAR SCAN WITH CONTAINMENT + SPILLS\n");
	// Create the last use (live range end) to decide spills
	map<string,vector<int>> dstmts_per_label;
	compute_decomposed_stmts_per_label (dstmts_per_label);
	map<string,int> last_use, last_write;
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) { 
			last_use[*j] = (*i)->get_stmt_num ();
			last_write[*j] = (*i)->get_stmt_num ();
		}
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) 
			last_use[*j] = (*i)->get_stmt_num (); 
	}
	for (int cnt=2; cnt<=reg_count; cnt++) {
		// Clear the data structures
		register_pool.clear ();
		register_mapping.clear ();
		live_labels.clear ();
		map<string,int> reuse = label_reuse;
		vector<string> spilled_labels;
		int spill_decisions=0, spill_loads=0, spill_stores=0, regs_used=0;

		for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
			vector<string> l_labels = (*i)->get_lhs_labels ();
			vector<string> r_labels = (*i)->get_rhs_labels ();
			// First process rhs
			for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
				if (find (spilled_labels.begin(), spilled_labels.end(), *j) != spilled_labels.end()) 
					spill_loads++;
				if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
					live_labels.push_back (*j);
					// If we have not reached the spill count yet
					if (register_pool.size()>0 || regs_used<cnt) {
						if (register_pool.size () == 0) {
							string s = "_r_" + to_string (regs_used++) + "_";
							register_mapping[*j] = s;
						}
						else {
							register_mapping[*j] = register_pool.front ();
							register_pool.pop_front ();
						}
					}
					// SpillAtHoles + SpillAtInterval
					else {
						vector<string>::iterator spill_it = prev (live_labels.end ());
						int max_range = last_use[*j];	
						for (vector<string>::iterator it=live_labels.begin(); it!=prev(live_labels.end()); it++) {
							bool label_in_stmt = false;
							for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							if (label_in_stmt) continue;
							if (DEBUG) assert (dstmts_per_label.find (*it) != dstmts_per_label.end() && "label not found in dstmts_per_label (linear_scan_hole_spill ())");
							int next_range = -1;
							for (vector<int>::iterator jt=dstmts_per_label[*it].begin(); jt!=dstmts_per_label[*it].end(); jt++) {
								if (*jt > (*i)->get_stmt_num ()) {
									next_range = *jt;
									break;
								}
							}
							if (next_range > max_range) { 
								spill_it = it;
								max_range = next_range;
							}
						}
						// If did not find a containment, revert to usual linear scan
						if (spill_it == prev (live_labels.end ())) {
							spill_it = prev (live_labels.end ());
							max_range = -1;
							for (vector<string>::iterator it=live_labels.begin(); it!=live_labels.end(); it++) {
								// Do not consider already spilled labels, or labels used in this statement for spilling
								bool label_in_stmt = false;
								for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
									if ((*it).compare (*s1) == 0) label_in_stmt = true;
								for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
									if ((*it).compare (*s1) == 0) label_in_stmt = true;
								if (label_in_stmt || find (spilled_labels.begin(), spilled_labels.end(), *it) != spilled_labels.end()) continue;
								if (DEBUG) assert (reuse[*it] != 0 && "Reuse 0, but not removed from live labels (linear_scan_spills ())");
								if (last_use[*it] >= max_range) {
									spill_it = it;
									max_range = last_use[*it];
								}
							}
						}
						// Could not find anything to spill? Just assume memory op
						if (spill_it != live_labels.end () && find (spilled_labels.begin(), spilled_labels.end(), *spill_it) == spilled_labels.end()) {
							// Check if the current range added is to be spilled
							if (spill_it != prev(live_labels.end())) {
								if (last_write.find (*spill_it) != last_write.end ()) spill_stores++;
								register_mapping[*j] = register_mapping[*spill_it];
							}
							else spill_loads++;
							spill_decisions++;
							if (find (spilled_labels.begin(), spilled_labels.end(), *spill_it) == spilled_labels.end())
								spilled_labels.push_back (*spill_it);
							if (register_mapping.find (*spill_it) != register_mapping.end ()) register_mapping.erase (*spill_it);
							live_labels.erase (spill_it);
						}
					}
				}
				// Remove the spilled label from live_labels, and free the assigned register
				if (find (spilled_labels.begin(), spilled_labels.end(), *j) != spilled_labels.end()) {
					if (find (live_labels.begin(), live_labels.end(), *j) != live_labels.end()) {
						live_labels.erase (remove (live_labels.begin(), live_labels.end(), *j), live_labels.end());
						if (register_mapping.find (*j) != register_mapping.end ()) {
							if (find (register_pool.begin(), register_pool.end(), register_mapping[*j]) == register_pool.end ())
								register_pool.push_back (register_mapping[*j]);
							register_mapping.erase (*j);
						}
					}
				}
			}
			// iterate over r_labels, and decrement their use (expireoldintervals)
			for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++) {
				if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (linear_scan)");
				reuse[*s1] = reuse[*s1] - 1;
				if (reuse[*s1]==0) {
					live_labels.erase (remove (live_labels.begin(), live_labels.end(), *s1), live_labels.end());		 
					if (register_mapping.find (*s1) != register_mapping.end ()) {
						if (find (register_pool.begin(), register_pool.end(), register_mapping[*s1]) == register_pool.end ())
							register_pool.push_back (register_mapping[*s1]);
						register_mapping.erase (*s1);
					}
				}
			}
			// Then process lhs
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
				if (find (spilled_labels.begin(), spilled_labels.end(), *j) != spilled_labels.end()) {
					spill_stores++;
					if ((*i)->get_op_type () != ST_EQ) spill_loads++;
				}
				if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
					live_labels.push_back (*j);
					// If we have not reached the spill count yet
					if (register_pool.size()>0 || regs_used<cnt) {
						if (register_pool.size () == 0) {
							string s = "_r_" + to_string (regs_used++) + "_";
							register_mapping[*j] = s;
						}
						else {
							register_mapping[*j] = register_pool.front ();
							register_pool.pop_front ();
						}
					}
					// SpillAtHoles + SpillAtInterval
					else {
						vector<string>::iterator spill_it = prev (live_labels.end ());
						int max_range = last_use[*j];   
						for (vector<string>::iterator it=live_labels.begin(); it!=prev(live_labels.end()); it++) {
							bool label_in_stmt = false;
							for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							if (label_in_stmt) continue;
							if (DEBUG) assert (dstmts_per_label.find (*it) != dstmts_per_label.end() && "label not found in dstmts_per_label (linear_scan_hole_spill ())");
							int next_range = -1;
							for (vector<int>::iterator jt=dstmts_per_label[*it].begin(); jt!=dstmts_per_label[*it].end(); jt++) {
								if (*jt > (*i)->get_stmt_num ()) {
									next_range = *jt;
									break;
								}
							}
							if (next_range > max_range) {
								spill_it = it;
								max_range = next_range;
							}
						}
						// If did not find a containment, revert to usual linear scan
						if (spill_it == prev (live_labels.end ())) {
							spill_it = prev (live_labels.end ());
							max_range = -1;
							for (vector<string>::iterator it=live_labels.begin(); it!=live_labels.end(); it++) {
								// Do not consider already spilled labels, or labels used in this statement for spilling
								bool label_in_stmt = false;
								for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
									if ((*it).compare (*s1) == 0) label_in_stmt = true;
								for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
									if ((*it).compare (*s1) == 0) label_in_stmt = true;
								if (label_in_stmt || find (spilled_labels.begin(), spilled_labels.end(), *it) != spilled_labels.end()) continue;
								if (DEBUG) assert (reuse[*it] != 0 && "Reuse 0, but not removed from live labels (linear_scan_spills ())");
								if (last_use[*it] >= max_range) {
									spill_it = it;
									max_range = last_use[*it];
								}
							}
						}
						// Could not find anything to spill? Just assume memory op
						if (spill_it != live_labels.end () && find (spilled_labels.begin(), spilled_labels.end(), *spill_it) == spilled_labels.end()) {
							// Check if the current range added is to be spilled
							if (spill_it != prev(live_labels.end())) {
								if (last_write.find (*spill_it) != last_write.end ()) spill_stores++;
								register_mapping[*j] = register_mapping[*spill_it];
							}
							else spill_stores++;
							spill_decisions++;
							if (find (spilled_labels.begin(), spilled_labels.end(), *spill_it) == spilled_labels.end())
								spilled_labels.push_back (*spill_it);
							if (register_mapping.find (*spill_it) != register_mapping.end ()) register_mapping.erase (*spill_it);
							live_labels.erase (spill_it);
						}
					}
				}
				// Remove the spilled label from live_labels, and free the assigned register
				if (find (spilled_labels.begin(), spilled_labels.end(), *j) != spilled_labels.end()) {
					if (find (live_labels.begin(), live_labels.end(), *j) != live_labels.end()) {
						live_labels.erase (remove (live_labels.begin(), live_labels.end(), *j), live_labels.end());
						if (register_mapping.find (*j) != register_mapping.end ()) {
							if (find (register_pool.begin(), register_pool.end(), register_mapping[*j]) == register_pool.end ())
								register_pool.push_back (register_mapping[*j]);
							register_mapping.erase (*j);
						}
					}
				}
			}
			// iterate over l_labels, and decrement their use (expireoldintervals)
			for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
				if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (linear_scan)");
				reuse[*s1] = reuse[*s1] - 1;
				if (reuse[*s1]==0) {
					live_labels.erase (remove (live_labels.begin(), live_labels.end(), *s1), live_labels.end());
					if (register_mapping.find (*s1) != register_mapping.end ()) {
						if (find (register_pool.begin(), register_pool.end(), register_mapping[*s1]) == register_pool.end ())
							register_pool.push_back (register_mapping[*s1]);
						register_mapping.erase (*s1);
					}
				}
			}
		}
		if (DEBUG) printf ("For %d registers, spill decisions = %d spill loads = %d spill_stores = %d\n", cnt, spill_decisions, spill_loads, spill_stores);
	}
}

// Implement a linear-scan, generate splits
void funcdefn::linear_scan_split (void) {
	if (DEBUG) printf ("\nPERFORMING LINEAR SCAN WITH SPLITS\n");
	// Create the last use (live range end) to decide splits
	map<string,int> last_use, last_write;
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
			last_use[*j] = (*i)->get_stmt_num ();
			last_write[*j] = (*i)->get_stmt_num ();
		}
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) 
			last_use[*j] = (*i)->get_stmt_num (); 
	}
	for (int cnt=2; cnt<=reg_count; cnt++) {
		// Clear the data structures
		register_pool.clear ();
		register_mapping.clear ();
		live_labels.clear ();
		map<string,int> reuse = label_reuse;
		vector<string> split_labels;
		int split_decisions=0, split_loads=0, split_stores=0, regs_used=0;

		for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
			vector<string> l_labels = (*i)->get_lhs_labels ();
			vector<string> r_labels = (*i)->get_rhs_labels ();
			// First process rhs
			for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
				// Remove the label from split labels, since it might become live now
				if (find (split_labels.begin(), split_labels.end(), *j) != split_labels.end()) {
					split_labels.erase (remove (split_labels.begin(), split_labels.end(), *j), split_labels.end());
					split_loads++;
				}
				if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
					live_labels.push_back (*j);
					// If we have not reached the split count yet
					if (register_pool.size()>0 || regs_used<cnt) {
						if (register_pool.size () == 0) {
							string s = "_r_" + to_string (regs_used++) + "_";
							register_mapping[*j] = s;
						}
						else {
							register_mapping[*j] = register_pool.front ();
							register_pool.pop_front ();
						}
					}
					// SplitAtInterval
					else {
						vector<string>::iterator split_it = prev (live_labels.end ());
						int max_range = -1;
						for (vector<string>::iterator it=live_labels.begin(); it!=live_labels.end(); it++) {
							// Do not consider already splited labels, or labels used in this statement for spliting
							bool label_in_stmt = false;
							for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							if (label_in_stmt || find (split_labels.begin(), split_labels.end(), *it) != split_labels.end()) continue;
							if (DEBUG) assert (reuse[*it] != 0 && "Reuse 0, but not removed from live labels (linear_scan_splits ())");
							if (last_use[*it] >= max_range) {
								split_it = it;
								max_range = last_use[*it];
							}
						}
						// Could not find anything to split? Just assume memory op
						if (split_it != live_labels.end () && find (split_labels.begin(), split_labels.end(), *split_it) == split_labels.end()) {
							// Check if the current range added is to be splited
							if (split_it != prev(live_labels.end())) {
								if (last_write.find (*split_it) != last_write.end ()) split_stores++;
								register_mapping[*j] = register_mapping[*split_it];
							}
							else split_loads++;
							split_decisions++;
							if (find (split_labels.begin(), split_labels.end(), *split_it) == split_labels.end())
								split_labels.push_back (*split_it);
							if (register_mapping.find (*split_it) != register_mapping.end ()) register_mapping.erase (*split_it);
							live_labels.erase (split_it);
						}
					}
				}
			}
			// Iterate over r_labels, and decrement their use (ExpireOldIntervals)
			for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++) {
				if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (linear_scan)");
				reuse[*s1] = reuse[*s1] - 1;
				if (reuse[*s1]==0) {
					live_labels.erase (remove (live_labels.begin(), live_labels.end(), *s1), live_labels.end());		 
					if (register_mapping.find (*s1) != register_mapping.end ()) {
						if (find (register_pool.begin(), register_pool.end(), register_mapping[*s1]) == register_pool.end ())
							register_pool.push_back (register_mapping[*s1]);
						register_mapping.erase (*s1);
					}
				}
			}
			// Then process lhs
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
				// Remove the label from split labels, since it might become live now
				if (find (split_labels.begin(), split_labels.end(), *j) != split_labels.end()) {
					split_labels.erase (remove (split_labels.begin(), split_labels.end(), *j), split_labels.end());
					if ((*i)->get_op_type () != ST_EQ) split_loads++;
				}
				if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
					live_labels.push_back (*j);
					// If we have not reached the split count yet
					if (register_pool.size()>0 || regs_used<cnt) {
						if (register_pool.size () == 0) {
							string s = "_r_" + to_string (regs_used++) + "_";
							register_mapping[*j] = s;
						}
						else {
							register_mapping[*j] = register_pool.front ();
							register_pool.pop_front ();
						}
					}
					// SplitAtInterval
					else {
						vector<string>::iterator split_it = prev (live_labels.end ());
						int max_range = -1;
						for (vector<string>::iterator it=live_labels.begin(); it!=live_labels.end(); it++) {
							// Do not consider already splited labels, or labels used in this statement for spliting
							bool label_in_stmt = false;
							for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							if (label_in_stmt || find (split_labels.begin(), split_labels.end(), *it) != split_labels.end()) continue;
							if (DEBUG) assert (reuse[*it] != 0 && "Reuse 0, but not removed from live labels (linear_scan_splits ())");
							if (last_use[*it] >= max_range) {
								split_it = it;
								max_range = last_use[*it];
							}
						}
						// Could not find anything to split? Just assume memory op
						if (split_it != live_labels.end () && find (split_labels.begin(), split_labels.end(), *split_it) == split_labels.end()) {
							// Check if the current range added is to be splited
							if (split_it != prev(live_labels.end())) {
								if (last_write.find (*split_it) != last_write.end ()) split_stores++;
								register_mapping[*j] = register_mapping[*split_it];
							}
							else split_stores++;
							split_decisions++;
							if (find (split_labels.begin(), split_labels.end(), *split_it) == split_labels.end())
								split_labels.push_back (*split_it);
							if (register_mapping.find (*split_it) != register_mapping.end ()) register_mapping.erase (*split_it);
							live_labels.erase (split_it);
						}
					}
				}
			}
			// Iterate over l_labels, and decrement their use (ExpireOldIntervals)
			for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
				if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (linear_scan)");
				reuse[*s1] = reuse[*s1] - 1;
				if (reuse[*s1]==0) {
					live_labels.erase (remove (live_labels.begin(), live_labels.end(), *s1), live_labels.end());
					if (register_mapping.find (*s1) != register_mapping.end ()) {
						if (find (register_pool.begin(), register_pool.end(), register_mapping[*s1]) == register_pool.end ())
							register_pool.push_back (register_mapping[*s1]);
						register_mapping.erase (*s1);
					}
				}
			}
		}
		if (DEBUG) printf ("For %d registers, split decisions = %d split loads = %d split_stores = %d\n", cnt, split_decisions, split_loads, split_stores);
	}
}

// Implement a linear-scan, generate containment splits
void funcdefn::linear_scan_containment_split (void) {
	if (DEBUG) printf ("\nPERFORMING LINEAR SCAN WITH CONTAINMENT + SPLITS\n");
	// Create the last use (live range end) to decide splits
	map<string,vector<int>> dstmts_per_label;
	compute_decomposed_stmts_per_label (dstmts_per_label);
	map<string,int> last_use, last_write;
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
			last_use[*j] = (*i)->get_stmt_num ();
			last_write[*j] = (*i)->get_stmt_num ();
		}
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) 
			last_use[*j] = (*i)->get_stmt_num (); 
	}
	for (int cnt=2; cnt<=reg_count; cnt++) {
		// Clear the data structures
		register_pool.clear ();
		register_mapping.clear ();
		live_labels.clear ();
		map<string,int> reuse = label_reuse;
		vector<string> split_labels;
		int split_decisions=0, split_loads=0, split_stores=0, regs_used=0;

		for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
			vector<string> l_labels = (*i)->get_lhs_labels ();
			vector<string> r_labels = (*i)->get_rhs_labels ();
			// First process rhs
			for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
				// Remove the label from split labels, since it might become live now
				if (find (split_labels.begin(), split_labels.end(), *j) != split_labels.end()) {
					split_labels.erase (remove (split_labels.begin(), split_labels.end(), *j), split_labels.end());
					split_loads++;
				}
				if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
					live_labels.push_back (*j);
					// If we have not reached the split count yet
					if (register_pool.size()>0 || regs_used<cnt) {
						if (register_pool.size () == 0) {
							string s = "_r_" + to_string (regs_used++) + "_";
							register_mapping[*j] = s;
						}
						else {
							register_mapping[*j] = register_pool.front ();
							register_pool.pop_front ();
						}
					}
					// SplitAtInterval
					else {
						vector<string>::iterator split_it = prev (live_labels.end ());
						int max_range = last_use[*j];
						for (vector<string>::iterator it=live_labels.begin(); it!=prev(live_labels.end()); it++) {
							bool label_in_stmt = false;
							for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							if (label_in_stmt) continue;
							if (DEBUG) assert (dstmts_per_label.find (*it) != dstmts_per_label.end() && "label not found in dstmts_per_label (linear_scan_hole_spill ())");
							int next_range = -1;
							for (vector<int>::iterator jt=dstmts_per_label[*it].begin(); jt!=dstmts_per_label[*it].end(); jt++) {
								if (*jt > (*i)->get_stmt_num ()) {
									next_range = *jt;
									break;
								}
							}
							if (next_range > max_range) {
								split_it = it;
								max_range = next_range;
							}
						}
						// If did not find a containment, revert to usual linear scan
						if (split_it == prev (live_labels.end ())) {
							split_it = prev (live_labels.end ());
							max_range = -1;
							for (vector<string>::iterator it=live_labels.begin(); it!=live_labels.end(); it++) {
								// Do not consider already splited labels, or labels used in this statement for spliting
								bool label_in_stmt = false;
								for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
									if ((*it).compare (*s1) == 0) label_in_stmt = true;
								for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
									if ((*it).compare (*s1) == 0) label_in_stmt = true;
								if (label_in_stmt || find (split_labels.begin(), split_labels.end(), *it) != split_labels.end()) continue;
								if (DEBUG) assert (reuse[*it] != 0 && "Reuse 0, but not removed from live labels (linear_scan_splits ())");
								if (last_use[*it] >= max_range) {
									split_it = it;
									max_range = last_use[*it];
								}
							}
						}
						// Could not find anything to split? Just assume memory op
						if (split_it != live_labels.end () && find (split_labels.begin(), split_labels.end(), *split_it) == split_labels.end()) {
							// Check if the current range added is to be splited
							if (split_it != prev(live_labels.end())) {
								if (last_write.find (*split_it) != last_write.end ()) split_stores++;
								register_mapping[*j] = register_mapping[*split_it];
							}
							else split_loads++;
							split_decisions++;
							if (find (split_labels.begin(), split_labels.end(), *split_it) == split_labels.end())
								split_labels.push_back (*split_it);
							if (register_mapping.find (*split_it) != register_mapping.end ()) register_mapping.erase (*split_it);
							live_labels.erase (split_it);
						}
					}
				}
			}
			// Iterate over r_labels, and decrement their use (ExpireOldIntervals)
			for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++) {
				if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (linear_scan)");
				reuse[*s1] = reuse[*s1] - 1;
				if (reuse[*s1]==0) {
					live_labels.erase (remove (live_labels.begin(), live_labels.end(), *s1), live_labels.end());		 
					if (register_mapping.find (*s1) != register_mapping.end ()) {
						if (find (register_pool.begin(), register_pool.end(), register_mapping[*s1]) == register_pool.end ())
							register_pool.push_back (register_mapping[*s1]);
						register_mapping.erase (*s1);
					}
				}
			}
			// Then process lhs
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
				// Remove the label from split labels, since it might become live now
				if (find (split_labels.begin(), split_labels.end(), *j) != split_labels.end()) {
					split_labels.erase (remove (split_labels.begin(), split_labels.end(), *j), split_labels.end());
					if ((*i)->get_op_type () != ST_EQ) split_loads++;
				}
				if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
					live_labels.push_back (*j);
					// If we have not reached the split count yet
					if (register_pool.size()>0 || regs_used<cnt) {
						if (register_pool.size () == 0) {
							string s = "_r_" + to_string (regs_used++) + "_";
							register_mapping[*j] = s;
						}
						else {
							register_mapping[*j] = register_pool.front ();
							register_pool.pop_front ();
						}
					}
					// SplitAtInterval
					else {
						vector<string>::iterator split_it = prev (live_labels.end ());
						int max_range = last_use[*j];
						for (vector<string>::iterator it=live_labels.begin(); it!=prev(live_labels.end()); it++) {
							bool label_in_stmt = false;
							for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
								if ((*it).compare (*s1) == 0) label_in_stmt = true;
							if (label_in_stmt) continue;
							if (DEBUG) assert (dstmts_per_label.find (*it) != dstmts_per_label.end() && "label not found in dstmts_per_label (linear_scan_hole_spill ())");
							int next_range = -1;
							for (vector<int>::iterator jt=dstmts_per_label[*it].begin(); jt!=dstmts_per_label[*it].end(); jt++) {
								if (*jt > (*i)->get_stmt_num ()) {
									next_range = *jt;
									break;
								}
							}
							if (next_range > max_range) {
								split_it = it;
								max_range = next_range;
							}
						}
						// If did not find a containment, revert to usual linear scan
						if (split_it == prev (live_labels.end ())) {
							split_it = prev (live_labels.end ());
							max_range = -1;
							for (vector<string>::iterator it=live_labels.begin(); it!=live_labels.end(); it++) {
								// Do not consider already splited labels, or labels used in this statement for spliting
								bool label_in_stmt = false;
								for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++)
									if ((*it).compare (*s1) == 0) label_in_stmt = true;
								for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++)
									if ((*it).compare (*s1) == 0) label_in_stmt = true;
								if (label_in_stmt || find (split_labels.begin(), split_labels.end(), *it) != split_labels.end()) continue;
								if (DEBUG) assert (reuse[*it] != 0 && "Reuse 0, but not removed from live labels (linear_scan_splits ())");
								if (last_use[*it] >= max_range) {
									split_it = it;
									max_range = last_use[*it];
								}
							}
						}
						// Could not find anything to split? Just assume memory op
						if (split_it != live_labels.end () && find (split_labels.begin(), split_labels.end(), *split_it) == split_labels.end()) {
							// Check if the current range added is to be splited
							if (split_it != prev(live_labels.end())) {
								if (last_write.find (*split_it) != last_write.end ()) split_stores++;
								register_mapping[*j] = register_mapping[*split_it];
							}
							else split_stores++;
							split_decisions++;
							if (find (split_labels.begin(), split_labels.end(), *split_it) == split_labels.end())
								split_labels.push_back (*split_it);
							if (register_mapping.find (*split_it) != register_mapping.end ()) register_mapping.erase (*split_it);
							live_labels.erase (split_it);
						}
					}
				}
			}
			// Iterate over l_labels, and decrement their use (ExpireOldIntervals)
			for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
				if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (linear_scan)");
				reuse[*s1] = reuse[*s1] - 1;
				if (reuse[*s1]==0) {
					live_labels.erase (remove (live_labels.begin(), live_labels.end(), *s1), live_labels.end());
					if (register_mapping.find (*s1) != register_mapping.end ()) {
						if (find (register_pool.begin(), register_pool.end(), register_mapping[*s1]) == register_pool.end ())
							register_pool.push_back (register_mapping[*s1]);
						register_mapping.erase (*s1);
					}
				}
			}
		}
		if (DEBUG) printf ("For %d registers, split decisions = %d split loads = %d split_stores = %d\n", cnt, split_decisions, split_loads, split_stores);
	}
}

void funcdefn::analyze_statements (stringstream &output) {
	// Clear the data structures
	reg_count = 0;
	register_pool.clear ();
	register_mapping.clear ();
	live_labels.clear ();
	fireable_stmts.clear ();	
	map<string,int> reuse = label_reuse;
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
			if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
				live_labels.push_back (*j);
				if (register_pool.size () == 0) {
					string s = "_r_" + to_string (reg_count++) + "_";
					register_mapping[*j] = s;
				}
				else {
					register_mapping[*j] = register_pool.front ();
					register_pool.pop_front ();
				}
			}
		}
		// Iterate over r_labels, and decrement their use
		for (vector<string>::iterator s1=r_labels.begin(); s1!=r_labels.end(); s1++) {
			if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (analyze_statements)");
			reuse[*s1] = reuse[*s1] - 1;
			if (reuse[*s1] == 0) 
				register_pool.push_back (register_mapping[*s1]);
		}
		for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
			if (find (live_labels.begin(), live_labels.end(), *j) == live_labels.end()) {
				live_labels.push_back (*j);
				if (register_pool.size () == 0) {
					string s = "_r_" + to_string (reg_count++) + "_";
					register_mapping[*j] = s;
				}
				else {
					register_mapping[*j] = register_pool.front ();
					register_pool.pop_front ();
				}
			}
		}
		// Iterate over l_labels, and decrement their use
		for (vector<string>::iterator s1=l_labels.begin(); s1!=l_labels.end(); s1++) {
			if (DEBUG) assert ((reuse.find (*s1) != reuse.end() && reuse[*s1] != 0) && "Reuse already 0 (analyze_statements)");
			reuse[*s1] = reuse[*s1] - 1;
			if (reuse[*s1] == 0) 
				register_pool.push_back (register_mapping[*s1]);
		}
		// Push the statement in fireable list
		fireable_stmts.push_back (*i);
	}
	if (DEBUG) printf ("ORIGINAL VERSION TAKES %d REGISTERS\n", reg_count);
	max_reg = min (max_reg, reg_count);
	if (DEBUG) printf ("DETERMINED MAX_REG = %d\n", max_reg);

	deque<stmtnode*>orig_fireable_stmts;
	for (deque<stmtnode*>::iterator it=fireable_stmts.begin(); it!=fireable_stmts.end(); it++) {
		stmtnode *fired_node = new stmtnode ((*it)->get_op_type (), (*it)->get_lhs_expr (), (*it)->get_rhs_expr ());
		fired_node->set_lhs_labels ((*it)->get_lhs_labels ());
		fired_node->set_rhs_labels ((*it)->get_rhs_labels ());
		orig_fireable_stmts.push_back (fired_node);
	}
	// First figure out the last writes
	map<string,stmtnode*> last_write;
	for (deque<stmtnode*>::iterator it=orig_fireable_stmts.begin(); it!=orig_fireable_stmts.end(); it++) {
		if ((*it)->get_lhs_expr()->get_expr_type() == T_SHIFTVEC) {
			vector<string> l_labels = (*it)->get_lhs_labels ();
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++)
				last_write[*j] = *it;
		}
	}
	print_reordered_stmts (output, last_write, orig_fireable_stmts, INIT_EMBED);
}

/* Compute transitive dependence graph */
void funcdefn::compute_transitive_dependences (map<int,vector<int>> &transitive_dependence_graph, map<int,vector<int>> dep_graph) {
	for (map<int,vector<int>>::reverse_iterator it=dep_graph.rbegin(); it!=dep_graph.rend(); it++) {
		int dest = it->first;
		for (vector<int>::iterator jt=(it->second).begin(); jt!=(it->second).end(); jt++) {
			// For all j->k->l, add an edge j->l
			if (dep_graph.find (*jt) != dep_graph.end ()) {
				for (vector<int>::iterator kt=dep_graph[*jt].begin(); kt!=dep_graph[*jt].end(); kt++) {
					if (transitive_dependence_graph.find (dest) == transitive_dependence_graph.end ()) {
						vector<int> dep_vec;
						transitive_dependence_graph[dest] = dep_vec;	
					}
					if (find (transitive_dependence_graph[dest].begin(), transitive_dependence_graph[dest].end(), *kt) == transitive_dependence_graph[dest].end()) 
						transitive_dependence_graph[dest].push_back (*kt);
				}
			}
		}
	}
}

// Return true if transitive dependence exists between the two nodes
bool funcdefn::transitive_dependence_exists (map<int, vector<int>> transitive_dependence_graph, int m1, int m2) {
	// Check m1 -> m2
	if (transitive_dependence_graph.find (m1) != transitive_dependence_graph.end ()) {
		vector<int> source_vec = transitive_dependence_graph[m1];
		if (find (source_vec.begin(), source_vec.end(), m2) != source_vec.end())
			return true;
	}
	// check m2 -> m1
	if (transitive_dependence_graph.find (m2) != transitive_dependence_graph.end ()) {
		vector<int> source_vec = transitive_dependence_graph[m2];
		if (find (source_vec.begin(), source_vec.end(), m1) != source_vec.end())
			return true;
	}
	return false;
}

void funcdefn::create_topological_sort (void) {
	int num_orig_stmts = total_orig_stmts;
	map<int, vector<int>> clustering;
	vector<int> topological_order;
	map<int, vector<int>> dependence_graph = cluster_dependence_graph;
	map<int, boost::dynamic_bitset<>> label_use = labels_per_stmt;
	initial_priority.clear ();
	if (num_orig_stmts <= 1) {
		vector<int> cluster_vec;
		clustering[0] = cluster_vec;
	}
	bool fuse_further = true;
	while (fuse_further) {
		fuse_further = true;
		// Compute label reuse map
		map<tuple<int,int>, boost::dynamic_bitset<>> reuse_graph;
		compute_reuse_graph (reuse_graph, label_use);
		if (DEBUG) print_reuse_graph (reuse_graph);
		// Compute transitive dependence
		map<int, vector<int>> transitive_dependence_graph;
		compute_transitive_dependences (transitive_dependence_graph, dependence_graph);
		if (DEBUG) print_dependence_graph (dependence_graph);
		if (DEBUG) print_transitive_dependence_graph (transitive_dependence_graph);
		// Find the one with maximum sharing
		tuple<int,int> max_iter;
		int max_reuse = -1;
		for (map<tuple<int,int>, boost::dynamic_bitset<>>::iterator it=reuse_graph.begin(); it!=reuse_graph.end(); it++) {
			tuple<int,int> jt = it->first;
			// There should be no transitive dependence between the tuple elements
			boost::dynamic_bitset<> reuse_bitset = it->second;
			if (transitive_dependence_exists (transitive_dependence_graph, get<0>(jt), get<1>(jt))) continue;
			if ((int)reuse_bitset.count () > max_reuse) {
				max_iter = (dependence_exists_in_dependence_graph (dependence_graph, get<1>(jt), get<0>(jt))) ? make_tuple(get<1>(jt), get<0>(jt)) : jt; 
				max_reuse = (int)reuse_bitset.count ();
			}
		}
		if (max_reuse >= 0) {
			if (DEBUG) assert (max_reuse >= 0 && "Something went wrong with topological sorting"); 
			// Choose the source and destination
			int source = get<0>(max_iter);
			int dest = get<1>(max_iter);
			// Add the pair to clustering
			merge_nodes_in_topological_clustering (clustering, source, dest);
			// Merge the dependences of the two nodes
			merge_nodes_in_dependence_graph (dependence_graph, source, dest);
			// Modify the labels for the fused nodes
			label_use[source] = label_use[source] | label_use[dest];
			label_use.erase (dest);
		}
		else fuse_further = false;
	}
	if (DEBUG) assert ((int)clustering.size() == 1 && "Incorrect clustering");

	// Compute the final topological sort
	for (map<int,vector<int>>::iterator it=clustering.begin(); it!=clustering.end(); it++) {
		topological_order.push_back (it->first);
		topological_order.insert (topological_order.end(), it->second.begin(), it->second.end());
	}
	if (DEBUG) assert ((int)topological_order.size() == total_orig_stmts && "Incorrect topological sort");
	// Print the final topological order
	if (DEBUG) {
		cout << "Final topological ordering : ( ";
		for (vector<int>::iterator it=topological_order.begin(); it!=topological_order.end(); it++) {
			cout << *it << " ";
		}
		cout << " )\n" << endl;
	}
	// Verify the topological sort
	bool dependences_verified = true;
	for (vector<int>::iterator it=topological_order.begin(); it!=prev(topological_order.end()); it++) { 
		int pred = *it;
		for (vector<int>::iterator jt=next(it); jt!=topological_order.end(); jt++) {
			int succ = *jt;
			assert (pred != succ && "Multiple entries in topological sort");
			bool ret = verify_dependence (cluster_dependence_graph, pred, succ);	
			if (ret == false) cout << "FAILED for " << pred << " and " << succ << endl;
			dependences_verified &= ret;
		}
	}
	if (DEBUG) assert (dependences_verified && "Topological sort does not preserve dependences");
	// Set the initial priority for statements
	num_orig_stmts = total_orig_stmts-1;
	for (vector<int>::iterator it=topological_order.begin(); it!=topological_order.end(); it++,num_orig_stmts--) {
		assert (initial_priority.find (*it) == initial_priority.end () && "Multiple entries in topological sort");
		initial_priority[*it] = num_orig_stmts; 
	}
}

// Select a single most profitable statement from the schedulable list, and fire it.
// Make all its labels live
void funcdefn::fire_single_schedulable_statement (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>> order_metric) {
	// Create necessary maps to find the leading statement
	vector<int> leading_stmts;
	map<int, vector<int>> leading_stmt_map;
	compute_leading_stmt (leading_stmt_map);
	if (!leading_stmt_map.empty()) 
		leading_stmts = leading_stmt_map.rbegin()->second;
	int lead_highest_priority = INT_MAX, highest_priority = INT_MAX, highest_st_num = INT_MAX;
	stmtnode *lead_fire_candidate, *fire_candidate;
	bool lead_found = false;
	for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end(); i++) {
		int st_num = (*i)->get_orig_stmt_num ();
		int stmt_priority = 0;
		// Select the statement with labels with highest priority
		vector<string> l_labels = (*i)->get_lhs_labels ();
		vector<string> r_labels = (*i)->get_rhs_labels ();
		for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
			if (!is_live (*j)) {
				// Find the position of the label in order_metric
				int pos = 0;
				for (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>>::const_iterator k=order_metric.begin(); k!=order_metric.end(); k++,pos++) {
					if (get<0>(*k).compare (*j) == 0) { 
						stmt_priority += pos;
						break;
					}
				}
			}
		}
		for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
			if (!is_live (*j)) {
				// Find the position of the label in order_metric
				int pos = 0;
				for (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>>::const_iterator k=order_metric.begin(); k!=order_metric.end(); k++,pos++) {
					if (get<0>(*k).compare (*j) == 0) {
						stmt_priority += pos;
						break;
					}
				}
			}
		}
		if ((!leading_stmts.empty()) && find (leading_stmts.begin(), leading_stmts.end(), st_num) != leading_stmts.end ()) {
			if (stmt_priority < lead_highest_priority) {
				lead_highest_priority = stmt_priority;
				lead_fire_candidate = *i;
				lead_found = true;
			}
		}
		else {
			if ((st_num < highest_st_num) || ((st_num == highest_st_num) && (stmt_priority < lead_highest_priority))) {
				highest_priority = stmt_priority;
				highest_st_num = st_num;
				fire_candidate = *i;
			}
		}
	}
	// Fire the statement
	stmtnode *fired_stmt = (lead_found == true) ? lead_fire_candidate : fire_candidate;
	vector<string> l_labels = fired_stmt->get_lhs_labels ();
	vector<string> r_labels = fired_stmt->get_rhs_labels ();
	for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
		if (!is_live (*j)) 
			make_label_live (*j,1);
	}
	for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
		if (!is_live (*j))
			make_label_live (*j,1); 
	}
}

void funcdefn::reorder_statements (stringstream &output) {
	// Clear the data structures
	reg_count = 0;
	register_pool.clear ();
	register_mapping.clear ();
	live_labels.clear ();
	fireable_stmts.clear ();
	bool labels_remaining = true;
	bool spilled = false;
	// First create a topological sort based on greedy clustering
	if (TOPOLOGICAL_SORT) create_topological_sort ();	
	while (labels_remaining) {
		if (DEBUG) printf ("max_reg = %d, reg_count = %d, register_pool size = %d\n\n", max_reg, reg_count, (int)register_pool.size());
		//if (reg_count >= max_reg && register_pool.size () == 0) {
		//	make_label_dead ();
		//	spilled = true;
		//}
		// Look at all the schedulable statements, and see if any of them has all the labels a) either live, or b) single-use,
		// and execute it if it frees some registers.
		//fire_non_interlock_executable_stmts ();

		compute_order_metric ();
		if (DEBUG) print_order_metric ();
		if (DEBUG) assert (order_metric.size() > 0 && "No labels in schedulable statements! (reorder_statements)");
		// If none of the schedulable statements has only 1 non-live label, then 
		// pick one and fire it. This would avoid the reiteration of labels.
		if (OPERATION_VIEW) {
			if (get<4>(order_metric.front()) == 0) {
				if (DEBUG) assert (!imminently_fireable () && "Forcefully firing single statement despite a statement being imminently fireable");
				fire_single_schedulable_statement (order_metric);		
			}
			else {
				// Pick DPAR_LOADS labels, make them live.
				for (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>>::const_iterator i=order_metric.begin(); i!=order_metric.begin()+PAR_LOADS; i++) {
					string s = get<0>(*i);
					int r_pot = get<1>(*i);
					int f_pot = get<4>(*i); 
					if (DEBUG) cout << "\nMaking label " << s.c_str() << " live, release_potential = " << r_pot << ", firing potential = " << f_pot << endl;
					// Make label live, and add all the statement that it fires to fireable list
					make_label_live (s, f_pot);
				}
			}
		}
		else {
			// Pick DPAR_LOADS labels, make them live.
			for (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>>::const_iterator i=order_metric.begin(); i!=order_metric.begin()+PAR_LOADS; i++) {
				string s = get<0>(*i);
				int r_pot = get<1>(*i);
				int f_pot = get<4>(*i); 
				if (DEBUG) cout << "\nMaking label " << s.c_str() << " live, release_potential = " << r_pot << ", firing potential = " << f_pot << endl;
				// Make label live, and add all the statement that it fires to fireable list
				make_label_live (s, f_pot);
			}
		}
		// Check if need to continue or not
		if (schedulable_stmts.size () == 0 && fireable_stmts.size () == total_stmts) {
			// Check that all the statements are executed
			vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
			for (vector<stmtnode*>::iterator i=stmts.begin(); i!=stmts.end(); i++) 
				if (DEBUG) assert ((*i)->is_executed () && "Something wrong with reordering: statement not executed, but not schedulable");
			// Check that all the reuses are 0
			for (map<string,int>::iterator i=label_reuse.begin(); i!=label_reuse.end(); i++)
				if (DEBUG) assert (i->second == 0 && "Something wrong with reordering: label reuse not 0"); 
			labels_remaining = false;	
		}
	}
	if (spilled) reg_count = max (reg_count, max_reg); 
	if (DEBUG) printf ("REORDERED VERSION TAKES %d REGISTERS\n", reg_count);

	// First figure out the arrays written 
	map<string,stmtnode*> last_write;
	for (deque<stmtnode*>::iterator it=fireable_stmts.begin(); it!=fireable_stmts.end(); it++) {
		if ((*it)->get_lhs_expr()->get_expr_type() == T_SHIFTVEC) {
			vector<string> l_labels = (*it)->get_lhs_labels ();
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) 
				last_write[*j] = NULL;
		}
	}
	if (SPLIT_ACCS) print_reordered_stmts (output, last_write, fireable_ilp_stmts, INIT_EMBED);
	else print_reordered_stmts (output, last_write, fireable_stmts, INIT_EMBED);
	if (DEBUG) assert (nonlive_labels.size () == 0 && "Some labels are not live yet (reorder_statements)");
}

// Print the reordered statements
void funcdefn::print_reordered_stmts (stringstream &output, map<string,stmtnode*> &last_write, deque<stmtnode*> &fireable_stmts, PRINT_OPTION print) {
	vector<stmtnode*> init = initial_assignments;
	// Concretize the last writes
	int stmt_num = 0;
	for (deque<stmtnode*>::iterator it=fireable_stmts.begin(); it!=fireable_stmts.end(); it++,stmt_num++) {
		vector<string> l_labels = (*it)->get_lhs_labels ();
		if ((*it)->get_lhs_expr()->get_expr_type() == T_SHIFTVEC) {	
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) 
				if (last_write.find (*j) != last_write.end ()) 
					last_write[*j] = *it;
		}
	}
	// Assignments are separate statements
	if (print == INIT_ASSIGN && !PRINT_INTRINSICS) {
		vector<string> initialized_labels;
		// Print out all the statements from the fireable list in order
		stringstream header_output, tmp_output;
		while (fireable_stmts.size () != 0) {
			stmtnode *fired_node = fireable_stmts.front ();
			expr_node *lhs = fired_node->get_lhs_expr ();
			// Iterate over initial assignments, and add the initialization if lhs match
			for (vector<stmtnode*>::iterator i=init.begin(); i!=init.end();) {
				if ((*i)->get_lhs_expr () == lhs) {
					header_output << (*i)->print_statement (tmp_output, initialized_labels, iters);
					i = init.erase (i);
					break;
				}
				else ++i; 
			}
			header_output << fired_node->print_statement (tmp_output, initialized_labels, iters);
			// Print a store if this is the last write
			vector<string> l_labels = fired_node->get_lhs_labels ();
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
				if (last_write.find (*j) != last_write.end () && last_write[*j] == fired_node) 
					tmp_output << dynamic_cast<shiftvec_node*>(fired_node->get_lhs_expr())->print_array () << " = " << *j << ";\n";
			}
		}
		output << header_output.rdbuf () << endl << tmp_output.rdbuf ();
	}
	else if (print == INIT_ASSIGN && PRINT_INTRINSICS) {
		vector<string> initialized_labels;
		// Print out all the statements from the fireable list in order
		stringstream header_output, tmp_output;
		while (fireable_stmts.size () != 0) {
			stmtnode *fired_node = fireable_stmts.front ();
			fireable_stmts.pop_front ();
			expr_node *lhs = fired_node->get_lhs_expr ();
			// Iterate over initial assignments, and add the initialization if lhs match
			for (vector<stmtnode*>::iterator i=init.begin(); i!=init.end();) {
				if ((*i)->get_lhs_expr () == lhs) {
					header_output << (*i)->print_statement (tmp_output, initialized_labels, iters);
					i = init.erase (i);
					break;
				}
				else ++i; 
			}
			header_output << fired_node->print_statement (tmp_output, initialized_labels, iters);
			// Print a store if this is the last write
			vector<string> l_labels = fired_node->get_lhs_labels ();
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
				if (last_write.find (*j) != last_write.end () && last_write[*j] == fired_node) {
					DATA_TYPE lhs_type = fired_node->get_lhs_expr()->get_type();
					if (lhs_type == DOUBLE) tmp_output << "_mm256_storeu_pd (&";
					else if (lhs_type == FLOAT) tmp_output << "_mm256_storeu_ps (&";
					else tmp_output << "_mm256_storeu_epi64 (&";
					tmp_output << dynamic_cast<shiftvec_node*>(fired_node->get_lhs_expr())->print_array () << ", " << *j << ");\n";
				}
			}
		}
		output << header_output.rdbuf () << endl << tmp_output.rdbuf ();
	}
	// Assignments are embedded into following statements (convert the first A=0;A+=B to A=B)
	else if (print == INIT_EMBED && !PRINT_INTRINSICS) {
		vector<string> initialized_labels;
		stringstream header_output, tmp_output;
		// Print out all the statements from the fireable list in order
		while (fireable_stmts.size () != 0) {
			stmtnode *fired_node = fireable_stmts.front ();
			fireable_stmts.pop_front ();
			expr_node *lhs = fired_node->get_lhs_expr ();
			// Iterate over initial assignments, and add the initialization if lhs match
			bool init_found = false;
			for (vector<stmtnode*>::iterator i=init.begin(); i!=init.end();) {
				if ((*i)->get_lhs_expr () == lhs) {
					init_found = true;
					i = init.erase (i);
					break;
				}
				else ++i; 
			}
			// Modify the fired node if init_stmt is not null
			if (init_found) {
				if (fired_node->get_op_type () == ST_MINUSEQ)
					fired_node->set_rhs_expr (new uminus_node (fired_node->get_rhs_expr()));
				fired_node->set_op_type (ST_EQ); 
			}
			header_output << fired_node->print_statement (tmp_output, initialized_labels, iters);
			// Print a store if this is the last write
			vector<string> l_labels = fired_node->get_lhs_labels ();
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
				if (last_write.find (*j) != last_write.end () && last_write[*j] == fired_node) 
					tmp_output << dynamic_cast<shiftvec_node*>(fired_node->get_lhs_expr())->print_array () << " = " << *j << ";\n";
			}
		}
		output << header_output.rdbuf () << endl << tmp_output.rdbuf ();
	}
	// Assignments are embedded into following statements (convert the first A=0;A+=B to A=B)
	else if (print == INIT_EMBED && PRINT_INTRINSICS) {
		vector<string> initialized_labels;
		stringstream header_output, tmp_output;
		// Print out all the statements from the fireable list in order
		while (fireable_stmts.size () != 0) {
			stmtnode *fired_node = fireable_stmts.front ();
			fireable_stmts.pop_front ();
			expr_node *lhs = fired_node->get_lhs_expr ();
			// Iterate over initial assignments, and add the initialization if lhs match
			bool init_found = false;
			for (vector<stmtnode*>::iterator i=init.begin(); i!=init.end();) {
				if ((*i)->get_lhs_expr () == lhs) {
					init_found = true;
					i = init.erase (i);
					break;
				}
				else ++i; 
			}
			// Modify the fired node if init_stmt is not null
			if (init_found) {
				if (fired_node->get_op_type () == ST_MINUSEQ)
					fired_node->set_rhs_expr (new uminus_node (fired_node->get_rhs_expr()));
				fired_node->set_op_type (ST_EQ); 
			}
			header_output << fired_node->print_statement (tmp_output, initialized_labels, iters);
			// Print a store if this is the last write
			vector<string> l_labels = fired_node->get_lhs_labels ();
			for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
				if (last_write.find (*j) != last_write.end () && last_write[*j] == fired_node) {
					DATA_TYPE lhs_type = fired_node->get_lhs_expr()->get_type();
					if (lhs_type == DOUBLE) tmp_output << "_mm256_storeu_pd (&";
					else if (lhs_type == FLOAT) tmp_output << "_mm256_storeu_ps (&";
					else tmp_output << "_mm256_storeu_epi64 (&";
					tmp_output << dynamic_cast<shiftvec_node*>(fired_node->get_lhs_expr())->print_array () << ", " << *j << ");\n";
				}
			}
		}
		output << header_output.rdbuf () << endl << tmp_output.rdbuf ();
	}
}

// Compute the primary affinity to labels that are live
int funcdefn::get_primary_affinity_to_live_labels (string s) {
	int p_aff = 0;
	if (primary_affinity.find (s) != primary_affinity.end ()) {
		map<string,int> rhs = primary_affinity[s];
		for (map<string,int>::iterator i=rhs.begin(); i!=rhs.end(); i++) {
			if (is_live (i->first))
				p_aff += i->second; 	
		}
	}
	return p_aff; 
}

// Compute the maximum depth of labels that are live, and have primary affinity to s
float funcdefn::get_primary_depth_to_live_labels (string s) {
	float p_depth = 0.0f;
	int count = 0;
	if (primary_affinity.find (s) != primary_affinity.end ()) {
		map<string,int> rhs = primary_affinity[s];
		for (map<string,int>::iterator i=rhs.begin(); i!=rhs.end(); i++) {
			if (is_live (i->first)) {
				p_depth += live_index (i->first);
				count++;
			}
		}
	}
	return count!=0 ? (p_depth/count) : p_depth; 
}

// Compute the secondary affinity to labels that are live
int funcdefn::get_secondary_affinity_to_live_labels (string s) {
	int s_aff = 0;
	if (secondary_affinity.find (s) != secondary_affinity.end ()) {
		map<string,int> rhs = secondary_affinity[s];
		for (map<string,int>::iterator i=rhs.begin(); i!=rhs.end(); i++) {
			if (is_live (i->first))
				s_aff += i->second; 	
		}
	}
	return s_aff; 
}

// Compute the maximum depth of labels that are live, and have secondary affinity to s
float funcdefn::get_secondary_depth_to_live_labels (string s) {
	float s_depth = 0.0f;
	int count = 0;
	if (secondary_affinity.find (s) != secondary_affinity.end ()) {
		map<string,int> rhs = secondary_affinity[s];
		for (map<string,int>::iterator i=rhs.begin(); i!=rhs.end(); i++) {
			if (is_live (i->first)) {
				s_depth += live_index (i->first);
				count++;
			}
		}
	}
	return count!=0 ? (s_depth/count) : s_depth; 
}

// Check if any schedulable statement's all labels become live if 
// the input label is made live  
int funcdefn::get_first_level_fire_potential (string s) {
	int f_pot = 0;
	for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end(); i++) {
		int frequency = 0;
		if ((*i)->is_label_present (s, frequency)) {
			int n_ct = (*i)->get_nonlive_count () - frequency;
			if (n_ct == 0) f_pot++;
		}
	}
	return f_pot;
}

int funcdefn::get_first_level_non_interlock_fire_potential (string s) {
	int n_i_f_pot = 0;
	for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end(); i++) {
		int frequency = 0;
		if ((*i)->is_label_present (s, frequency)) {
			bool interlocks = false;
			vector<string> r_labels = (*i)->get_rhs_labels ();
			for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
				if (find (interlock_lhs.begin (), interlock_lhs.end (), make_tuple (*j, 0)) != interlock_lhs.end ())
					interlocks = true; 
			}
			if (!interlocks) {
				int n_ct = (*i)->get_nonlive_count () - frequency;
				if (n_ct == 0) n_i_f_pot++;
			}
		}
	}
	return n_i_f_pot;
}

int funcdefn::get_leading_stmt_fire_potential (string s, map<int, vector<int>> leading_stmt_map) {
	int f_pot_lead = 0;
	vector<int> leading_stmts = leading_stmt_map.rbegin()->second;
	for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end(); i++) {
		int st_num = (*i)->get_orig_stmt_num ();
		if (find (leading_stmts.begin(), leading_stmts.end(), st_num) != leading_stmts.end ()) {
			int frequency = 0;
			if ((*i)->is_label_present (s, frequency)) {
				int n_ct = (*i)->get_nonlive_count () - frequency;
				if (n_ct == 0) f_pot_lead++;
			}
		}
	}
	return f_pot_lead;
}

int funcdefn::assign_label_tree_priority (string s) {
	if (DEBUG) assert (stmts_per_label.find (s) != stmts_per_label.end () && "Could not find the label in any original statement");
	vector<int> orig_stmt_vec = stmts_per_label[s];
	int priority = 0;
	for (vector<int>::iterator it=orig_stmt_vec.begin(); it!=orig_stmt_vec.end(); it++) { 
		priority = max (priority, initial_priority[*it]);
		if (clusterwise_stmts_executed[*it]> 0)
			priority = max (priority, (clusterwise_stmts_executed[*it]+total_orig_stmts));
	}
	return priority; 
}

// Check if any schedulable statement's all labels become live if 
// the input label, and one of the labels that it has affinity to is made live  
void funcdefn::get_second_level_fire_potential (string s, vector<string> affinity_labels, int &max_f_pot, int &acc_f_pot) {
	for (vector<string>::iterator i=affinity_labels.begin(); i!=affinity_labels.end(); i++) {
		int l_f_pot = 0;
		for (vector<stmtnode*>::iterator j=schedulable_stmts.begin(); j!=schedulable_stmts.end(); j++) {
			int frequency1 = 0, frequency2 = 0;
			bool label_present = (*j)->is_label_present (s, frequency1);
			label_present |= (*j)->is_label_present (*i, frequency2); 
			if (label_present) {
				int n_ct = (*j)->get_nonlive_count () - frequency1 - frequency2;
				if (n_ct == 0) l_f_pot++;
			}
		}
		max_f_pot = max (max_f_pot, l_f_pot);
		acc_f_pot += l_f_pot;
	}
}

// Check if any label's reuse becomes 0 if the input label is made live
int funcdefn::get_first_level_release_potential (string s) {
	int r_pot = 0;
	map<string,int> reuse = label_reuse;
	for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end(); i++) {
		int frequency = 0;
		if ((*i)->is_label_present (s, frequency)) {
			if (((*i)->get_nonlive_count()-frequency) == 0) {
				// Reduce the reuse of all involved labels in map reuse
				vector<string> l_labels = (*i)->get_lhs_labels ();
				vector<string> r_labels = (*i)->get_rhs_labels ();
				for (vector<string>::iterator i=l_labels.begin(); i!=l_labels.end(); i++) {
					if (DEBUG) assert ((reuse.find (*i) != reuse.end () && reuse[*i] > 0) && "Something wrong with reuse (get_release_potential)"); 
					reuse[*i] = reuse[*i] - 1; 
					if (reuse[*i] == 0) r_pot++; 
				}
				for (vector<string>::iterator i=r_labels.begin(); i!=r_labels.end(); i++) {
					if (DEBUG) assert ((reuse.find (*i) != reuse.end () && reuse[*i] > 0) && "Something wrong with reuse (get_release_potential)"); 
					reuse[*i] = reuse[*i] - 1;
					if (reuse[*i] == 0) r_pot++; 
				}
			}
		}
	}
	reuse.clear ();
	return r_pot;
}

// Check if any label's reuse becomes 0 if the input label is made live
int funcdefn::get_first_level_non_interlock_release_potential (string s) {
	int n_i_r_pot = 0;
	map<string,int> reuse = label_reuse;
	for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end(); i++) {
		int frequency = 0;
		if ((*i)->is_label_present (s, frequency)) {
			if (((*i)->get_nonlive_count()-frequency) == 0) {
				// Reduce the reuse of all involved labels in map reuse
				vector<string> l_labels = (*i)->get_lhs_labels ();
				vector<string> r_labels = (*i)->get_rhs_labels ();
				bool interlocks = false;
				for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
					if (find (interlock_lhs.begin (), interlock_lhs.end (), make_tuple (*j, 0)) != interlock_lhs.end ())
						interlocks = true; 
				}
				if (!interlocks) {
					for (vector<string>::iterator i=l_labels.begin(); i!=l_labels.end(); i++) {
						if (DEBUG) assert ((reuse.find (*i) != reuse.end () && reuse[*i] > 0) && "Something wrong with reuse (get_release_potential)"); 
						reuse[*i] = reuse[*i] - 1; 
						if (reuse[*i] == 0) n_i_r_pot++; 
					}
					for (vector<string>::iterator i=r_labels.begin(); i!=r_labels.end(); i++) {
						if (DEBUG) assert ((reuse.find (*i) != reuse.end () && reuse[*i] > 0) && "Something wrong with reuse (get_release_potential)"); 
						reuse[*i] = reuse[*i] - 1;
						if (reuse[*i] == 0) n_i_r_pot++; 
					}
				}
			}
		}
	}
	reuse.clear ();
	return n_i_r_pot;
}

// Check if any label's reuse becomes 0 if s, and one of non-live labels that s has an affinity to, are made live
void funcdefn::get_second_level_release_potential (string s, vector<string> affinity_labels, int &max_r_pot, int &acc_r_pot) {
	// Now iterate over all the non-live labels that s has affinity to, 
	// and for each label, update reuse metric if it is made live along with s
	for (vector<string>::iterator i=affinity_labels.begin(); i!=affinity_labels.end(); i++) {
		map<string,int> temp_reuse = label_reuse;
		int l_r_pot = 0;
		for (vector<stmtnode*>::iterator j=schedulable_stmts.begin(); j!=schedulable_stmts.end(); j++) {
			int frequency1 = 0, frequency2 = 0;
			bool label_present = (*j)->is_label_present (s, frequency1);
			label_present |= (*j)->is_label_present (*i, frequency2);
			if (label_present && ((*j)->get_nonlive_count()-frequency1-frequency2) == 0) {
				// Reduce the reuse of all involved labels in map reuse
				vector<string> l_labels = (*j)->get_lhs_labels ();
				vector<string> r_labels = (*j)->get_rhs_labels ();
				for (vector<string>::iterator k=l_labels.begin(); k!=l_labels.end(); k++) {
					if (DEBUG) assert ((temp_reuse.find (*k) != temp_reuse.end () && temp_reuse[*k] > 0) && "Something wrong with temp_reuse (get_release_potential)"); 
					temp_reuse[*k] = temp_reuse[*k] - 1;
					if (temp_reuse[*k] == 0) l_r_pot++; 
				}
				for (vector<string>::iterator k=r_labels.begin(); k!=r_labels.end(); k++) {
					if (DEBUG) assert ((temp_reuse.find (*k) != temp_reuse.end () && temp_reuse[*k] > 0) && "Something wrong with temp_reuse (get_release_potential)"); 
					temp_reuse[*k] = temp_reuse[*k] - 1;
					if (temp_reuse[*k] == 0) l_r_pot++; 
				}
			}
		}
		max_r_pot = max (max_r_pot, l_r_pot);
		acc_r_pot += l_r_pot;
		temp_reuse.clear();
	}
}

// Iterate over scatter/gather, and see how many values the input label touches that
// are not yet live
int funcdefn::get_nonlive_values_touched (string s) {
	int n_val = 0;
	// Check the nonlives touched in gather that have reuse greater than 1
	if (gather_contributions.find (s) != gather_contributions.end ()) { 
		vector<string> rhs = gather_contributions[s];
		for (vector<string>::iterator i=rhs.begin(); i!=rhs.end(); i++) {
			if (!is_live (*i) && !single_use (*i)) n_val++;
		}
	}
	// Check the nonlives touched in scatter that have reuse greater than 1
	if (scatter_contributions.find (s) != scatter_contributions.end ()) { 
		vector<string> rhs = scatter_contributions[s];
		for (vector<string>::iterator i=rhs.begin(); i!=rhs.end(); i++) {
			if (!is_live (*i) && !single_use (*i)) n_val++;
		}
	}
	return n_val;
}

// For all the schedulable statements that this label participates in, return the number of 
// statements that do not read/write interlocked values 
int funcdefn::get_non_interlock_value (string s) {
	int non_interlock_val = 0;
	for (vector<stmtnode*>::iterator i=schedulable_stmts.begin(); i!=schedulable_stmts.end(); i++) {
		if ((*i)->is_label_present (s)) {
			// lhs will not interlock, since lhs can only be accumulation
			bool interlocks = false;
			vector<string> r_labels = (*i)->get_rhs_labels ();
			for (vector<string>::iterator j=r_labels.begin(); j!=r_labels.end(); j++) {
				if (find (interlock_lhs.begin (), interlock_lhs.end (), make_tuple (*j, 0)) != interlock_lhs.end ())
					interlocks = true; 
			}
			if (!interlocks) ++non_interlock_val;
		}
	}
	return non_interlock_val;
}

/* Populate the first level of incoming tuple. 
   The 1st field is the release potential.
   The 2nd field is fire potential in the leading statement.
   The 3rd field is the priority wrt the leading set
   The 4th field is the fire potential 
   The 9th field is the primary affinity.
   The 10th field is the secondary affinity.
   The 15th field is the values touched that aren't live.
   The 16th field is the release potential for non-interlocking stmts
   The 17th field is the fire potential for non-interlocking stmts 
   The 18th field is the number of statements this label contributes to that do not interlock with the last few statements
   The 19th field is the index offsets. */
tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>> funcdefn::fill_first_level_metric (string s, map<int, vector<int>> leading_stmt_map) {
	int r_pot = get_first_level_release_potential (s);
	int f_pot_lead = leading_stmt_map.empty() ? 0 : get_leading_stmt_fire_potential (s, leading_stmt_map);
	int reg_tree_priority = assign_label_tree_priority (s);
	int f_pot = get_first_level_fire_potential (s);
	int p_aff = get_primary_affinity_to_live_labels (s);
	int s_aff = get_secondary_affinity_to_live_labels (s);
	int n_val = get_nonlive_values_touched (s);
	int n_i_r_pot = get_first_level_non_interlock_release_potential (s); 
	int n_i_f_pot = get_first_level_non_interlock_fire_potential (s); 
	int n_i_val = get_non_interlock_value (s);
	vector<int> index_offsets;
	if (DEBUG) assert (label_to_node_map.find (s) != label_to_node_map.end () && "Label not present in label_to_node_map (fill_first_level_metric)");
	expr_node *a_expr = label_to_node_map[s];
	// Pad with 0's 
	int vec_size = 0;
	if (a_expr->get_expr_type () == T_SHIFTVEC) {
		shiftvec_node *a_vec = dynamic_cast<shiftvec_node*> (a_expr);
		vec_size = a_vec->get_index_size ();
	}
	for (int i=0; i<dim-vec_size; i++) 
		index_offsets.push_back (0);
	// Put real offsets for arrays
	if (a_expr->get_expr_type () == T_SHIFTVEC) {
		shiftvec_node *a_vec = dynamic_cast<shiftvec_node*> (a_expr);
		a_vec->lexical_index_offsets (index_offsets);
	}
	return make_tuple (s, r_pot, f_pot_lead, reg_tree_priority, f_pot, 0, 0, 0, 0, p_aff, s_aff, 0, 0, 0, 0, n_val, n_i_r_pot, n_i_f_pot, n_i_val, index_offsets);
}

/* Compute the second level of metrics. 
   The 5th field is the max of release potential of what I have primary affinity with
   The 6th field is the max of fire potential of what I have primary affinity with
   The 7th field is the sum of release potential of what I have primary affinity with
   The 8th field is the sum of fire potential of what I have primary affinity with
   The 11th field is the max of primary affinity of what I have primary affinity with
   The 12th field is the max of secondary affinity of what I have primary affinity with 
   The 13th field is the sum of primary affinity of what I have primary affinity with 
   The 14th field is the sum of secondary affinity of what I have primary affinity with */
void funcdefn::fill_second_level_metric (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>> &order_metric) {
	vector<string> metric_labels;
	for (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>>::iterator i=order_metric.begin(); i!=order_metric.end(); i++)
		metric_labels.push_back (get<0>(*i));

	for (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>>::iterator i=order_metric.begin(); i!=order_metric.end(); i++) {
		string s = get<0>(*i);
		vector<string> affinity_labels;
		if (primary_affinity.find (s) != primary_affinity.end ()) {
			map<string,int> rhs = primary_affinity[s];
			for (map<string,int>::iterator i=rhs.begin(); i!=rhs.end(); i++) {
				if (find (metric_labels.begin(), metric_labels.end(), i->first) != metric_labels.end () && find (affinity_labels.begin (), affinity_labels.end (), i->first) == affinity_labels.end ())
					affinity_labels.push_back (i->first);
			}
		}
		// Compute second level release potential
		int max_r_pot = 0, acc_r_pot = 0;
		get_second_level_release_potential (s, affinity_labels, max_r_pot, acc_r_pot);
		int max_f_pot = 0, acc_f_pot = 0;
		get_second_level_fire_potential (s, affinity_labels, max_f_pot, acc_f_pot);
		get<5>(*i) = max_r_pot;
		get<6>(*i) = max_f_pot;
		get<7>(*i) = acc_r_pot;
		get<8>(*i) = acc_f_pot;
		int max_p_aff = 0, max_s_aff = 0;
		int acc_p_aff = 0, acc_s_aff = 0;
		for (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>>::iterator j=order_metric.begin(); j!=order_metric.end(); j++) {
			if (find (affinity_labels.begin(), affinity_labels.end(), get<0>(*j)) != affinity_labels.end()) {
				max_p_aff = max (max_p_aff, get<9>(*j));
				acc_p_aff += get<9>(*j);
				max_s_aff = max (max_s_aff, get<10>(*j));
				acc_s_aff += get<10>(*j);
			}
		}
		get<11>(*i) = max_p_aff;
		get<12>(*i) = max_s_aff;
		get<13>(*i) = acc_p_aff;
		get<14>(*i) = acc_s_aff; 
	}
}

void funcdefn::print_order_metric (void) {
	if (FIRST_LEVEL && !SECOND_LEVEL) { 
		for (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>>::iterator i=order_metric.begin(); i!=order_metric.end(); i++) {
			cout << "\nThe metric for " << (get<0>(*i)).c_str() << " is : (" << get<1>(*i) << ", " << get<2>(*i) << ", " << get<3>(*i) << ", " << get<4>(*i) << ", " << get<9>(*i) << ", " << get<10>(*i) << ", " << get<15>(*i) << ", " << get<16>(*i) << ", " << get<17>(*i) << ", " << get<18>(*i) << ", [ ";
			vector<int> index_offsets = get<19>(*i);
			for (int i=0; i<index_offsets.size(); i++)
				cout << index_offsets[i] << " ";
			cout << "])"; 
		}
	}
	else if (SECOND_LEVEL) {
		for (vector<tuple<string,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,vector<int>>>::iterator i=order_metric.begin(); i!=order_metric.end(); i++) {
			cout << "\nThe metric for " << (get<0>(*i)).c_str() << " is : (" << get<1>(*i) << ", " << get<2>(*i) << ", " << get<3>(*i) << ", " << get<4>(*i) << ", " << get<5>(*i) << ", " << get<6>(*i) << ", " << get<7>(*i) << ", " << get<8>(*i) << ", " << get<9>(*i) << ", " << get<10>(*i) << ", " << get<11>(*i) << ", " << get<12>(*i) << ", " << get<13>(*i) << ", " << get<14>(*i) << ", " << get<15>(*i) << ", " << get<16>(*i) << ", " <<  get<17>(*i) << ", " << get<18>(*i) << ", [ ";
			vector<int> index_offsets = get<19>(*i);
			for (int i=0; i<index_offsets.size(); i++)
				cout << index_offsets[i] << " ";
			cout << "])"; 
		}
	}
}

void funcdefn::print_spill_metric (vector<tuple<string, int, int, int, int, int>> spill_metric) {
	for (vector<tuple<string, int, int, int, int, int>>::const_iterator i=spill_metric.begin(); i!=spill_metric.end(); i++) {
		cout << "\nThe spill metric for " << (get<0>(*i)).c_str() << " is : (" << get<1>(*i) << ", " << get<2>(*i) << ", " << get<3>(*i) << ", " << get<4>(*i) << ", " << get<5>(*i) << ")";	
	}
}

bool start_node::is_array_decl (string s) {
	for (vector<array_decl *>::const_iterator i=array_decls.begin(); i!=array_decls.end(); i++) {
		string name = (*i)->get_array_name ();
		if (s.compare (name) == 0) return true;
	}
	return false;
}

DATA_TYPE start_node::get_var_type (string s) {
	DATA_TYPE ret = INT;
	map<string, DATA_TYPE> list = var_decls->get_symbol_list ();
	for (map<string, DATA_TYPE>::const_iterator i=list.begin(); i!=list.end(); i++) {
		string name = i->first;
		if (s.compare (name) == 0) 
			ret = i->second;
	}
	return ret;
}

DATA_TYPE start_node::get_array_type (string s) {
	DATA_TYPE ret = INT;
	for (vector<array_decl *>::const_iterator i=array_decls.begin(); i!=array_decls.end(); i++) {
		string name = (*i)->get_array_name ();
		if (s.compare (name) == 0) 
			ret = (*i)->get_array_type ();
	}
	return ret;
}

int start_node::get_max_dimensionality (void) {
	int ret = 1;
	for (vector<array_decl *>::const_iterator i=array_decls.begin(); i!=array_decls.end(); i++) {
		string name = (*i)->get_array_name ();
		vector<array_range *> range = (*i)->get_array_range ();
		ret = max (ret, (int) range.size ());
	}
	return ret;
}

int start_node::get_array_dimensionality (string s) {
	for (vector<array_decl *>::const_iterator i=array_decls.begin(); i!=array_decls.end(); i++) {
		string name = (*i)->get_array_name ();
		if (s.compare (name) == 0) {
			vector<array_range *> range = (*i)->get_array_range ();
			return range.size ();
		}
	}
	// Assume that the input is a variable
	return 1;
}

bool start_node::is_temp_decl (string s) {
	for (vector<string>::const_iterator i=temp_decls.begin(); i!=temp_decls.end(); i++) {
		if (s.compare (*i) == 0) 
			return true;
	}
	return false;
}

bool start_node::is_incoming_decl (string s) {
	vector<func_call *> calls = get_func_calls ();
	for (vector<func_call *>::const_iterator j=calls.begin(); j!=calls.end(); j++) {
		vector<string> out = (*j)->get_out_list ();
		for (vector<string>::const_iterator k=out.begin(); k!=out.end(); k++) {
			if (s.compare (*k) == 0)
				return true; 
		}
	}
	return !is_temp_decl (s);
}

void start_node::push_temp_decl (string_list * list) {
	vector<string> temp_vec = list->get_list ();
	for (vector<string>::const_iterator i=temp_vec.begin(); i!=temp_vec.end(); i++) {
		if (DEBUG) assert (find (temp_decls.begin (), temp_decls.end (), (*i)) == temp_decls.end () && "Temporary array declared twice");
		temp_decls.push_back (*i);
	}
}

void start_node::push_unroll_decl (map<string,int> &ud) {
	unroll_decls = ud;
}

void start_node::push_unroll_decl (char *s, int val) {
	string name = string (s);
	if (DEBUG) assert (val > 0 && "Unroll factor must be greater than 0");
	if (DEBUG) assert (unroll_decls.find (name) == unroll_decls.end () && "Unroll factor declared twice");
	if (val > 1) unroll_decls[s] = val;	 
}

void start_node::push_iterator (char *s) {
	iters.push_back (string (s));
}

void start_node::push_coefficient (char *s) {
	coefficients.push_back (string (s));
}

void start_node::set_reg_limit (int val) {
	max_reg = val;
}

int start_node::get_reg_limit (void) {
	return max_reg;
}

range_list *start_node::get_range_list (string s) {
	for (vector<array_decl *>::const_iterator j=array_decls.begin(); j!=array_decls.end(); j++) {
		string array_name = (*j)->get_array_name ();
		if (s.compare (array_name) == 0) 
			return (*j)->get_range_list ();
	}
}

vector<array_range *> start_node::get_array_range (string s) {
	for (vector<array_decl *>::const_iterator j=array_decls.begin(); j!=array_decls.end(); j++) {
		string array_name = (*j)->get_array_name ();
		if (s.compare (array_name) == 0) 
			return (*j)->get_array_range ();
	}
}

funcdefn *start_node::get_func_defn (string s) {
	map<string, funcdefn*> func_defn = func_defns->get_symbol_list ();
	for (map<string, funcdefn*>::const_iterator j=func_defn.begin(); j!=func_defn.end(); j++) {
		if (s.compare (j->first) == 0)
			return j->second;
	}
	if (DEBUG) assert (0 && "Called func is not defined");
}
