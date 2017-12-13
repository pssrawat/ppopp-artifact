#include "treenode.hpp"
using namespace std;

// Add an operand to the RHS expression
void accnode::add_operand (treenode *node, map<string, int> label_bitset_index, int label_count) {
	used_labels |= node->get_used_labels ();
	addArrays (use_frequency, node->get_use_frequency(), label_count);
	// Increment use frequency of node's lhs label
	string lhs_label = node->get_lhs_label ();
	if (!node->is_data_node ()) use_frequency[label_bitset_index[lhs_label]] += 1;
	rhs_operands.push_back (node);
}

// Populate the bitset of all leaf nodes
void accnode::compute_leaf_nodes (boost::dynamic_bitset<> &leaf_bitset, map<string, int> label_bitset_index) {
	for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
		if ((*it)->is_data_node ()) continue;
		(*it)->compute_leaf_nodes (leaf_bitset, label_bitset_index);
	}
}

// Returns true if this rhs expr has any only leafs in it
bool accnode::subtree_has_only_leafs (void) {
	bool leaf_children = true;
	for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
		if ((*it)->is_data_node ()) continue;
		if (!(*it)->is_leaf_node ()) leaf_children = false;
	}
	return leaf_children;
}

// Modify the rhs_expr after slicing, i.e. recompute the use frequencies from scratch
void accnode::recompute_rhs_expr (map<string, int> label_bitset_index, vector<string> &cull_labels, int label_count) {
	used_labels.reset ();
	resetArray (use_frequency, label_count);
	for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
		if ((*it)->is_data_node ()) continue;
		(*it)->recompute_tree (label_bitset_index, cull_labels, label_count);
		used_labels |= (*it)->get_used_labels () | (*it)->get_appended_labels ();
		string lhs_label = (*it)->get_lhs_label ();
		addArrays (use_frequency, (*it)->get_use_frequency(), label_count);
		addArrays (use_frequency, (*it)->get_appended_frequency(), label_count);
		use_frequency[label_bitset_index[lhs_label]] += 1;
	}
	if (DEBUG) {
		printf ("In inter, the label use bitset for expr %s after recompute is ", expr_string.c_str());
		print_bitset (label_bitset_index, used_labels, use_frequency, label_count);
	}
}

// Same as above, but put in the minimal trees (having leaf as children) in the computations vector
bool accnode::recompute_rhs_expr (map<string, int> label_bitset_index, vector<tuple<treenode*,accnode*>> &computations, vector<string> &cull_labels, int label_count) {
	used_labels.reset ();
	resetArray (use_frequency, label_count);
	bool leaf_children = true;
	for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
		if ((*it)->is_data_node ()) continue;
		(*it)->recompute_tree (label_bitset_index, computations, cull_labels, label_count);
		used_labels |= (*it)->get_used_labels () | (*it)->get_appended_labels ();
		string lhs_label = (*it)->get_lhs_label ();
		addArrays (use_frequency, (*it)->get_use_frequency(), label_count);
		addArrays (use_frequency, (*it)->get_appended_frequency(), label_count);
		use_frequency[label_bitset_index[lhs_label]] += 1;
		if (!(*it)->is_leaf_node ()) leaf_children = false;
	}
	if (DEBUG) {
		printf ("In intra, the label use bitset for expr %s after recompute is ", expr_string.c_str());
		print_bitset (label_bitset_index, used_labels, use_frequency, label_count);
	}
	return leaf_children;
}

// Populate the bitset of all leaf nodes
void treenode::compute_leaf_nodes (boost::dynamic_bitset<> &leaf_bitset, map<string, int> label_bitset_index) {
	if (is_data_node ()) return;
	if (is_leaf_node ()) leaf_bitset[label_bitset_index[lhs_label]] = true;
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) 
		(*it)->compute_leaf_nodes (leaf_bitset, label_bitset_index);
}

// Add a new RHS node to the tree (in case of accumulation)
void treenode::add_rhs_expr (accnode *node, map<string, int> label_bitset_index, int label_count) {
	// First perform a union of the liveset of child nodes
	used_labels |= node->get_used_labels ();
	addArrays (use_frequency, node->get_use_frequency(), label_count);
	rhs_accs.push_back (node);
	// Increment use frequency of LHS
	use_frequency[label_bitset_index[lhs_label]] += 1;
}

//void accnode::append_to_code (treenode *tree) {
//	if (tree->is_data_node () || tree->is_leaf_node ()) return;
//	vector<accnode*> t_rhs_accs = tree->get_rhs_operands ();
//	for (vector<accnode*>::iterator it=t_rhs_accs.begin(); it!=t_rhs_accs.end(); it++) {
//		//Recursively append the code of the subtrees
//		vector<treenode*> t_rhs_operands = (*it)->get_operands ();
//		for (vector<treenode*>::iterator jt=t_rhs_operands.begin(); jt!=t_rhs_operands.end(); jt++)
//			append_to_code (*jt);
//		// Now print the code for the tree
//		code_string << tree->get_lhs_label ();
//		STMT_OP asgn_op = (*it)->get_assignment_op ();
//		code_string << print_stmt_op (asgn_op);
//		if (tree->is_uminus_node ()) code_string << "-(";
//		code_string << (*it)->get_expr_string ();
//		if (tree->is_uminus_node ()) code_string << ")";
//		code_string << ";" << endl;
//	}
//}

void accnode::add_spliced_treenode (treenode *tree) {
	// First check if there is another tree with the same lhs
	string tree_lhs = tree->get_lhs_label ();
	bool same_lhs_found = false;
	for (vector<treenode*>::iterator it=spliced_treenodes.begin(); it!=spliced_treenodes.end(); it++) {
		if (tree_lhs.compare ((*it)->get_lhs_label ()) == 0) {
			vector<accnode*> new_rhs = tree->get_rhs_operands ();
			((*it)->get_rhs_operands()).insert (((*it)->get_rhs_operands()).end(), new_rhs.begin(), new_rhs.end());
			same_lhs_found = true;
			break;
		}
	}
	if (!same_lhs_found) 
		spliced_treenodes.push_back (tree);
}

//void treenode::append_to_code (treenode *tree) {
//	if (tree->is_data_node () || tree->is_leaf_node ()) return;
//	vector<accnode*> t_rhs_accs = tree->get_rhs_operands ();
//	for (vector<accnode*>::iterator it=t_rhs_accs.begin(); it!=t_rhs_accs.end(); it++) {
//		// Recursively append the code of the subtrees
//		vector<treenode*> t_rhs_operands = (*it)->get_operands ();
//		for (vector<treenode*>::iterator jt=t_rhs_operands.begin(); jt!=t_rhs_operands.end(); jt++) 
//			append_to_code (*jt);
//		// Now print the code for the tree
//		code_string << tree->get_lhs_label ();
//		STMT_OP asgn_op = (*it)->get_assignment_op ();
//		code_string << print_stmt_op (asgn_op);
//		if (tree->is_uminus_node ()) code_string << "-(";
//		code_string << (*it)->get_expr_string ();
//		if (tree->is_uminus_node ()) code_string << ")";
//		code_string << ";" << endl;
//	}
//}

//string treenode::print_tree () {
//	stringstream output;
//	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
//		// First make sure that all the trees are printed
//		vector<treenode*> rhs_operands = (*it)->get_operands ();
//		for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) 
//			output << (*jt)->print_tree ();
//		// Now print this tree;
//		output << lhs_label;
//		STMT_OP asgn_op = (*it)->get_assignment_op ();
//		output << print_stmt_op (asgn_op);
//		if (is_uminus_node ()) output << "-(";
//		output << (*it)->get_expr_string ();
//		if (is_uminus_node ()) output << ")";
//		output << ";" << endl;
//		// Now print the code string
//		output << (*it)->get_code_string ();
//	}
//	output << code_string.str();
//	return output.str ();
//}

void treenode::copy_propagation (map<string, treenode*> &asgn_map, unsigned int *label_frequency, map<string, int> label_bitset_index) {
	if (is_leaf_node () || is_data_node ()) return;
	// Recursively perform copy propagation
    for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		vector<treenode*> rhs_operands = (*it)->get_operands ();
		for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++)
			(*jt)->copy_propagation (asgn_map, label_frequency, label_bitset_index);	
	}
	// Perform copy propagation from the map
    for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end();) {
		vector<treenode*> &rhs_operands = (*it)->get_operands ();
		if (rhs_operands.size () == 1) {
			string rhs_label = (rhs_operands.front())->get_lhs_label ();
			if (label_frequency[label_bitset_index[rhs_label]] == 2 && asgn_map.find (rhs_label) != asgn_map.end ()) {
				// Put the rhs from original tree here
				accnode *t_acc = ((asgn_map[rhs_label])->get_rhs_operands()).front ();
				bool uminus_val = ((rhs_operands.front())->is_uminus_node () != (t_acc->get_operands()).front()->is_uminus_node ());
				(t_acc->get_operands().front())->set_uminus_val (uminus_val);
				t_acc->set_assignment_op ((*it)->get_assignment_op ());
				it = rhs_accs.erase (it);
				rhs_accs.insert (it, t_acc);
				// Cull the original tree
				asgn_map[rhs_label]->reset_rhs_accs ();
				asgn_map.erase (rhs_label);
			}
			else it++;
		}
		else it++;
	}

	// Push the simple assignment in the map
	bool simple_asgn = (rhs_accs.size () == 1 && (rhs_accs.front())->is_asgn_eq_op () && label_frequency[label_bitset_index[lhs_label]] == 2 && spliced_treenodes.empty ());
	if (simple_asgn) {
		if (DEBUG) assert (asgn_map.find (lhs_label) == asgn_map.end () && "A same LHS already appears in the tree (copy_propagation)");
		asgn_map[lhs_label] = this;
	}
}

void treenode::allocate_registers (stringstream &output, int &reg_count, queue<int> &avail_regs, map<string,int> &alloc_map, unsigned int *label_frequency, unsigned int *label_use, map<string, int> label_bitset_index) {
	if (is_data_node () || is_leaf_node ()) return;
	// Assign register to LHS
	int lhs_reg;
	if (alloc_map.find (lhs_label) == alloc_map.end ()) {
		if (avail_regs.size () > 0) {
			lhs_reg = avail_regs.front ();
			avail_regs.pop ();
		}
		else {
			lhs_reg = reg_count;
			reg_count += 1;
		}
		alloc_map[lhs_label] = lhs_reg;
	}
	else 
		lhs_reg = alloc_map[lhs_label]; 

	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		vector<treenode*> rhs_operands = (*it)->get_operands ();
		// Recursively allocate registers
		for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++)
			(*jt)->allocate_registers (output, reg_count, avail_regs, alloc_map, label_frequency, label_use, label_bitset_index);

		stringstream tmp_output;
		tmp_output << "_r_" << lhs_reg << "_";
		STMT_OP asgn_op = (*it)->get_assignment_op ();
		tmp_output << print_stmt_op (asgn_op);
		if (is_uminus_node ()) tmp_output << "-(";
		int single_rhs = true;
		for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) {
			string rhs_operand = (*jt)->get_lhs_label ();
			if (!single_rhs) tmp_output << print_operator ((*it)->get_rhs_operator ());
			if ((*jt)->is_uminus_node ()) tmp_output << "-";
			int rhs_reg;
			if (alloc_map.find (rhs_operand) == alloc_map.end ()) {
				if (avail_regs.size () > 0) {
					rhs_reg = avail_regs.front ();
					avail_regs.pop ();
				}
				else {
					rhs_reg = reg_count;
					reg_count += 1;
				}
				alloc_map[rhs_operand] = rhs_reg;
				output << "_r_" << alloc_map[rhs_operand] << "_ = " << rhs_operand << ";\n";
			}
			else 
				rhs_reg = alloc_map[rhs_operand];
 
			tmp_output << "_r_" << rhs_reg << "_";
			single_rhs = false;
		}
		if (is_uminus_node ()) tmp_output << ")";
		tmp_output << ";" << endl;
		output << tmp_output.str ();
		for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) {
			string rhs_operand = (*jt)->get_lhs_label ();
			if (DEBUG) assert (alloc_map.find (rhs_operand) != alloc_map.end () && "Error in allocating register (allocate_registers)");
			// Increase the use of RHS, and free register if all uses are exhausted
			int idx = label_bitset_index[rhs_operand];
			label_use[idx] += 1;
			if (label_use[idx] == label_frequency[idx]) { 
				avail_regs.push (alloc_map[rhs_operand]);
				alloc_map.erase (rhs_operand);
			}
		}

		// Increase the use of LHS, and free register if all uses are exhausted
		int idx = label_bitset_index[lhs_label];
		label_use[idx] += 1;
		if (label_use[idx] == label_frequency[idx]) {
			output << lhs_label << " = _r_" << lhs_reg << "_;\n";
			avail_regs.push (lhs_reg);
		}

		// Allocate registers to the spliced subtrees
		vector<treenode*> tnodes = (*it)->get_spliced_treenodes ();
		if (tnodes.size () > 0) {
			for (vector<treenode*>::iterator jt=tnodes.begin(); jt!=tnodes.end(); jt++) 
				(*jt)->allocate_registers (output, reg_count, avail_regs, alloc_map, label_frequency, label_use, label_bitset_index);
		}
	}
	// Allocate registers to the spliced subtrees
	for (vector<treenode*>::iterator jt=spliced_treenodes.begin(); jt!=spliced_treenodes.end(); jt++)
		(*jt)->allocate_registers (output, reg_count, avail_regs, alloc_map, label_frequency, label_use, label_bitset_index);
}

void treenode::print_temp_type (vector<expr_node*> &tvars, expr_node *lhs, stringstream &output, DATA_TYPE gdata_type) {
    for (vector<expr_node*>::iterator i=tvars.begin(); i!=tvars.end();) {
        if (lhs == *i) {
            output << print_data_type (gdata_type);
            tvars.erase (i);
        }
        else ++i;
    }
}

void treenode::print_finalized_tree (stringstream &output, vector<expr_node*> &tvars, vector<expr_node*> &init, map<string,string> &lhs_init, DATA_TYPE gdata_type) {
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		vector<treenode*> rhs_operands = (*it)->get_operands ();
		for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) 
			(*jt)->print_finalized_tree (output, tvars, init, lhs_init, gdata_type);

		// For printing the initializations
        print_temp_type (tvars, lhs, output, gdata_type);
        // Iterate over initial assignments, and add the initialization if lhs match
        bool init_found = false;
        for (vector<expr_node*>::iterator i=init.begin(); i!=init.end();) {
            if (*i == lhs) {
                init_found = true;
                init.erase (i);
                break;
            }
            else ++i;
        }
		STMT_OP asgn_op = (*it)->get_assignment_op ();
        // Modify the fired node if init_stmt is not null
		bool uminus_node = is_uminus_node ();
        if (init_found) {
		   if (asgn_op == ST_MINUSEQ) 
				uminus_node = true;
           asgn_op = ST_EQ;
		}
		// In printing lhs, print the initialization if the lhs has not been seen yet
		if (lhs_init.find (lhs_label) == lhs_init.end() && lhs_label.compare (lhs_printing)!=0) {
			lhs_init[lhs_label] = lhs_printing;
			output << print_data_type (gdata_type) << lhs_label;
			if (asgn_op != ST_EQ) {
				output << " = " << lhs_printing << ";\n";
				output << lhs_label;	
			}
		}
		else {
			output << lhs_label;
		}

		output << print_stmt_op (asgn_op);
		if (uminus_node) output << "-(";
		output << (*it)->get_rhs_printing_string ();
		if (uminus_node) output << ")";
		output << ";" << endl;
		vector<treenode*> tnodes = (*it)->get_spliced_treenodes ();
		if (tnodes.size () > 0) {
			for (vector<treenode*>::iterator jt=tnodes.begin(); jt!=tnodes.end(); jt++) 
				(*jt)->print_finalized_tree (output, tvars, init, lhs_init, gdata_type);
		}
	}
	for (vector<treenode*>::iterator jt=spliced_treenodes.begin(); jt!=spliced_treenodes.end(); jt++)
		(*jt)->print_finalized_tree (output, tvars, init, lhs_init, gdata_type);
}

void treenode::add_spliced_treenode (treenode *tree) {
	// First check if there is another tree with the same lhs
	string tree_lhs = tree->get_lhs_label ();
	bool same_lhs_found = false;
	for (vector<treenode*>::iterator it=spliced_treenodes.begin(); it!=spliced_treenodes.end(); it++) {
		if (tree_lhs.compare ((*it)->get_lhs_label ()) == 0) {
			vector<accnode*> new_rhs = tree->get_rhs_operands ();
			((*it)->get_rhs_operands()).insert (((*it)->get_rhs_operands()).end(), new_rhs.begin(), new_rhs.end());
			same_lhs_found = true;
			break;
		}
	}
	if (!same_lhs_found) 
		spliced_treenodes.push_back (tree);
}

bool treenode::append_subtree_to_code (treenode *tree, boost::dynamic_bitset<> leaf_labels, boost::dynamic_bitset<> single_use_labels, map<string, int> label_bitset_index, int label_count) {
	bool appended = false;
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		vector<treenode*> rhs_operands = (*it)->get_operands ();
		for (vector<treenode*>::reverse_iterator jt=rhs_operands.rbegin(); jt!=rhs_operands.rend(); jt++) {
			appended |= (*jt)->append_subtree_to_code (tree, leaf_labels, single_use_labels, label_bitset_index, label_count);
			if (appended) break;
		}
		if (!appended) {
			boost::dynamic_bitset<> tree_labels (label_count);
			tree_labels = tree->get_used_labels () & leaf_labels;
			boost::dynamic_bitset<> rhs_labels (label_count);
			rhs_labels = (*it)->get_used_labels ();
			rhs_labels[label_bitset_index[lhs_label]] = true;
			boost::dynamic_bitset<> a_bitset (label_count);
			a_bitset = tree_labels & ~(rhs_labels | (*it)->get_appended_labels() | single_use_labels);
			if (a_bitset.none ()) {
				//(*it)->append_to_code (tree);
				(*it)->update_appended_info (tree, label_bitset_index, label_count);
				// Add the treenode to the tree
				(*it)->add_spliced_treenode (tree);
				if (DEBUG) {
					printf ("Appending tree with %s to tree with %s - %s\n", tree->get_lhs_label().c_str(), lhs_label.c_str(), (*it)->get_expr_string().c_str());
					printf ("tree_labels : "); print_bitset (label_bitset_index, tree_labels, label_count);
					printf ("rhs_used_labels : "); print_bitset (label_bitset_index, rhs_labels, label_count); 
					printf ("single used_labels : "); print_bitset (label_bitset_index, single_use_labels, label_count); 
					printf ("appended used_labels : "); print_bitset (label_bitset_index, (*it)->get_appended_labels (), label_count);
				}
				appended = true;
			}
		}
		if (appended) break;
	}
	return appended;
}

void accnode::update_appended_info (treenode *tree, map<string, int> label_bitset_index, int label_count) {
	appended_labels |= tree->get_used_labels ();
	addArrays (appended_frequency, tree->get_use_frequency(), label_count);
}

void treenode::update_appended_info (treenode *tree, map<string, int> label_bitset_index, int label_count) {
	appended_labels |= tree->get_used_labels ();
	addArrays (appended_frequency, tree->get_use_frequency(), label_count);
}

string treenode::print_tree (vector<treenode*> &subtree_vec, boost::dynamic_bitset<>leaf_labels, boost::dynamic_bitset<>single_use_labels, int label_count) {
	stringstream output;
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		// First make sure that all the trees are printed
		vector<treenode*> rhs_operands = (*it)->get_operands ();
		for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) 
			output << (*jt)->print_tree (subtree_vec, leaf_labels, single_use_labels, label_count);
		// Now print this tree;
		output << lhs_label;
		STMT_OP asgn_op = (*it)->get_assignment_op ();
		output << print_stmt_op (asgn_op);
		if (is_uminus_node ()) output << "-(";
		output << (*it)->get_expr_string ();
		if (is_uminus_node ()) output << ")";
		output << ";" << endl;
		vector<treenode*> t_vec;
		// Now iterate over subtree_vec, and find subtrees that use the same subset of labels as the current tree
		for (vector<treenode*>::iterator kt=subtree_vec.begin(); kt!=subtree_vec.end();) {
			boost::dynamic_bitset<> a_bitset (label_count);
			a_bitset = ((*kt)->get_used_labels() & leaf_labels) & ~((*it)->get_used_labels() | (*it)->get_appended_labels() | single_use_labels);
			if (a_bitset.none ()) {
				t_vec.push_back (*kt);
				kt = subtree_vec.erase (kt);
			}
			else kt++;
		}
		for (vector<treenode*>::iterator kt=t_vec.begin(); kt!=t_vec.end(); kt++) 
			output << (*kt)->print_tree (subtree_vec, leaf_labels, single_use_labels, label_count);
	}
	return output.str ();
}

tuple<boost::dynamic_bitset<>,unsigned int*> treenode::compute_liveness (boost::dynamic_bitset<> &livein, unsigned int *livein_freq, unsigned int *label_frequency, int label_count) {
	livein |= used_labels | appended_labels;
	addArrays (livein_freq, use_frequency, label_count);
	addArrays (livein_freq, appended_frequency, label_count);
	for (int i=0; i<label_count; i++) {
		if (livein[i] == true && livein_freq[i] == label_frequency[i])
			livein[i] = false; 
	}
	boost::dynamic_bitset<> liveout (livein);
	unsigned int *liveout_freq = new unsigned int[label_count] ();
	copyArray (liveout_freq, livein_freq, label_count);
	return make_tuple (liveout, liveout_freq);
}

void treenode::create_expr_lhs_map (map<string,tuple<string,expr_node*>> &expr_lhs_map) {
	if (is_data_node () || is_leaf_node ()) return;
	if (!is_accumulation_node ()) {
		for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
			string rhs_expr = (*it)->get_expr_string ();
			if ((*it)->is_asgn_eq_op ()) {
				if (expr_lhs_map.find (rhs_expr) == expr_lhs_map.end ())
					expr_lhs_map[rhs_expr] = make_tuple (lhs_label, lhs);
			}
		}
	}
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		vector<treenode*> rhs_operands = (*it)->get_operands ();
		for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) {
			(*jt)->create_expr_lhs_map (expr_lhs_map);
		}
	}
}

// Modify the tree after slicing, i.e. recompute the use frequencies from scratch
void treenode::recompute_tree (map<string, int> label_bitset_index, vector<string> &cull_labels, int label_count) {
	// Iterate over accumulations
	if (is_data_node ()) return;
	used_labels.reset ();
	used_labels[label_bitset_index[lhs_label]] = true;
	resetArray (use_frequency, label_count);
	// Check if the lhs_label is in cull_labels. If yes, then clear the rhs operands and return
	if (!is_accumulation_node ()) {
		if (find (cull_labels.begin(), cull_labels.end(), lhs_label) != cull_labels.end ()) {
			rhs_accs.clear ();
			//return;
		}
	}
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		(*it)->recompute_rhs_expr (label_bitset_index, cull_labels, label_count);
		used_labels |= (*it)->get_used_labels () | (*it)->get_appended_labels ();
		addArrays (use_frequency, (*it)->get_use_frequency(), label_count);
		addArrays (use_frequency, (*it)->get_appended_frequency(), label_count);
		// Increment use frequency of LHS if it is an accumulation
		use_frequency[label_bitset_index[lhs_label]] += 1;
	}
	// Add the information in appended labels
	used_labels |= appended_labels;
	addArrays (use_frequency, appended_frequency, label_count);
	if (DEBUG) {
		printf ("In inter, the label use bitset for tree %s after recompute is ", lhs_label.c_str());
		print_bitset (label_bitset_index, used_labels, use_frequency, label_count);
	}
}

// Same as above, but return a vector of minimal trees visited (trees that have leaf as an operand)
void treenode::recompute_tree (map<string,int> label_bitset_index, vector<tuple<treenode*,accnode*>> &computations, vector<string> &cull_labels, int label_count) {
	// Iterate over accumulations
	if (is_data_node ()) return;
	used_labels.reset ();
	used_labels[label_bitset_index[lhs_label]] = true;
	resetArray (use_frequency, label_count);
	// Check if the lhs_label is in cull_labels. If yes, then clear the rhs operands and return
	if (find (cull_labels.begin(), cull_labels.end(), lhs_label) != cull_labels.end ()) {
		rhs_accs.clear ();
		//return;
	}
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		bool expr_has_leafs = (*it)->recompute_rhs_expr (label_bitset_index, computations, cull_labels, label_count);
		used_labels |= (*it)->get_used_labels () | (*it)->get_appended_labels ();
		addArrays (use_frequency, (*it)->get_use_frequency(), label_count);
		addArrays (use_frequency, (*it)->get_appended_frequency(), label_count);
		// Increment use frequency of LHS if it is an accumulation
		use_frequency[label_bitset_index[lhs_label]] += 1;
		// Add the treenode, accnode pair to recomutations vector
		if (expr_has_leafs)
			computations.push_back (make_tuple (this, *it));
	}
	// Add the information in appended labels
	used_labels |= appended_labels;
	addArrays (use_frequency, appended_frequency, label_count);
	if (DEBUG) {
		printf ("In intra, the label use bitset for tree %s after recompute is ", lhs_label.c_str());
		print_bitset (label_bitset_index, used_labels, use_frequency, label_count);
	}
}

// Modify the out values of main tree after a subtree is added to its cluster
void treenode::update_host_tree (treenode *subtree, int label_count) {
	used_labels |= subtree->get_used_labels ();
	addArrays (use_frequency, subtree->get_use_frequency(), label_count);
}

/*
   We need to reorder all the children of the tree if 
   1. The subtrees share anything in common, and that value is not both live-in and live-out.
   2. Something became non-live in the output (i.e., some label's uses were exhausted in the tree)
   Otherwise we consider the children independently 
 */
tuple<int, int> treenode::compute_register_optimal_schedule (boost::dynamic_bitset<> &livein, unsigned int *livein_freq, map<string, int> label_bitset_index, unsigned int *label_frequency, int label_count, map<treenode*, vector<int>> &treenode_config, map<accnode*, vector<int>> &accnode_config) {
	// Handle data node
	if (is_data_node ()) return make_tuple (0, 0);
	// Handle leafs
	if (is_leaf_node ()) {
		int idx = label_bitset_index[lhs_label];
		return (livein[idx] == false ? make_tuple (1, 0) : make_tuple (0, 0));
	}
	// Otherwise compute live-out and dead-set
	//boost::dynamic_bitset<> dead_labels (label_count);
	boost::dynamic_bitset<> liveout (label_count);
	unsigned int *liveout_freq = new unsigned int[label_count] ();
	liveout = livein | used_labels;
	for (int i=0; i<label_count; i++) {
		liveout_freq[i] = livein_freq[i] + use_frequency[i];
		if (liveout[i] == true && liveout_freq[i] == label_frequency[i]) {
			liveout[i] = false; 
			//dead_labels[i] = true;
		}
	}
	// First check if there's already a memoized output available. To do so, first intersect in and out liveset
	// to find the labels that are live-in and live-out, and then filter out labels that are used in this tree
	boost::dynamic_bitset<> mem_key_a (label_count);
	mem_key_a = livein & ~((livein & liveout) & ~used_labels);
	boost::dynamic_bitset<> mem_key_b (label_count);
	mem_key_b = liveout & ~((livein & liveout) & ~used_labels);
	tuple<boost::dynamic_bitset<>,boost::dynamic_bitset<>> mem_key = make_tuple (mem_key_a, mem_key_b);
	if (optimal_reg_cache.find (mem_key) != optimal_reg_cache.end ()) {
		if (DEBUG) printf ("For tree with lhs %s, memoization returns (%u, %u)\n", lhs_label.c_str (), get<0>(optimal_reg_cache[mem_key]), get<1>(optimal_reg_cache[mem_key]));
		// Change livein and livein_freq going out of this node
		livein = liveout;
		copyArray (livein_freq, liveout_freq, label_count);
		delete[] liveout_freq;
		liveout.clear ();
		// Update treenode_config and accnode_config
		treenode_config.insert (get<3>(optimal_reg_cache[mem_key]).begin (), get<3>(optimal_reg_cache[mem_key]).end ());
		accnode_config.insert (get<4>(optimal_reg_cache[mem_key]).begin (), get<4>(optimal_reg_cache[mem_key]).end ());
		treenode_config[this] = get<2>(optimal_reg_cache[mem_key]); 
		return make_tuple (get<0>(optimal_reg_cache[mem_key]), get<1>(optimal_reg_cache[mem_key]));
	}
	vector<int> dependent_operands;
	vector<int> independent_operands;
	// Commenting this post PLDI because I think this is not required. Instead, adding the latter part 
	//// Separate the independent operands from dependent operands (operands that share some data input, or cause labels to become dead)
	//// 2. If some labels became dead, add the subtrees that use them to dependent operands
	//if (dead_labels.any ()) {
	//	boost::dynamic_bitset<> intersection;
	//	intersection.resize (label_count);
	//	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
	//		intersection = dead_labels & (*it)->get_used_labels ();
	//		if (intersection.any ())
	//			dependent_operands.push_back (it-rhs_accs.begin());
	//		else independent_operands.push_back (it-rhs_accs.begin());
	//	}
	//}
	//// If no labels became dead, see if some label became live in the subtrees, and add such subtrees to dependent set 
	//else {
	//	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
	//		bool shares_label = false;
	//		// 1. Check if the children share something in common
	//		boost::dynamic_bitset<> a_bitset = (*it)->get_used_labels ();
	//		for (vector<accnode*>::iterator jt=rhs_accs.begin(); jt!=rhs_accs.end(); jt++) {
	//			if (it == jt) continue;
	//			boost::dynamic_bitset<> intersection;
	//			intersection.resize (label_count);
	//			intersection = a_bitset & (*jt)->get_used_labels ();
	//			// Now check that the intersection is not live-in and live-out
	//			if (intersection.any ()) {
	//				if (((intersection & livein) != intersection) || ((intersection & liveout) != intersection)) {
	//					shares_label = true;
	//					break;
	//				}
	//			}
	//		}
	//		if (shares_label) 
	//			dependent_operands.push_back (it-rhs_accs.begin());
	//		else independent_operands.push_back (it-rhs_accs.begin());
	//	}
	//}
	// Added instead of the code above
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		bool shares_label = false;
		// 1. Check if the children share something in common
		boost::dynamic_bitset<> a_bitset = (*it)->get_used_labels ();
		boost::dynamic_bitset<> intersection (label_count);
		for (vector<accnode*>::iterator jt=rhs_accs.begin(); jt!=rhs_accs.end(); jt++) {
			if (it == jt) continue;
			intersection = a_bitset & (*jt)->get_used_labels ();
			// Now check that the intersection is not live-in and live-out
			if (intersection.any ()) {
				if (((intersection & livein) != intersection) || ((intersection & liveout) != intersection)) {
					shares_label = true;
					break;
				}
			}
		}
		if (shares_label) 
			dependent_operands.push_back (it-rhs_accs.begin());
		else independent_operands.push_back (it-rhs_accs.begin());
	}

	vector<tuple<int, int, vector<int>>> reg_count;
	map<treenode*, vector<int>> opt_local_treenode_config;
	map<accnode*, vector<int>> opt_local_accnode_config;
	// Step A: Traverse independent operands as it is
	for (vector<int>::iterator it=independent_operands.begin(); it!=independent_operands.end(); it++) {
		boost::dynamic_bitset<> t_livein (livein);
		unsigned int *t_livein_freq = new unsigned int[label_count] ();
		copyArray (t_livein_freq, livein_freq, label_count);
		vector<int> t_vec;
		t_vec.push_back (*it);
		tuple<int,int> t_reg_count = rhs_accs[*it]->compute_register_optimal_schedule (t_livein, t_livein_freq, label_bitset_index, label_frequency, label_count, opt_local_treenode_config, opt_local_accnode_config);
		reg_count.push_back (make_tuple (get<0>(t_reg_count), get<1>(t_reg_count), t_vec));
		delete[] t_livein_freq;
		t_livein.clear ();
	}
	if (dependent_operands.size () > 0) {
		// Step B. Traverse the dependence operands by trying all permutations
		vector<int> opt_dep_seq;
		map<treenode*, vector<int>> dep_treenode_config;
		map<accnode*, vector<int>> dep_accnode_config;
		int opt_reg_used = INT_MAX, opt_free_regs = 0, permutations = 0;
		do {
			vector<tuple<int, int>> t_reg_count;
			boost::dynamic_bitset<> t_livein (livein);
			unsigned int *t_livein_freq = new unsigned int[label_count] ();
			copyArray (t_livein_freq, livein_freq, label_count);
			map<treenode*, vector<int>> t_treenode_config;
			map<accnode*, vector<int>> t_accnode_config;
			// Optimize for this order
			for (vector<int>::iterator it=dependent_operands.begin(); it!=dependent_operands.end(); it++)
				t_reg_count.push_back (rhs_accs[*it]->compute_register_optimal_schedule (t_livein, t_livein_freq, label_bitset_index, label_frequency, label_count, t_treenode_config, t_accnode_config));
			delete[] t_livein_freq;
			t_livein.clear ();
			int reg_used = 0, free_regs = 0;
			for (vector<tuple<int, int>>::iterator jt=t_reg_count.begin(); jt!=t_reg_count.end(); jt++) {
				if (get<0>(*jt) >= free_regs) {
					reg_used += (get<0>(*jt) - free_regs);
					free_regs = get<1>(*jt);
				}
				else {
					free_regs -= get<0>(*jt);
					free_regs += get<1>(*jt);
				}
			}
			t_reg_count.clear ();
			if (reg_used < opt_reg_used) {
				opt_reg_used = reg_used;
				opt_free_regs = free_regs;
				opt_dep_seq.clear (); dep_treenode_config.clear (); dep_accnode_config.clear ();
				opt_dep_seq = dependent_operands;
				dep_treenode_config = t_treenode_config;
				dep_accnode_config = t_accnode_config;
			}
			t_treenode_config.clear ();
			t_accnode_config.clear ();
			permutations++;
		} while (next_permutation (dependent_operands.begin (), dependent_operands.end ()) && permutations<PERM_LIMIT);
		if (permutations == PERM_LIMIT) {
			permutations = 0;
			while (permutations < PERM_LIMIT) {
				random_shuffle (dependent_operands.begin (), dependent_operands.end ()); 
				vector<tuple<int, int>> t_reg_count;
				boost::dynamic_bitset<> t_livein (livein);
				unsigned int *t_livein_freq = new unsigned int[label_count] ();
				copyArray (t_livein_freq, livein_freq, label_count);
				map<treenode*, vector<int>> t_treenode_config;
				map<accnode*, vector<int>> t_accnode_config;
				// Optimize for this order
				for (vector<int>::iterator it=dependent_operands.begin(); it!=dependent_operands.end(); it++)
					t_reg_count.push_back (rhs_accs[*it]->compute_register_optimal_schedule (t_livein, t_livein_freq, label_bitset_index, label_frequency, label_count, t_treenode_config, t_accnode_config));
				delete[] t_livein_freq;
				t_livein.clear ();
				int reg_used = 0, free_regs = 0;
				for (vector<tuple<int, int>>::iterator jt=t_reg_count.begin(); jt!=t_reg_count.end(); jt++) {
					if (get<0>(*jt) >= free_regs) {
						reg_used += (get<0>(*jt) - free_regs);
						free_regs = get<1>(*jt);
					}
					else {
						free_regs -= get<0>(*jt);
						free_regs += get<1>(*jt);
					}
				}
				t_reg_count.clear ();
				if (reg_used < opt_reg_used) {
					opt_reg_used = reg_used;
					opt_free_regs = free_regs;
					opt_dep_seq.clear (); dep_treenode_config.clear (); dep_accnode_config.clear ();
					opt_dep_seq = dependent_operands;
					dep_treenode_config = t_treenode_config;
					dep_accnode_config = t_accnode_config;
				}
				t_treenode_config.clear ();
				t_accnode_config.clear ();
				permutations++;
			}
		}
		reg_count.push_back (make_tuple (opt_reg_used, opt_free_regs, opt_dep_seq));
		// Copy the optimal treenode_config and accnode_config in the local copies
		opt_local_treenode_config.insert (dep_treenode_config.begin (), dep_treenode_config.end ());
		opt_local_accnode_config.insert (dep_accnode_config.begin (), dep_accnode_config.end ());
		dep_treenode_config.clear ();
		dep_accnode_config.clear ();
	}
	// Insert the local optimal treenode_config and accnode_config into the incoming treenode_config and accnode_config
	treenode_config.insert (opt_local_treenode_config.begin(), opt_local_treenode_config.end ());
	accnode_config.insert (opt_local_accnode_config.begin(), opt_local_accnode_config.end ());
	// Sort the reg count based on their decreasing 2nd value (freed registers)
	if (reg_count.size () > 1) stable_sort (reg_count.begin(), reg_count.end(), sort_opt_reg_cost);
	int reg_used = 0, free_regs = 0;
	vector<int> exec_order;
	// If the tree is an accumulation node, reduce 1 from the first tuple for storing output
	int idx = label_bitset_index[lhs_label];
	bool lhs_assigned = (livein[idx] == true);
	bool lhs_live = (liveout[idx] == true);
	if (!lhs_assigned && is_accumulation_node ()) {
		for (vector<tuple<int, int, vector<int>>>::iterator kt=reg_count.begin(); kt!=next(reg_count.begin()); kt++) {
			if (get<1>(*kt) > 0)
				get<1>(*kt) -= 1;
			else
				get<0>(*kt) += 1;
		}
		lhs_assigned = true;
	}
	for (vector<tuple<int, int, vector<int>>>::iterator kt=reg_count.begin(); kt!=reg_count.end(); kt++) {
		if (get<0>(*kt) >= free_regs) {
			reg_used += (get<0>(*kt) - free_regs);
			free_regs = get<1>(*kt);
		}
		else {
			free_regs -= get<0>(*kt);
			free_regs += get<1>(*kt);
		}
		exec_order.insert (exec_order.end (), get<2>(*kt).begin (), get<2>(*kt).end ());
	}
	reg_count.clear ();
	// Add the register to store the output (lhs)
	if (!lhs_assigned && lhs_live) {
		if (free_regs > 0) free_regs--;
		else reg_used++;
	}
	// Change livein and livein_freq going out of this node
	livein = liveout;
	copyArray (livein_freq, liveout_freq, label_count);
	delete[] liveout_freq;
	liveout.clear ();
	// Make the entry in treenode_config
	treenode_config[this] = exec_order;
	// Make the entry in memoization map
	optimal_reg_cache[mem_key] = make_tuple (reg_used, free_regs, exec_order, opt_local_treenode_config, opt_local_accnode_config);
	if (DEBUG) printf ("For tree with lhs %s, returning (%u, %u)\n", lhs_label.c_str (), reg_used, free_regs);
	return make_tuple (reg_used, free_regs);
}

/*
   We need to reorder all the operands of an expression if 
   1. The subtrees share anything in common, and that value is not both live-in and live-out.
   2. Something became non-live in the output (i.e., some label's uses were exhausted in the tree)
   Otherwise we consider the children independently 
 */
tuple<int, int> accnode::compute_register_optimal_schedule (boost::dynamic_bitset<> &livein, unsigned int *livein_freq, map<string, int> label_bitset_index, unsigned int *label_frequency, int label_count, map<treenode*, vector<int>> &treenode_config, map<accnode*, vector<int>> &accnode_config) {
	// Compute live-out and dead-set
	//boost::dynamic_bitset<> dead_labels (label_count);
	boost::dynamic_bitset<> liveout (label_count);
	unsigned int *liveout_freq = new unsigned int[label_count] ();
	liveout = livein | used_labels;
	for (int i=0; i<label_count; i++) {
		liveout_freq[i] = livein_freq[i] + use_frequency[i];
		if (liveout[i] == true && liveout_freq[i] == label_frequency[i]) {
			liveout[i] = false; 
			//dead_labels[i] = true;
		}
	}

	// First check if there's already a memoized output available. To do so, first intersect in and out liveset
	// to find the labels that are live-in and live-out, and then filter out labels that are used in this tree
	boost::dynamic_bitset<> mem_key_a (label_count);
	mem_key_a = livein & ~((livein & liveout) & ~used_labels);
	boost::dynamic_bitset<> mem_key_b (label_count);
	mem_key_b = liveout & ~((livein & liveout) & ~used_labels);
	tuple<boost::dynamic_bitset<>,boost::dynamic_bitset<>> mem_key = make_tuple (mem_key_a, mem_key_b);
	if (optimal_reg_cache.find (mem_key) != optimal_reg_cache.end ()) {
		if (DEBUG) printf ("For subtree computing %s, memoization returns (%u, %u)\n", expr_string.c_str (), get<0>(optimal_reg_cache[mem_key]), get<1>(optimal_reg_cache[mem_key]));
		// Change livein and livein_freq
		livein = liveout;
		copyArray (livein_freq, liveout_freq, label_count);
		delete[] liveout_freq;
		liveout.clear ();
		// Update treenode_config and accnode_config
		treenode_config.insert (get<3>(optimal_reg_cache[mem_key]).begin (), get<3>(optimal_reg_cache[mem_key]).end ());
		accnode_config.insert (get<4>(optimal_reg_cache[mem_key]).begin (), get<4>(optimal_reg_cache[mem_key]).end ());
		accnode_config[this] = get<2>(optimal_reg_cache[mem_key]); 
		return make_tuple (get<0>(optimal_reg_cache[mem_key]), get<1>(optimal_reg_cache[mem_key]));
	}
	vector<int> dependent_trees;
	vector<int> independent_trees;
	// Commenting this post PLDI because I think this is not required. Instead, adding the latter part
	//// 2. If some labels became dead, add the subtrees that use them to dependent trees 
	//if (dead_labels.any ()) {
	//	boost::dynamic_bitset<> intersection;
	//	intersection.resize (label_count);
	//	for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
	//		intersection = dead_labels & (*it)->get_used_labels ();
	//		if (intersection.any ())
	//			dependent_trees.push_back (it-rhs_operands.begin());
	//		else
	//			independent_trees.push_back (it-rhs_operands.begin());
	//	}
	//}
	//// If no labels became dead, see if some label became live in the subtrees, and add such subtrees to dependent set 
	//else {
	//	for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
	//		bool shares_label = false;
	//		// 1. Check if the children share something in common
	//		boost::dynamic_bitset<> a_bitset = (*it)->get_used_labels (); 
	//		for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) {
	//			if (it == jt) continue;
	//			boost::dynamic_bitset<> intersection;
	//			intersection.resize (label_count);
	//			intersection = a_bitset & (*jt)->get_used_labels ();
	//			// Now check that the intersection is not live-in and live-out
	//			if (intersection.any ()) {
	//				if (((intersection & livein) != intersection) || ((intersection & liveout) != intersection)) {
	//					shares_label = true;
	//					break;
	//				}
	//			}
	//		}
	//		if (shares_label) 
	//			dependent_trees.push_back (it-rhs_operands.begin());
	//		else independent_trees.push_back (it-rhs_operands.begin());
	//	}
	//}
	// Added instead of the code above
	for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
		bool shares_label = false;
		// 1. Check if the children share something in common
		boost::dynamic_bitset<> a_bitset = (*it)->get_used_labels ();
		boost::dynamic_bitset<> intersection (label_count);
		for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) {
			if (it == jt) continue;
			intersection = a_bitset & (*jt)->get_used_labels ();
			// Now check that the intersection is not live-in and live-out
			if (intersection.any ()) {
				if (((intersection & livein) != intersection) || ((intersection & liveout) != intersection)) {
					shares_label = true;
					break;
				}
			}
		}
		if (shares_label)
			dependent_trees.push_back (it-rhs_operands.begin());
		else independent_trees.push_back (it-rhs_operands.begin());
		intersection.clear ();
	}

	vector<tuple<int, int, vector<int>>> reg_count;
	map<treenode*, vector<int>> opt_local_treenode_config;
	map<accnode*, vector<int>> opt_local_accnode_config;
	// Step A: Traverse independent trees as it is
	for (vector<int>::iterator it=independent_trees.begin(); it!=independent_trees.end(); it++) {
		boost::dynamic_bitset<> t_livein (livein);
		unsigned int *t_livein_freq = new unsigned int[label_count] ();
		copyArray (t_livein_freq, livein_freq, label_count);
		vector<int> t_vec;
		t_vec.push_back (*it);
		tuple<int,int> t_reg_count = rhs_operands[*it]->compute_register_optimal_schedule (t_livein, t_livein_freq, label_bitset_index, label_frequency, label_count, opt_local_treenode_config, opt_local_accnode_config);
		reg_count.push_back (make_tuple (get<0>(t_reg_count), get<1>(t_reg_count), t_vec));
		delete[] t_livein_freq;
		t_livein.clear ();
	}
	// Step B. Traverse the dependence trees by trying all permutations
	if (dependent_trees.size () > 0) {
		vector<int> opt_dep_seq;
		map<treenode*, vector<int>> dep_treenode_config;
		map<accnode*, vector<int>> dep_accnode_config;
		int opt_reg_used = INT_MAX, opt_free_regs = 0, permutations = 0;
		do {
			vector<tuple<int, int>> t_reg_count;
			boost::dynamic_bitset<> t_livein (livein);
			unsigned int *t_livein_freq = new unsigned int[label_count] ();
			copyArray (t_livein_freq, livein_freq, label_count);
			map<treenode*, vector<int>> t_treenode_config;
			map<accnode*, vector<int>> t_accnode_config;
			// Optimize for this order
			for (vector<int>::iterator it=dependent_trees.begin(); it!=dependent_trees.end(); it++)
				t_reg_count.push_back (rhs_operands[*it]->compute_register_optimal_schedule (t_livein, t_livein_freq, label_bitset_index, label_frequency, label_count, t_treenode_config, t_accnode_config));
			delete[] t_livein_freq;
			t_livein.clear ();
			int reg_used = 0, free_regs = 0;
			for (vector<tuple<int, int>>::iterator jt=t_reg_count.begin(); jt!=t_reg_count.end(); jt++) {
				if (get<0>(*jt) >= free_regs) {
					reg_used += (get<0>(*jt) - free_regs);
					free_regs = get<1>(*jt);
				}
				else {
					free_regs -= get<0>(*jt);
					free_regs += get<1>(*jt);
				}
			}
			t_reg_count.clear ();
			if (reg_used < opt_reg_used) {
				opt_reg_used = reg_used;
				opt_free_regs = free_regs;
				opt_dep_seq.clear (); dep_treenode_config.clear (); dep_accnode_config.clear ();
				opt_dep_seq = dependent_trees;
				dep_treenode_config = t_treenode_config;
				dep_accnode_config = t_accnode_config;
			}
			t_treenode_config.clear ();
			t_accnode_config.clear ();
			permutations++;
		} while (next_permutation (dependent_trees.begin (), dependent_trees.end ()) && permutations<PERM_LIMIT);
		// If the permuatations were stopped due to permutations reaching PERM_LIMIT, try random shuffles
		if (permutations == PERM_LIMIT) {
			permutations = 0;
			while (permutations < PERM_LIMIT) {
				random_shuffle (dependent_trees.begin (), dependent_trees.end ());
				vector<tuple<int, int>> t_reg_count;
				boost::dynamic_bitset<> t_livein (livein);
				unsigned int *t_livein_freq = new unsigned int[label_count] ();
				copyArray (t_livein_freq, livein_freq, label_count);
				map<treenode*, vector<int>> t_treenode_config;
				map<accnode*, vector<int>> t_accnode_config;
				// Optimize for this order
				for (vector<int>::iterator it=dependent_trees.begin(); it!=dependent_trees.end(); it++)
					t_reg_count.push_back (rhs_operands[*it]->compute_register_optimal_schedule (t_livein, t_livein_freq, label_bitset_index, label_frequency, label_count, t_treenode_config, t_accnode_config));
				delete[] t_livein_freq;
				t_livein.clear ();
				int reg_used = 0, free_regs = 0;
				for (vector<tuple<int, int>>::iterator jt=t_reg_count.begin(); jt!=t_reg_count.end(); jt++) {
					if (get<0>(*jt) >= free_regs) {
						reg_used += (get<0>(*jt) - free_regs);
						free_regs = get<1>(*jt);
					}
					else {
						free_regs -= get<0>(*jt);
						free_regs += get<1>(*jt);
					}
				}
				t_reg_count.clear ();
				if (reg_used < opt_reg_used) {
					opt_reg_used = reg_used;
					opt_free_regs = free_regs;
					opt_dep_seq.clear (); dep_treenode_config.clear (); dep_accnode_config.clear ();	
					opt_dep_seq = dependent_trees;
					dep_treenode_config = t_treenode_config;
					dep_accnode_config = t_accnode_config;
				}
				t_treenode_config.clear ();
				t_accnode_config.clear ();
				permutations++;
			}
		}
		reg_count.push_back (make_tuple (opt_reg_used, opt_free_regs, opt_dep_seq));
		// Copy the optimal treenode_config and accnode_config in the local copies
		opt_local_treenode_config.insert (dep_treenode_config.begin (), dep_treenode_config.end ());
		opt_local_accnode_config.insert (dep_accnode_config.begin (), dep_accnode_config.end ());
		dep_treenode_config.clear ();
		dep_accnode_config.clear ();
	}
	// Insert the local optimal treenode_config and accnode_config into the incoming treenode_config and accnode_config
	treenode_config.insert (opt_local_treenode_config.begin(), opt_local_treenode_config.end ());
	accnode_config.insert (opt_local_accnode_config.begin(), opt_local_accnode_config.end ());
	// Sort the reg count based on their decreasing 2nd value (freed registers)
	if (reg_count.size () > 1) stable_sort (reg_count.begin(), reg_count.end(), sort_opt_reg_cost);
	int reg_used = 0, free_regs = 0;
	vector<int> exec_order;
	for (vector<tuple<int, int, vector<int>>>::iterator kt=reg_count.begin(); kt!=reg_count.end(); kt++) {
		if (get<0>(*kt) >= free_regs) {
			reg_used += (get<0>(*kt) - free_regs);
			free_regs = get<1>(*kt);
		}
		else {
			free_regs -= get<0>(*kt);
			free_regs += get<1>(*kt);
		}
		exec_order.insert (exec_order.end (), get<2>(*kt).begin (), get<2>(*kt).end ());
	}
	reg_count.clear ();
	// Increment the free count to reflect the lhs_labels that won't be alive after after this computation
	for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
		if ((*it)->is_data_node ()) continue;
		string lhs_label = (*it)->get_lhs_label ();
		int idx = label_bitset_index[lhs_label];
		if (liveout[idx] == false)
			free_regs++;
	}
	// Change livein and livein_freq
	livein = liveout;
	copyArray (livein_freq, liveout_freq, label_count);
	delete[] liveout_freq;
	liveout.clear ();
	// Make the entry in accnode_config
	accnode_config[this] = exec_order;
	// Make the entry in memoization map
	optimal_reg_cache[mem_key] = make_tuple (reg_used, free_regs, exec_order, opt_local_treenode_config, opt_local_accnode_config);	
	if (DEBUG) printf ("For subtree computing %s, returning (%u, %u)\n", expr_string.c_str (), reg_used, free_regs);
	return make_tuple (reg_used, free_regs);
}

// Reorient the nodes after optimal schedule is computed
void treenode::retrace_register_optimal_schedule (map<treenode*, vector<int>> &treenode_config, map<accnode*, vector<int>> &accnode_config) {
	// No need to reorient if the node is a data or leaf node
	if (is_data_node () || is_leaf_node ()) return;
	// Otherwise first recursively reorient the trees rooted at rhs_accs, 
	// and then reorient the nodes by finding their order from treenode_config
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) 
		(*it)->retrace_register_optimal_schedule (treenode_config, accnode_config);
	if (DEBUG) assert (treenode_config.find (this) != treenode_config.end () && "Could not retrace treenode (retrace_register_optimal_schedule)");
	vector<int> exec_order = treenode_config[this];
	vector<accnode*> rearranged_rhs_accs;
	for (vector<int>::iterator it=exec_order.begin(); it!=exec_order.end(); it++)
		rearranged_rhs_accs.push_back (rhs_accs[*it]);
	reset_rhs_accs (rearranged_rhs_accs);
}

// Reorient the nodes after optimal schedule is computed
void accnode::retrace_register_optimal_schedule (map<treenode*, vector<int>> &treenode_config, map<accnode*, vector<int>> &accnode_config) {
	// First recursively reorient the trees rooted at rhs_operands
	for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++)
		(*it)->retrace_register_optimal_schedule (treenode_config, accnode_config);
	// Now reorient the rhs operands
	if (DEBUG) assert (accnode_config.find (this) != accnode_config.end () && "Could not retrace accnode (retrace_register_optimal_schedule)");
	vector<int> exec_order = accnode_config[this];
	vector<treenode*> rearranged_rhs_operands;
	for (vector<int>::iterator it=exec_order.begin(); it!=exec_order.end(); it++)
		rearranged_rhs_operands.push_back (rhs_operands[*it]);
	reset_rhs_operand (rearranged_rhs_operands);
}

///*
//   We need to reorder all the children of the tree if 
//   1. The subtrees share anything in common, and that value is not both live-in and live-out.
//   2. Something became non-live in the output (i.e., some label's uses were exhausted in the tree)
//   Otherwise we consider the children independently 
// */
//tuple<int, int> treenode::retrace_register_optimal_schedule (boost::dynamic_bitset<> &livein, unsigned int *livein_freq, map<string, int> label_bitset_index, unsigned int *label_frequency, int label_count) {
//	vector<tuple<int, int, vector<int>>> reg_count;
//	// Handle data node
//	if (is_data_node ()) return make_tuple (0, 0);
//	// Handle leafs
//	if (is_leaf_node ()) {
//		int idx = label_bitset_index[lhs_label];
//		return (livein[idx] == false ? make_tuple (1, 0) : make_tuple (0, 0));
//	}
//	// Otherwise compute live-out and dead-set
//	boost::dynamic_bitset<> dead_labels;
//	dead_labels.resize (label_count);
//	boost::dynamic_bitset<> liveout;
//	liveout.resize (label_count);
//	unsigned int *liveout_freq = new unsigned int[label_count] ();
//	liveout = livein | get_used_labels ();
//	unsigned int *t_freq = get_use_frequency ();
//	for (int i=0; i<label_count; i++) {
//		liveout_freq[i] = livein_freq[i] + t_freq[i];
//		if (liveout[i] == true && liveout_freq[i] == label_frequency[i]) {
//			liveout[i] = false; 
//			dead_labels[i] = true;
//		}
//	}
//	if (DEBUG) {
//		printf ("Inside retrace, use frequency for tree rooted at %s is ", lhs_label.c_str());
//		print_frequency (label_bitset_index, get_use_frequency (), label_count);	
//		printf ("Inside retrace, liveout for tree rooted at %s is ", lhs_label.c_str());
//		print_bitset (label_bitset_index, liveout, label_count);
//	}
//	// First check if there's already a memoized output available. To do so, first intersect in and out liveset
//	// to find the labels that are live-in and live-out, and then filter out labels that are used in this tree
//	boost::dynamic_bitset<> mem_key_a;
//	mem_key_a.resize (label_count);
//	mem_key_a = livein & ~((livein & liveout) & ~(get_used_labels()));
//	boost::dynamic_bitset<> mem_key_b;
//	mem_key_b.resize (label_count);
//	mem_key_b = liveout & ~((livein & liveout) & ~(get_used_labels()));
//	tuple<boost::dynamic_bitset<>,boost::dynamic_bitset<>> mem_key = make_tuple (mem_key_a, mem_key_b);
//	if (retrace_reg_cache.find (mem_key) != retrace_reg_cache.end ()) {
//		if (DEBUG) printf ("For tree with lhs %s, retracing memoization returns (%u, %u)\n", lhs_label.c_str (), get<0>(retrace_reg_cache[mem_key]), get<1>(retrace_reg_cache[mem_key]));
//		// Change livein and livein_freq going out of this node
//		livein = liveout;
//		for (int i=0; i<label_count; i++)
//			livein_freq[i] = liveout_freq[i];
//		// Set the execution order
//		vector<int> execution_order = get<2>(retrace_reg_cache[mem_key]);
//		vector<accnode*> rearranged_rhs_accs;
//		for (vector<int>::iterator it=execution_order.begin(); it!=execution_order.end(); it++)
//			rearranged_rhs_accs.push_back (rhs_accs[*it]);
//		reset_rhs_accs (rearranged_rhs_accs);
//		return make_tuple (get<0>(retrace_reg_cache[mem_key]), get<1>(retrace_reg_cache[mem_key]));
//	}
//
//	vector<int> dependent_operands;
//	vector<int> independent_operands;
//	// Separate the independent operands from dependent operands (operands that share some data input, or cause labels to become dead)
//	// 2. If some labels became dead, add the subtrees that use them to dependent operands
//	if (dead_labels.any ()) {
//		boost::dynamic_bitset<> intersection;
//		intersection.resize (label_count);
//		for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
//			intersection = dead_labels & (*it)->get_used_labels ();
//			if (intersection.any ())
//				dependent_operands.push_back (it-rhs_accs.begin());
//			else independent_operands.push_back (it-rhs_accs.begin());
//		}
//	}
//	// If no labels became dead, see if some label became live in the subtrees, and add such subtrees to dependent set 
//	else {
//		for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
//			bool shares_label = false;
//			// 1. Check if the children share something in common
//			boost::dynamic_bitset<> a_bitset = (*it)->get_used_labels ();
//			for (vector<accnode*>::iterator jt=rhs_accs.begin(); jt!=rhs_accs.end(); jt++) {
//				if (it == jt) continue;
//				boost::dynamic_bitset<> intersection;
//				intersection.resize (label_count);
//				intersection = a_bitset & (*jt)->get_used_labels ();
//				// Now check that the intersection is not live-in and live-out
//				if (intersection.any ()) {
//					if (((intersection & livein) != intersection) || ((intersection & liveout) != intersection)) {
//						shares_label = true;
//						break;
//					}
//				}
//			}
//			if (shares_label) 
//				dependent_operands.push_back (it-rhs_accs.begin());
//			else independent_operands.push_back (it-rhs_accs.begin());
//		}
//	}
//	// Step A: Traverse independent operands as it is
//	for (vector<int>::iterator it=independent_operands.begin(); it!=independent_operands.end(); it++) {
//		boost::dynamic_bitset<> t_livein (livein);
//		unsigned int *t_livein_freq = new unsigned int[label_count] ();
//		for (int i=0; i<label_count; i++)
//			t_livein_freq[i] = livein_freq[i];
//		// Create temporary placeholder
//		vector<int> t_operand;
//		t_operand.push_back (*it);
//		tuple<int,int> opt_reg_count = rhs_accs[*it]->retrace_register_optimal_schedule (t_livein, t_livein_freq, label_bitset_index, label_frequency, label_count);
//		reg_count.push_back (make_tuple (get<0>(opt_reg_count), get<1>(opt_reg_count), t_operand));
//	}
//	// Step B. Traverse the dependence operands by trying all permutations
//	vector<int> opt_op_seq;
//	if (dependent_operands.size () > 0) {
//		int opt_reg_used = INT_MAX, opt_free_regs = 0;
//		int permutations = 0;
//		do {
//			vector<tuple<int, int>> t_reg_count;
//			boost::dynamic_bitset<> t_livein;
//			t_livein.resize (label_count);
//			t_livein = livein;
//			unsigned int *t_livein_freq = new unsigned int[label_count] ();
//			for (int i=0; i<label_count; i++) 
//				t_livein_freq[i] = livein_freq[i];
//			// Optimize for this order
//			for (vector<int>::iterator it=dependent_operands.begin(); it!=dependent_operands.end(); it++)
//				t_reg_count.push_back (rhs_accs[*it]->retrace_register_optimal_schedule (t_livein, t_livein_freq, label_bitset_index, label_frequency, label_count));
//			int reg_used = 0, free_regs = 0;
//			for (vector<tuple<int, int>>::iterator jt=t_reg_count.begin(); jt!=t_reg_count.end(); jt++) {
//				if (get<0>(*jt) >= free_regs) {
//					reg_used += (get<0>(*jt) - free_regs);
//					free_regs = get<1>(*jt);
//				}
//				else {
//					free_regs -= get<0>(*jt);
//					free_regs += get<1>(*jt);
//				}
//			}
//			if (reg_used < opt_reg_used) {
//				opt_reg_used = reg_used;
//				opt_free_regs = free_regs;
//				opt_op_seq = dependent_operands;
//			}
//			permutations++;
//		} while (next_permutation (dependent_operands.begin (), dependent_operands.end ()) && permutations<PERM_LIMIT);
//		reg_count.push_back (make_tuple (opt_reg_used, opt_free_regs, opt_op_seq));  
//	}
//	// Sort the reg count based on their decreasing 2nd value (freed registers)
//	if (reg_count.size () > 1) stable_sort (reg_count.begin(), reg_count.end(), sort_opt_reg_cost);
//	int reg_used = 0, free_regs = 0;
//	vector<int> execution_order;
//	for (vector<tuple<int, int, vector<int>>>::iterator kt=reg_count.begin(); kt!=reg_count.end(); kt++) {
//		if (get<0>(*kt) >= free_regs) {
//			reg_used += (get<0>(*kt) - free_regs);
//			free_regs = get<1>(*kt);
//		}
//		else {
//			free_regs -= get<0>(*kt);
//			free_regs += get<1>(*kt);
//		}
//		execution_order.insert (execution_order.end(), get<2>(*kt).begin(), get<2>(*kt).end());
//	}
//	// Add the register to store the output
//	if (free_regs > 0) free_regs--;
//	else reg_used++;
//	// Change livein and livein_freq going out of this node
//	livein = liveout;
//	for (int i=0; i<label_count; i++)
//		livein_freq[i] = liveout_freq[i];
//	// Set the execution order
//	vector<accnode *> new_rhs_accs;
//	for (vector<int>::iterator it=execution_order.begin(); it!=execution_order.end(); it++) 
//		new_rhs_accs.push_back (rhs_accs[*it]);
//	reset_rhs_accs (new_rhs_accs);
//	// Make the entry in memoization map
//	retrace_reg_cache[mem_key] = make_tuple (reg_used, free_regs, execution_order);	
//	if (DEBUG) printf ("Retracing tree with lhs %s, returns (%u, %u)\n", lhs_label.c_str (), reg_used, free_regs);
//	return make_tuple (reg_used, free_regs);
//}
//
///*
//   We need to reorder all the operands of an expression if 
//   1. The subtrees share anything in common, and that value is not both live-in and live-out.
//   2. Something became non-live in the output (i.e., some label's uses were exhausted in the tree)
//   Otherwise we consider the children independently 
// */
//tuple<int, int> accnode::retrace_register_optimal_schedule (boost::dynamic_bitset<> &livein, unsigned int *livein_freq, map<string, int> label_bitset_index, unsigned int *label_frequency, int label_count) {
//	vector<tuple<int, int, vector<int>>> reg_count;
//	// Compute live-out and dead-set
//	boost::dynamic_bitset<> dead_labels;
//	dead_labels.resize (label_count);
//	boost::dynamic_bitset<> liveout;
//	liveout.resize (label_count);
//	unsigned int *liveout_freq = new unsigned int[label_count] ();
//	liveout = livein | get_used_labels ();
//	unsigned int *t_freq = get_use_frequency ();
//	for (int i=0; i<label_count; i++) {
//		liveout_freq[i] = livein_freq[i] + t_freq[i];
//		if (liveout[i] == true && liveout_freq[i] == label_frequency[i]) {
//			liveout[i] = false; dead_labels[i] = true;
//		}
//	}
//
//	// First check if there's already a memoized output available. To do so, first intersect in and out liveset
//	// to find the labels that are live-in and live-out, and then filter out labels that are used in this tree
//	boost::dynamic_bitset<> mem_key_a;
//	mem_key_a.resize (label_count);
//	mem_key_a = livein & ~((livein & liveout) & ~(get_used_labels()));
//	boost::dynamic_bitset<> mem_key_b;
//	mem_key_b.resize (label_count);
//	mem_key_b = liveout & ~((livein & liveout) & ~(get_used_labels()));
//	tuple<boost::dynamic_bitset<>,boost::dynamic_bitset<>> mem_key = make_tuple (mem_key_a, mem_key_b);
//	if (retrace_reg_cache.find (mem_key) != retrace_reg_cache.end ()) {
//		if (DEBUG) printf ("For subtree computing %s, retracing memoization returns (%u, %u)\n", expr_string.c_str (), get<0>(retrace_reg_cache[mem_key]), get<1>(retrace_reg_cache[mem_key]));
//		// Change livein and livein_freq
//		livein = liveout;
//		for (int i=0; i<label_count; i++) 
//			livein_freq[i] = liveout_freq[i];
//		// Set the execution order
//		int seq_no = 0;
//		vector<int> execution_order = get<2>(retrace_reg_cache[mem_key]);
//		vector<treenode *> new_rhs_operands;
//		for (vector<int>::iterator it=execution_order.begin(); it!=execution_order.end(); it++,seq_no++) 
//			new_rhs_operands.push_back (rhs_operands[*it]);
//		reset_rhs_operand (new_rhs_operands);
//		return make_tuple (get<0>(retrace_reg_cache[mem_key]), get<1>(retrace_reg_cache[mem_key]));
//	}
//	vector<int> dependent_trees;
//	vector<int> independent_trees;
//	// 2. If some labels became dead, add the subtrees that use them to dependent trees 
//	if (dead_labels.any ()) {
//		boost::dynamic_bitset<> intersection;
//		intersection.resize (label_count);
//		for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
//			intersection = dead_labels & (*it)->get_used_labels ();
//			if (intersection.any ())
//				dependent_trees.push_back (it-rhs_operands.begin());
//			else
//				independent_trees.push_back (it-rhs_operands.begin());
//		}
//	}
//	// If no labels became dead, see if some label became live in the subtrees, and add such subtrees to dependent set 
//	else {
//		for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
//			bool shares_label = false;
//			// 1. Check if the children share something in common
//			boost::dynamic_bitset<> a_bitset = (*it)->get_used_labels (); 
//			for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) {
//				if (it == jt) continue;
//				boost::dynamic_bitset<> intersection;
//				intersection.resize (label_count);
//				intersection = a_bitset & (*jt)->get_used_labels ();
//				// Now check that the intersection is not live-in and live-out
//				if (intersection.any ()) {
//					if (((intersection & livein) != intersection) || ((intersection & liveout) != intersection)) {
//						shares_label = true;
//						break;
//					}
//				}
//			}
//			if (shares_label) 
//				dependent_trees.push_back (it-rhs_operands.begin());
//			else independent_trees.push_back (it-rhs_operands.begin());
//		}
//	}
//	// Step A: Traverse independent trees as it is
//	for (vector<int>::iterator it=independent_trees.begin(); it!=independent_trees.end(); it++) {
//		boost::dynamic_bitset<> t_livein;
//		t_livein.resize (label_count);
//		t_livein = livein;
//		unsigned int *t_livein_freq = new unsigned int[label_count] ();
//		for (int i=0; i<label_count; i++)
//			t_livein_freq[i] = livein_freq[i];
//		// Create temporary placeholder
//		vector<int> t_operand;
//		t_operand.push_back (*it);
//		tuple<int,int> opt_reg_count = rhs_operands[*it]->retrace_register_optimal_schedule (t_livein, t_livein_freq, label_bitset_index, label_frequency, label_count);
//		reg_count.push_back (make_tuple (get<0>(opt_reg_count), get<1>(opt_reg_count), t_operand));
//	}
//	// Step B. Traverse the dependence trees by trying all permutations
//	vector<int> opt_op_seq;
//	if (dependent_trees.size () > 0) {
//		int opt_reg_used = INT_MAX, opt_free_regs = 0;
//		int permutations = 0;
//		do {
//			vector<tuple<int, int>> t_reg_count;
//			boost::dynamic_bitset<> t_livein;
//			t_livein.resize (label_count);
//			t_livein = livein;
//			unsigned int *t_livein_freq = new unsigned int[label_count] ();
//			for (int i=0; i<label_count; i++) 
//				t_livein_freq[i] = livein_freq[i]; 
//			// Optimize for this order
//			for (vector<int>::iterator it=dependent_trees.begin(); it!=dependent_trees.end(); it++)
//				t_reg_count.push_back (rhs_operands[*it]->retrace_register_optimal_schedule (t_livein, t_livein_freq, label_bitset_index, label_frequency, label_count));
//			int reg_used = 0, free_regs = 0;
//			for (vector<tuple<int, int>>::iterator jt=t_reg_count.begin(); jt!=t_reg_count.end(); jt++) {
//				if (get<0>(*jt) >= free_regs) {
//					reg_used += (get<0>(*jt) - free_regs);
//					free_regs = get<1>(*jt);
//				}
//				else {
//					free_regs -= get<0>(*jt);
//					free_regs += get<1>(*jt);
//				}
//			}
//			if (reg_used < opt_reg_used) {
//				opt_reg_used = reg_used;
//				opt_free_regs = free_regs;
//				opt_op_seq = dependent_trees;
//			}
//			permutations++;
//		} while (next_permutation (dependent_trees.begin (), dependent_trees.end ()) && permutations<PERM_LIMIT);
//		reg_count.push_back (make_tuple (opt_reg_used, opt_free_regs, opt_op_seq));
//	}
//	// Sort the reg count based on their decreasing 2nd value (freed registers)
//	if (reg_count.size () > 1) stable_sort (reg_count.begin(), reg_count.end(), sort_opt_reg_cost);
//	int reg_used = 0, free_regs = 0;
//	vector<int> execution_order;
//	for (vector<tuple<int, int, vector<int>>>::iterator kt=reg_count.begin(); kt!=reg_count.end(); kt++) {
//		if (get<0>(*kt) >= free_regs) {
//			reg_used += (get<0>(*kt) - free_regs);
//			free_regs = get<1>(*kt);
//		}
//		else {
//			free_regs -= get<0>(*kt);
//			free_regs += get<1>(*kt);
//		}
//		execution_order.insert (execution_order.end(), get<2>(*kt).begin(), get<2>(*kt).end());
//	}
//	// Increment the free count to reflect the lhs_labels that won't be alive after after this computation
//	for (vector<treenode*>::iterator it=rhs_operands.begin(); it!=rhs_operands.end(); it++) {
//		if ((*it)->is_data_node ()) continue;
//		string lhs_label = (*it)->get_lhs_label ();
//		int idx = label_bitset_index[lhs_label];
//		if (liveout[idx] == false)
//			free_regs++;
//	}
//	// Change livein and livein_freq
//	livein = liveout;
//	for (int i=0; i<label_count; i++) 
//		livein_freq[i] = liveout_freq[i];
//	// Set the execution order
//	vector<treenode*> new_rhs_operands;
//	for (vector<int>::iterator it=execution_order.begin(); it!=execution_order.end(); it++) 
//		new_rhs_operands.push_back (rhs_operands[*it]);
//	reset_rhs_operand (new_rhs_operands);
//	// Make the entry in memoization map
//	retrace_reg_cache[mem_key] = make_tuple (reg_used, free_regs, execution_order);
//	if (DEBUG) printf ("Retracing subtree computing %s, returns (%u, %u)\n", expr_string.c_str (), reg_used, free_regs);
//	return make_tuple (reg_used, free_regs);
//}

//Filter out the subtrees that use leaves belonging to one of the following category: 
//   a. The leaves in the intersection_leafset
//   b. Other leaves that have only one use
// ONLY USE THIS IF YOU WANT TO REMOVE TEMP_LABELS
void treenode::identify_optimizable_subtrees (vector<tuple<treenode*,accnode*>> &subtrees, boost::dynamic_bitset<> intersection_leafset, boost::dynamic_bitset<> single_use_labels, unsigned int *label_frequency, int label_count) {
	// Iterate over accumulations 
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		boost::dynamic_bitset<> expr_labels = (*it)->get_used_labels ();
		unsigned int *u_freq = (*it)->get_use_frequency ();
		boost::dynamic_bitset<> temp_labels (label_count);
		for (int i=0; i<label_count; i++) {
			if (u_freq[i] == label_frequency[i]) temp_labels[i] = true;
		}
		bool suitable_subtree = true;
		// 1. Check that the intersection with intersection_leafset is non-null
		boost::dynamic_bitset<> check_suitability (label_count);
		check_suitability = intersection_leafset & expr_labels;
		if (check_suitability.none ())
			suitable_subtree = false;
		if (suitable_subtree) {
			// 2. Check if it uses any other labels than those in intersection_leaflet. If it does, they must belong to single_use_labels
			check_suitability = expr_labels & (intersection_leafset | single_use_labels | temp_labels);
			if (check_suitability != expr_labels)
				suitable_subtree = false;
		}
		check_suitability.clear (); temp_labels.clear ();
		// If the tree is suitable, add it to ret_subtrees
		if (suitable_subtree) {
			subtrees.push_back (make_tuple (this, *it));
			if (DEBUG) {
				printf ("found suitable subtree with %s = %s\n", get_lhs_label ().c_str(), (*it)->get_expr_string().c_str());
				printf ("label use " ); cout << expr_labels << endl; 
				printf ("intersection "); cout << intersection_leafset << endl;
				printf ("extra labels "); cout << single_use_labels << endl;
			}
		}
		// Otherwise recursively call the function
		else {
			vector<treenode*> &rhs_operands = (*it)->get_operands ();
			// iterate over the subtrees
			for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) 
				(*jt)->identify_optimizable_subtrees (subtrees, intersection_leafset, single_use_labels, label_frequency, label_count);
		}
	}
}

// same as above, but forces the subtrees to have atleast one leaf node
void treenode::identify_optimizable_subtrees (vector<tuple<treenode*,accnode*>> &subtrees, boost::dynamic_bitset<> intersection_leafset, boost::dynamic_bitset<> single_use_labels, boost::dynamic_bitset<> leaf_labels, unsigned int *label_frequency, int label_count) {
	// Iterate over accumulations 
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		boost::dynamic_bitset<> expr_labels = (*it)->get_used_labels ();
		unsigned int *u_freq = (*it)->get_use_frequency ();
		boost::dynamic_bitset<> temp_labels (label_count);
		for (int i=0; i<label_count; i++) {
			if (u_freq[i] == label_frequency[i]) temp_labels[i] = true;
		}
		bool suitable_subtree = true;
		boost::dynamic_bitset<> check_suitability (label_count);
		// 0. Check that the subtree rooted at it has atleast one leaf node
		check_suitability = leaf_labels & expr_labels;
		if (check_suitability.none ()) 
			suitable_subtree = false; 
		// 1. Check that the intersection with intersection_leafset is non-null
		if (suitable_subtree) {
			check_suitability = intersection_leafset & expr_labels;
			if (check_suitability.none ())
				suitable_subtree = false;
		}
		if (suitable_subtree) {
			// 2. Check if it uses any other labels than those in intersection_leaflet. If it does, they must belong to single_use_labels
			check_suitability = expr_labels & (intersection_leafset | single_use_labels | temp_labels);
			if (check_suitability != expr_labels)
				suitable_subtree = false;
		}
		check_suitability.clear (); temp_labels.clear ();
		// If the tree is suitable, add it to ret_subtrees
		if (suitable_subtree) {
			subtrees.push_back (make_tuple (this, *it));
			if (DEBUG) {
				printf ("found suitable subtree with %s = %s\n", get_lhs_label ().c_str(), (*it)->get_expr_string().c_str());
				printf ("label use " ); cout << expr_labels << endl; 
				printf ("intersection "); cout << intersection_leafset << endl;
				printf ("leaf labels "); cout << leaf_labels << endl;	
				printf ("extra labels "); cout << single_use_labels << endl;
			}
		}
		// Otherwise recursively call the function
		else {
			vector<treenode*> &rhs_operands = (*it)->get_operands ();
			// iterate over the subtrees
			for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) 
				(*jt)->identify_optimizable_subtrees (subtrees, intersection_leafset, single_use_labels, leaf_labels, label_frequency, label_count);
		}
	}
}

//Filter out the subtrees that use leaves belonging to one of the following category: 
//   a. The leaves in the intersection_leafset
//   b. Other leaves that have only one use
void treenode::identify_optimizable_subtrees (vector<tuple<treenode*,accnode*>> &subtrees, boost::dynamic_bitset<> intersection_leafset, boost::dynamic_bitset<> single_use_labels, int label_count) {
	// Iterate over accumulations 
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		boost::dynamic_bitset<> expr_labels = (*it)->get_used_labels ();
		bool suitable_subtree = true;
		// 1. Check that the intersection with intersection_leafset is non-null
		boost::dynamic_bitset<> check_suitability (label_count);
		check_suitability = intersection_leafset & expr_labels;
		if (check_suitability.none ())
			suitable_subtree = false;
		if (suitable_subtree) {
			// 2. Check if it uses any other labels than those in intersection_leaflet. If it does, they must belong to single_use_labels
			check_suitability = expr_labels & (intersection_leafset | single_use_labels);
			if (check_suitability != expr_labels) 
				suitable_subtree = false;
		}
		check_suitability.clear (); 
		// If the tree is suitable, add it to ret_subtrees
		if (suitable_subtree) {
			subtrees.push_back (make_tuple (this, *it));
			if (DEBUG) {
				printf ("found suitable subtree with %s = %s\n", get_lhs_label ().c_str(), (*it)->get_expr_string().c_str());
				printf ("label use " ); cout << expr_labels << endl; 
				printf ("intersection "); cout << intersection_leafset << endl;
				printf ("extra labels "); cout << single_use_labels << endl;
			}
		}
		// Otherwise recursively call the function
		else {
			vector<treenode*> &rhs_operands = (*it)->get_operands ();
			// iterate over the subtrees
			for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) 
				(*jt)->identify_optimizable_subtrees (subtrees, intersection_leafset, single_use_labels, label_count);
		}
	}
}

// Same as above, except the fact that the chosen subtree must have atleast one leaf node
void treenode::identify_optimizable_subtrees (vector<tuple<treenode*,accnode*>> &subtrees, boost::dynamic_bitset<> intersection_leafset, boost::dynamic_bitset<> single_use_labels, boost::dynamic_bitset<> leaf_labels, int label_count) {
	// Iterate over accumulations 
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		boost::dynamic_bitset<> expr_labels = (*it)->get_used_labels ();
		bool suitable_subtree = true;
		boost::dynamic_bitset<> check_suitability (label_count);
		// 0. Check that the subtree rooted at it has atleast one leaf node
		check_suitability = leaf_labels & expr_labels;
		if (check_suitability.none ()) 
			suitable_subtree = false; 
		// 1. Check that the intersection with intersection_leafset is non-null
		if (suitable_subtree) {
			check_suitability = intersection_leafset & expr_labels;
			if (check_suitability.none ())
				suitable_subtree = false;
		}
		if (suitable_subtree) {
			// 2. Check if it uses any other labels than those in intersection_leaflet. If it does, they must belong to single_use_labels
			check_suitability = expr_labels & (intersection_leafset | single_use_labels);
			if (check_suitability != expr_labels) 
				suitable_subtree = false;
		}
		check_suitability.clear (); 
		// If the tree is suitable, add it to ret_subtrees
		if (suitable_subtree) {
			subtrees.push_back (make_tuple (this, *it));
			if (DEBUG) {
				printf ("found suitable subtree with %s = %s\n", get_lhs_label ().c_str(), (*it)->get_expr_string().c_str());
				printf ("label use " ); cout << expr_labels << endl; 
				printf ("intersection "); cout << intersection_leafset << endl;
				printf ("leaf labels "); cout << leaf_labels << endl;	
				printf ("extra labels "); cout << single_use_labels << endl;
			}
		}
		// Otherwise recursively call the function
		else {
			vector<treenode*> &rhs_operands = (*it)->get_operands ();
			// iterate over the subtrees
			for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) 
				(*jt)->identify_optimizable_subtrees (subtrees, intersection_leafset, single_use_labels, leaf_labels, label_count);
		}
	}
}

// Identify optimizable subtrees based on the intra algorithm
void treenode::identify_leafy_computation_subtrees (vector<tuple<treenode*,accnode*>> &computations, boost::dynamic_bitset<> leaf_labels, map<string,int> label_bitset_index, int label_count) {
	// Iterate over accumulations
	for (vector<accnode*>::iterator it=rhs_accs.begin(); it!=rhs_accs.end(); it++) {
		boost::dynamic_bitset<> expr_labels = (*it)->get_used_labels ();
		boost::dynamic_bitset<> check_suitability (label_count);
		// 0. Check that the subtree rooted at it has all leaf nodes except for the lhs
		check_suitability = expr_labels & ~leaf_labels;
		check_suitability[label_bitset_index[lhs_label]] = false;
		if (check_suitability.none ()) { 
			computations.push_back (make_tuple (this, *it));
			//if (DEBUG) {
			//	printf ("found suitable subtree with %s = %s\n", get_lhs_label ().c_str(), (*it)->get_expr_string().c_str());
			//	printf ("leaf labels "); cout << leaf_labels << endl;	
			//}
		}
		// Otherwise recursively call the function
		else {
			vector<treenode*> &rhs_operands = (*it)->get_operands ();
			// iterate over the subtrees
			for (vector<treenode*>::iterator jt=rhs_operands.begin(); jt!=rhs_operands.end(); jt++) 
				(*jt)->identify_leafy_computation_subtrees (computations, leaf_labels, label_bitset_index, label_count);
		}
		check_suitability.clear ();
	}
}

void treenode::identify_optimizable_leafy_subtrees (vector<tuple<treenode*,accnode*>> &subtrees, vector<tuple<treenode*,accnode*>> &computations, boost::dynamic_bitset<> &grown_set, map<string,int> label_bitset_index, int label_count) {
	int prev_size = 0, updated_size = 0;
	do {
		prev_size = subtrees.size ();
		// Iterate over computations 
		for (vector<tuple<treenode*,accnode*>>::iterator it=computations.begin(); it!=computations.end(); it++) {
			if (find (subtrees.begin(), subtrees.end(), *it) != subtrees.end ()) continue;
			boost::dynamic_bitset<> expr_labels (label_count);
			expr_labels = (get<1>(*it))->get_used_labels ();
			expr_labels[label_bitset_index[(get<0>(*it))->get_lhs_label()]] = true;
			boost::dynamic_bitset<> intersection (label_count);
			intersection = expr_labels & grown_set;
			if (intersection.any ()) {
				subtrees.push_back (*it);
				grown_set |= expr_labels;
				if (DEBUG) {
					printf ("found suitable subtree with %s = %s\n", (get<0>(*it))->get_lhs_label ().c_str(), (get<1>(*it))->get_expr_string().c_str());
					printf ("grown set "); cout << grown_set << endl;	
				}
			}
			expr_labels.clear (); intersection.clear (); 
		}
		updated_size = subtrees.size ();
	} while (updated_size != prev_size);
}
