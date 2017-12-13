#define STMT_FOREST_LIMIT (10)
using namespace std;

void funcdefn::simplify_accumulations (vector<tuple<expr_node*,expr_node*,STMT_OP>> &tstmt, int &s_count) {
	vector<tuple<expr_node*,expr_node*,STMT_OP>> ret_tstmt;
	for (vector<tuple<expr_node*,expr_node*,STMT_OP>>::iterator i=tstmt.begin(); i!=tstmt.end(); i++) {
		if (get<2>(*i) == ST_EQ) 
			ret_tstmt.push_back (*i);
		else {
			// We have an accumulation. If RHS is binary and none of the children are datanodes, insert something else.
			if (get<1>(*i)->get_expr_type () == T_BINARY && !get<1>(*i)->is_data_type ()) {
				string name_t = "_v_" + to_string (s_count++) + "_";
				expr_node *temp = new id_node (name_t);
				temp_vars.push_back (temp);
				ret_tstmt.push_back (make_tuple (temp, get<1>(*i), ST_EQ));
				ret_tstmt.push_back (make_tuple (get<0>(*i), temp, get<2>(*i)));	
			}
			else ret_tstmt.push_back (*i);
		}
	}
	tstmt = ret_tstmt;
}

// Create clusters of trees. Initially, a cluster contains a tree that writes to 
// one value. Later, it may be expanded to trees that write to multiple values,
// but read from the same set of inputs. 
void funcdefn::create_tree_clusters (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	int curr_cluster_num = 0;
	map<string,treenode*> tree_map; 
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		int orig_stmt_num = (*i)->get_orig_stmt_num ();
		// Check if we need to start a new cluster
		if (orig_stmt_num != curr_cluster_num) {
			// Add whatever was in the tree map into the forest first 
			if (DEBUG) assert (tree_map.size () == 1 && "Failed to create a single cluster for the original input statement (create_tree_clusters)");
			if (DEBUG) assert (stmt_forest.find (curr_cluster_num) == stmt_forest.end () && "The cluster for statement already exists in the forest (create_tree_clusters)");
			vector <treenode*> forest_tree;
			map<string,treenode*>::iterator it = tree_map.begin ();
			forest_tree.push_back (it->second);
			if (DEBUG) printf ("Adding %s to forest\n", (it->first).c_str ());
			stmt_forest[curr_cluster_num] = forest_tree;
			tree_map.clear ();
			// Now update the cluster number
			curr_cluster_num = orig_stmt_num;
		}
		// Continue populating the tree now
		expr_node *lhs_node = (*i)->get_lhs_expr ();
		expr_node *rhs_node = (*i)->get_rhs_expr ();
		vector<string> l_labels = (*i)->get_lhs_labels ();
		stringstream lhs_print;
		lhs_node->print_node (lhs_print);
		for (vector<string>::iterator j=l_labels.begin(); j!=l_labels.end(); j++) {
			// Check if it is an accumulation
			treenode *tree_node;
			// Find the root of accumulation first
			if (tree_map.find (*j) != tree_map.end ())
				tree_node = tree_map[*j];
			else
				tree_node = new treenode (lhs_node, lhs_print.str(), *j, label_bitset_index, label_count, false);
			accnode *new_accnode = new accnode (label_count);
			rhs_node->populate_tree (new_accnode, tree_map, label_bitset_index, label_count, false);
			new_accnode->set_assignment_op ((*i)->get_op_type ());
			tree_node->add_rhs_expr (new_accnode, label_bitset_index, label_count);
			if (DEBUG) {
				printf ("The label use bitset for %s is : ", (*j).c_str ());
				print_bitset (label_bitset_index, tree_node->get_used_labels(), tree_node->get_use_frequency (), label_count); 
			}
			tree_map[*j] = tree_node;
		}
	}
	// Add the last tree to the forest
	if (DEBUG) assert (tree_map.size () == 1 && "Failed to create a single cluster for the original input statement (create_tree_clusters)");
	if (DEBUG) assert (stmt_forest.find (curr_cluster_num) == stmt_forest.end () && "The cluster for statement already exists in the forest (create_tree_clusters)");
	vector <treenode*> forest_tree;
	map<string,treenode*>::iterator it = tree_map.begin ();
	forest_tree.push_back (it->second);
	if (DEBUG) printf ("Adding %s to forest\n", (it->first).c_str ());
	stmt_forest[curr_cluster_num] = forest_tree;
	tree_map.clear ();
}

// Determine if the tree_sequence does not violate dependence
bool funcdefn::valid_permutation (vector<int> tree_sequence) {
	// Create a map which represents the position of cluster in tree_sequence
	map<int,int> tree_sequence_map;
	int pos = 0;
	for (vector<int>::iterator it=tree_sequence.begin(); it!=tree_sequence.end(); it++,pos++)
		tree_sequence_map[*it] = pos;

	// Now go over the dependence graph, and verify that all the dependences are satisfied
	for (map<int, vector<int>>::iterator it=cluster_dependence_graph.begin(); it!=cluster_dependence_graph.end(); it++) {
		int dest_cluster_pos = tree_sequence_map[it->first];
		vector<int> host_cluster_vec = it->second;
		for (vector<int>::iterator jt=host_cluster_vec.begin(); jt!=host_cluster_vec.end(); jt++) {
			int host_cluster_pos = tree_sequence_map[*jt];
			if (host_cluster_pos > dest_cluster_pos) return false;
		}
	}		
	tree_sequence_map.clear ();
	return true;
}

// Find the cluster ordering with lowest register cost
void funcdefn::get_lowest_cost_configuration (stringstream &original, stringstream &reordered) {
	// Post PLDI: Try to make some optimizations based on the size of stmt_forest. If there are a large
	// number of trees, then mark trees as independent if they share something that is shared by all the
	// trees. This means that if a value is live throughout the computation, then don't consider it while
	// determining independence of trees (don't count it as shared data).
	boost::dynamic_bitset<> common_labels (label_count);
	boost::dynamic_bitset<> leaf_labels (label_count); 
	if (stmt_forest.size () >= STMT_FOREST_LIMIT) {
		common_labels.set ();
		for (map<int,vector<treenode*>>::const_iterator it=stmt_forest.begin(); it!=stmt_forest.end(); it++) {
			treenode *tree = (it->second).front ();
			boost::dynamic_bitset<> label_use = tree->get_used_labels ();
			common_labels &= label_use;
		}
		if (DEBUG) {
			printf ("Since forest size (%d) exceeds STMT_FOREST_LIMIT, common label bitset is ", (int)stmt_forest.size());
			print_bitset (label_bitset_index, common_labels, label_count); 
		}
	}
	// Find the disjoint trees in the forest
	for (map<int,vector<treenode*>>::const_iterator it=stmt_forest.begin(); it!=stmt_forest.end(); it++) {
		bool disjoint_forest = true;
		boost::dynamic_bitset<> intersection (label_count);
		// Get the labels used in the source statement
		int src_tree_num = it->first;
		treenode* src_tree = (it->second).front ();
		boost::dynamic_bitset<> src_label_use = src_tree->get_used_labels ();
		for (map<int,vector<treenode*>>::const_iterator jt=stmt_forest.begin(); jt!=stmt_forest.end(); jt++) {
			int dest_tree_num = jt->first;
			if (src_tree_num == dest_tree_num) continue;
			// Get the labels used in the destination statement
			treenode *dest_tree = (jt->second).front ();
			boost::dynamic_bitset<> dest_label_use = dest_tree->get_used_labels ();
			intersection = (src_label_use & dest_label_use) & ~common_labels;
			if (intersection.any ()) {
				disjoint_forest = false;
				break;
			}
		}
		if (disjoint_forest) disjoint_forests.push_back (src_tree_num);
		// Compute a bitset of all the labels that are leaf nodes
		src_tree->compute_leaf_nodes (leaf_labels, label_bitset_index);
	}
	// TODO: Add clustering technique based on common_labels

	if (DEBUG) printf ("%d disjoint forests found\n", (int)disjoint_forests.size());
	// Try the initial tree sequence for optimization.
	vector<int> tree_sequence;
	for (map<int,vector<treenode*>>::const_iterator it=stmt_forest.begin(); it!=stmt_forest.end(); it++) {
		if (find (disjoint_forests.begin(), disjoint_forests.end(), it->first) == disjoint_forests.end())	
			tree_sequence.push_back (it->first);
	}
	vector<tuple<int,int,vector<int>>> reg_count;
	// Compute the register required for disjoint forest
	for (vector<int>::iterator it=disjoint_forests.begin(); it!=disjoint_forests.end(); it++) {
		vector<int> t_vec;
		t_vec.push_back (*it);
		// Populate a map reflecting the orientation for lowest configuration
		map<treenode*, vector<int>> opt_treenode_config;
		map<accnode*, vector<int>> opt_accnode_config; 
		tuple<int,int> reg = compute_register_optimal_schedule (*it, opt_treenode_config, opt_accnode_config);
		reg_count.push_back (make_tuple (get<0>(reg), get<1>(reg), t_vec));
		// Now retrace the optimal configuration
		retrace_register_optimal_schedule (*it, opt_treenode_config, opt_accnode_config);
		opt_treenode_config.clear ();
		opt_accnode_config.clear ();	
	}
	// tree_sequence is sorted, so permute it
	vector<int> opt_sequence;
	if (tree_sequence.size () > 0) {
		map<treenode*, vector<int>> opt_treenode_config;
		map<accnode*, vector<int>> opt_accnode_config;
		int min_register = INT_MAX, free_regs = 0;
		int permutations = 0;
		// If there is dependence, then only try valid permutations
		if (cluster_dependence_graph.size () == 0) {
			// First try sequential permutations
			do {
				map <treenode*, vector<int>> treenode_config;
				map <accnode*, vector<int>> accnode_config; 
				tuple<int, int> reg = compute_register_optimal_schedule (tree_sequence, treenode_config, accnode_config);
				if (get<0>(reg) < min_register) {
					min_register = get<0>(reg);
					free_regs = get<1>(reg);
					opt_sequence.clear (); opt_treenode_config.clear (); opt_accnode_config.clear ();
					opt_sequence = tree_sequence;
					opt_treenode_config = treenode_config;
					opt_accnode_config = accnode_config;
				}
				treenode_config.clear (); 
				accnode_config.clear ();
				permutations++;
			} while (next_permutation (tree_sequence.begin(), tree_sequence.end()) && permutations<PERM_LIMIT);
			// If the permuatations were stopped due to permutations reaching PERM_LIMIT, try random shuffles
			if (permutations == PERM_LIMIT) {
				permutations = 0;
				while (permutations<PERM_LIMIT) {
					random_shuffle (tree_sequence.begin(), tree_sequence.end());
					map <treenode*, vector<int>> treenode_config;
					map <accnode*, vector<int>> accnode_config;
					tuple<int, int> reg = compute_register_optimal_schedule (tree_sequence, treenode_config, accnode_config);
					if (get<0>(reg) < min_register) {
						min_register = get<0>(reg);
						free_regs = get<1>(reg);
						opt_sequence.clear (); opt_treenode_config.clear (); opt_accnode_config.clear ();
						opt_sequence = tree_sequence;
						opt_treenode_config = treenode_config;
						opt_accnode_config = accnode_config;
					}
					treenode_config.clear ();
					accnode_config.clear ();
					permutations++;
				}
			}
		}
		else {
			// First try sequential permutations
			do {
				if (valid_permutation (tree_sequence)) {
					map <treenode*, vector<int>> treenode_config;
					map <accnode*, vector<int>> accnode_config; 
					tuple<int, int> reg = compute_register_optimal_schedule (tree_sequence, treenode_config, accnode_config);
					if (get<0>(reg) < min_register) {
						min_register = get<0>(reg);
						free_regs = get<1>(reg);
						opt_sequence.clear (); opt_treenode_config.clear (); opt_accnode_config.clear ();
						opt_sequence = tree_sequence;
						opt_treenode_config = treenode_config;
						opt_accnode_config = accnode_config;
					}
					treenode_config.clear (); 
					accnode_config.clear ();
				}
				permutations++;
			} while (next_permutation (tree_sequence.begin(), tree_sequence.end()) && permutations<PERM_LIMIT);
			// If the permuatations were stopped due to permutations reaching PERM_LIMIT, try random shuffles
			if (permutations == PERM_LIMIT) {
				permutations = 0;
				while (permutations<PERM_LIMIT) {
					random_shuffle (tree_sequence.begin(), tree_sequence.end());
					if (valid_permutation (tree_sequence)) {
						map <treenode*, vector<int>> treenode_config;
						map <accnode*, vector<int>> accnode_config;
						tuple<int, int> reg = compute_register_optimal_schedule (tree_sequence, treenode_config, accnode_config);
						if (get<0>(reg) < min_register) {
							min_register = get<0>(reg);
							free_regs = get<1>(reg);
							opt_sequence.clear (); opt_treenode_config.clear (); opt_accnode_config.clear ();
							opt_sequence = tree_sequence;
							opt_treenode_config = treenode_config;
							opt_accnode_config = accnode_config;
						}
						treenode_config.clear ();
						accnode_config.clear ();
					}
					permutations++;
				}
			}
		}
		// Add the registers required for the dependent trees into reg_count
		reg_count.push_back (make_tuple (min_register, free_regs, opt_sequence));
		// Now retrace the optimal configuration
		retrace_register_optimal_schedule (opt_sequence, opt_treenode_config, opt_accnode_config);
		opt_treenode_config.clear ();
		opt_accnode_config.clear ();
	}
	// Now compute the accumulated registers required
	int reg_used = 0, free_regs = 0;
	tree_sequence.clear ();
	if (reg_count.size () > 1) stable_sort (reg_count.begin(), reg_count.end(), sort_opt_reg_cost);
	for (vector<tuple<int, int, vector<int>>>::iterator it=reg_count.begin(); it!=reg_count.end(); it++) {
		if (get<0>(*it) >= free_regs) {
			reg_used += (get<0>(*it) - free_regs);
			free_regs = get<1>(*it); 
		}
		else {
			free_regs -= get<0>(*it);
			free_regs += get<1>(*it);
		}
		tree_sequence.insert (tree_sequence.end (), get<2>(*it).begin (), get<2>(*it).end ());
	}
	reg_count.clear ();
	if (DEBUG) printf ("Maximum Register Required = (%d, %d)\n", reg_used, free_regs);

	// Compute a bitset of all labels that are used only once (use and throw)
	boost::dynamic_bitset<> single_use_labels (label_count);
	for (int i=0; i<label_count; i++)
		single_use_labels[i] = (label_frequency[i] == 1);
	if (DEBUG) {
		printf ("single_use_labels bitset is : ");
		print_bitset (label_bitset_index, single_use_labels, label_count);
		printf ("leaf_labels bitset is : ");
		print_bitset (label_bitset_index, leaf_labels, label_count); 
	}
	// Perform optimizations across forests
	if (opt_sequence.size() > 1) {
		// Compute the live-out for all the dependent trees
		map<int, tuple<boost::dynamic_bitset<>,unsigned int*>> liveness_map;
		compute_liveness_map (liveness_map, opt_sequence);
		// Print the forests before inter-optimizing
		if (DEBUG) {
			cout << "\nCLUSTERS BEFORE INTER-CLUSTER OPTIMIZATIONS\n";
			original << "\n//CLUSTERS BEFORE INTER-CLUSTER OPTIMIZATIONS\n";
			print_forests (original, tree_sequence, leaf_labels, single_use_labels);
		}
		if (DEBUG) printf ("\nPERFORMING INTER-CLUSTER OPTIMIZATIONS\n");
		if (INTRA_TYPE_INTER_OPT) 
			fixed_order_intra_type_inter_forest_optimizations (opt_sequence, liveness_map, single_use_labels, leaf_labels);
		else {
			if (RESTRICT_INTER_OPT) 
				fixed_order_inter_forest_optimizations (opt_sequence, liveness_map, single_use_labels, leaf_labels);
			else 
				fixed_order_inter_forest_optimizations (opt_sequence, liveness_map, single_use_labels);
		}
		liveness_map.clear ();
	}
	// Print the forests after inter-optimizing
	if (DEBUG) {
		cout << "\nCLUSTERS AFTER INTER-CLUSTER OPTIMIZATIONS\n";
		//original << "\n//CLUSTERS AFTER INTER-CLUSTER OPTIMIZATIONS\n";
		print_forests (original, tree_sequence, leaf_labels, single_use_labels);
	}
	opt_sequence.clear ();
	if (DEBUG) printf ("\nPERFORMING INTRA-CLUSTER OPTIMIZATIONS\n");
	map<int, tuple<boost::dynamic_bitset<>,unsigned int*>> intra_liveness_map;
	compute_liveness_map (intra_liveness_map, tree_sequence);
	intra_forest_optimizations (tree_sequence, intra_liveness_map, leaf_labels, single_use_labels);
	copy_propagation (tree_sequence);
	if (DEBUG) {
		cout << "\nCLUSTER AFTER INTRA-CLUSTER OPTIMIZATIONS\n";
		//reordered << "\n//CLUSTER AFTER INTRA-CLUSTER OPTIMIZATIONS\n";
	}
	print_forests (reordered, tree_sequence);
	// Free all the data structures
	single_use_labels.clear (); leaf_labels.clear (); common_labels.clear (); 
	intra_liveness_map.clear ();
	tree_sequence.clear ();
}

void funcdefn::compute_liveness_map (map<int, tuple<boost::dynamic_bitset<>,unsigned int*>> &liveness_map, vector<int> opt_sequence) {
	boost::dynamic_bitset<> t_bitset (label_count);
	unsigned int *t_freq = new unsigned int[label_count] ();

	for (vector<int>::iterator it=opt_sequence.begin(); it!=opt_sequence.end(); it++) {
		treenode *tree = stmt_forest[*it].front ();
		liveness_map[*it] = tree->compute_liveness (t_bitset, t_freq, label_frequency, label_count);
	}
	// Free the resources
	t_bitset.clear ();
	delete[] t_freq;
}

tuple<int, int> funcdefn::compute_register_optimal_schedule (vector<int> tree_sequence, map<treenode*, vector<int>> &treenode_config, map<accnode*, vector<int>> &accnode_config) {
	boost::dynamic_bitset<> live (label_count);
	unsigned int *live_freq = new unsigned int[label_count] ();
	vector<tuple<int, int>> reg_count;
	// Iterate over each tree in the cluster, and compute register optimal schedule for it
	for (vector<int>::iterator it=tree_sequence.begin(); it!=tree_sequence.end(); it++) {
		treenode *tree = stmt_forest[*it].front ();
		// This is the recursive call
		if (DEBUG) {
			printf ("For tree %d, livein set is : ", *it);
			print_bitset (label_bitset_index, live, label_count);
		}
		reg_count.push_back (tree->compute_register_optimal_schedule (live, live_freq, label_bitset_index, label_frequency, label_count, treenode_config, accnode_config));
		if (DEBUG) {
			printf ("For tree %d, liveout set is : ", *it);
			print_bitset (label_bitset_index, live, label_count);
		}
	}
	// Now compute the accumulated registers required
	int reg_used = 0, free_regs = 0;
	for (vector<tuple<int, int>>::iterator it=reg_count.begin(); it!=reg_count.end(); it++) {
		if (get<0>(*it) >= free_regs) {
			reg_used += (get<0>(*it) - free_regs);
			free_regs = get<1>(*it); 
		}
		else {
			free_regs -= get<0>(*it);
			free_regs += get<1>(*it);
		}
	}
	// Free resources
	delete[] live_freq;
	live.clear ();
	reg_count.clear ();
	return make_tuple (reg_used, free_regs);
}

void funcdefn::retrace_register_optimal_schedule (vector<int> tree_sequence, map<treenode*, vector<int>> &treenode_config, map<accnode*, vector<int>> &accnode_config) {
	for (vector<int>::iterator it=tree_sequence.begin(); it!=tree_sequence.end(); it++) {
		treenode *tree = stmt_forest[*it].front ();
		tree->retrace_register_optimal_schedule (treenode_config, accnode_config);
	}
}

// Compute the register optimal schedule for an independent tree, and make the changes to orientation
tuple<int, int> funcdefn::compute_register_optimal_schedule (int tree_id, map<treenode*, vector<int>> &treenode_config, map<accnode*, vector<int>> &accnode_config) {
	boost::dynamic_bitset<> live (label_count);
	unsigned int *live_freq = new unsigned int[label_count] ();
	tuple<int, int> reg_count;
	treenode *tree = stmt_forest[tree_id].front ();
	if (DEBUG) {
		printf ("For tree %d, livein set is : ", tree_id);
		print_bitset (label_bitset_index, live, label_count);
	}
	reg_count = tree->compute_register_optimal_schedule (live, live_freq, label_bitset_index, label_frequency, label_count, treenode_config, accnode_config);
	if (DEBUG) {
		printf ("For tree %d, liveout set is : ", tree_id);
		print_bitset (label_bitset_index, live, label_count);
	}
	// Free resources
	delete[] live_freq;
	live.clear ();
	return reg_count;
}

void funcdefn::retrace_register_optimal_schedule (int tree_id, map<treenode*, vector<int>> &treenode_config, map<accnode*, vector<int>> &accnode_config) {
	treenode *tree = stmt_forest[tree_id].front ();
	tree->retrace_register_optimal_schedule (treenode_config, accnode_config);
}

//void funcdefn::retrace_register_optimal_schedule (vector<int> tree_sequence) {
//	boost::dynamic_bitset<> live (label_count);
//	unsigned int *live_freq = new unsigned int[label_count] ();
//	vector<tuple<int, int>> reg_count;
//	// Iterate over each tree in the cluster, and compute register optimal schedule for it
//	for (vector<int>::iterator it=tree_sequence.begin(); it!=tree_sequence.end(); it++) {
//		treenode *tree = stmt_forest[*it].front ();
//		// This is the recursive call
//		if (DEBUG) {
//			printf ("For tree %d, livein set is : ", *it);
//			print_bitset (label_bitset_index, live, label_count);
//		}
//		reg_count.push_back (tree->retrace_register_optimal_schedule (live, live_freq, label_bitset_index, label_frequency, label_count));
//		if (DEBUG) {
//			printf ("For tree %d, liveout set is : ", *it);
//			print_bitset (label_bitset_index, live, label_count);
//		}
//	}
//	// Now compute the accumulated registers required
//	int reg_used = 0, free_regs = 0;
//	for (vector<tuple<int, int>>::iterator it=reg_count.begin(); it!=reg_count.end(); it++) {
//		if (get<0>(*it) >= free_regs) {
//			reg_used += (get<0>(*it) - free_regs);
//			free_regs = get<1>(*it); 
//		}
//		else {
//			free_regs -= get<0>(*it);
//			free_regs += get<1>(*it);
//		}
//	}
//	printf ("Maximum Register Required After Retrace = (%d, %d)\n", reg_used, free_regs);
//	// Free resources
//	delete[] live_freq;
//	live.clear ();
//	reg_count.clear ();
//}

void funcdefn::intra_forest_optimizations (vector<int> tree_sequence, map<int, tuple<boost::dynamic_bitset<>,unsigned int*>> &liveness_map, boost::dynamic_bitset<>leaf_labels, boost::dynamic_bitset<>single_use_labels) {
	boost::dynamic_bitset<> livein (label_count);
	unsigned int *livein_freq = new unsigned int[label_count] ();
	// Iterate over each tree in the cluster, and compute register optimal schedule for it
	for (vector<int>::iterator it=tree_sequence.begin(); it!=tree_sequence.end(); it++) {
		treenode *tree = stmt_forest[*it].front ();
		if (DEBUG) {
			printf ("For tree %d, livein set is : ", *it);
			print_bitset (label_bitset_index, livein, label_count);
		}
		fixed_order_intra_tree_optimizations (*it, livein, livein_freq, single_use_labels);
		// Retrieve the liveout values from liveness_map
		livein = get<0>(liveness_map[*it]);
		livein_freq = get<1>(liveness_map[*it]);
		if (DEBUG) {
			printf ("For tree %d, liveout set is : ", *it);
			print_bitset (label_bitset_index, livein, label_count);
		}
		// After everything, now append the code of subtrees to main tree
		for (vector<treenode*>::iterator jt=next(stmt_forest[*it].begin()); jt!=stmt_forest[*it].end(); jt++) {
			if ((*jt)->is_data_node () || (*jt)->is_leaf_node ()) continue;
			if (DEBUG) {
				printf ("In here with %s, %s\n", (*jt)->get_lhs_label().c_str (), ((*jt)->get_rhs_operands().back())->get_expr_string().c_str());
				printf ("Trying to append to tree rooted at %s\n", tree->get_lhs_label().c_str());
			}
			bool appended = tree->append_subtree_to_code (*jt, leaf_labels, single_use_labels, label_bitset_index, label_count);
			// If failed to append, append it before this tree
			if (!appended) {
				printf ("Failed to append, appending to tree rooted at %s\n", tree->get_lhs_label().c_str());
				//tree->append_to_code (*jt);
				tree->update_appended_info (*jt, label_bitset_index, label_count);
				// Add the treenode to the tree
				tree->add_spliced_treenode (*jt);
			}
		}
		// Pop out all the subtrees from the forest
		while (stmt_forest[*it].size() != 1) 
			stmt_forest[*it].pop_back ();
	}
}

void funcdefn::fixed_order_intra_tree_optimizations (int cluster_id, boost::dynamic_bitset<> &livein, unsigned int *livein_freq, boost::dynamic_bitset<>single_use_labels) {
	vector<string> cull_labels;
	treenode * &mod_tree = (stmt_forest[cluster_id]).front();
	// Create a map of expr to lhs for host
	map<string,tuple<string,expr_node*>> expr_lhs_map;
	// Now perform the check. From last positions of the vector, we can move computations forward.
	int restart_pos = 0;
	bool tree_changed = false;
	do {
		// Iterate over the subtrees in the front of the cluster to form a vector of treenodes
		vector<tuple<treenode*,accnode*>> computations;
		mod_tree->recompute_tree (label_bitset_index, computations, cull_labels, label_count);
		if (computations.size () == 0) continue;

		tree_changed = false;
		for (vector<tuple<treenode*,accnode*>>::iterator it=computations.begin()+restart_pos; it!=computations.end(); it++,restart_pos++) {
			vector<int> opt_vec;
			// Make an entry for iterator it into expr_lhs_map if not an accumulation
			string lhs_label = (get<0>(*it))->get_lhs_label ();
			string rhs_string = (get<1>(*it))->get_expr_string ();
			if (!(get<0>(*it))->is_accumulation_node () && (get<1>(*it))->is_asgn_eq_op ()) {
				if (expr_lhs_map.find (rhs_string) == expr_lhs_map.end ())
					expr_lhs_map[rhs_string] = make_tuple (lhs_label, (get<0>(*it))->get_lhs());
			}
			boost::dynamic_bitset<> src_labels (label_count);
			src_labels = (get<1>(*it))->get_used_labels ();
			src_labels[label_bitset_index[lhs_label]] = true;
			// Now iterate over rest of the trees, and find optimizable trees
			if (DEBUG) {
				printf ("source is (%s - %s), label bitset is ", lhs_label.c_str(), rhs_string.c_str());
				print_bitset (label_bitset_index, src_labels, label_count);
				printf ("live-in at node (%s - %s) is ", lhs_label.c_str(), rhs_string.c_str());
				print_bitset (label_bitset_index, livein, label_count);
			}
			// First try the maximal clustering
			boost::dynamic_bitset<> grown_bitset (src_labels);
			vector<tuple<treenode*,accnode*>> visited_computations;
			bool grow_set = false;
			bool only_accumulation = true;
			do {
				grow_set = false;
				for (vector<tuple<treenode*,accnode*>>::iterator jt=next(it); jt!=computations.end(); jt++) {
					if (find (visited_computations.begin(), visited_computations.end(), *jt) != visited_computations.end()) continue;
					if (((get<0>(*jt))->get_spliced_treenodes()).size () > 0) continue;	
					// Find the intersection of labels between jt and it
					boost::dynamic_bitset <> dest_labels (label_count);
					dest_labels = (get<1>(*jt))->get_used_labels ();
					dest_labels[label_bitset_index[(get<0>(*jt))->get_lhs_label()]] = true;
					boost::dynamic_bitset<> intersection (label_count);
					intersection = dest_labels & grown_bitset;
					if (intersection.any ()) {
						grown_bitset |= dest_labels;
						if (DEBUG) {
							printf ("dest with non-empty intersection is (%s - %s), label bitset is ", (get<0>(*jt))->get_lhs_label().c_str(), (get<1>(*jt))->get_expr_string().c_str());
							print_bitset (label_bitset_index, dest_labels, label_count);
						}
						opt_vec.push_back (jt-computations.begin());
						visited_computations.push_back (*jt);
						// Check if dest intersected only because of the accumulation
						intersection = src_labels & dest_labels;
						intersection[label_bitset_index[lhs_label]] = false;
						if (intersection.any ()) only_accumulation = false;
						grow_set = true;
					}
					dest_labels.clear (); intersection.clear ();
				}
			} while (grow_set);
			// If opt_vec is is empty, try the minimal clustering (this is actually not required).
			if (opt_vec.size () == 0) {
				for (vector<tuple<treenode*,accnode*>>::iterator jt=next(it); jt!=computations.end(); jt++) {
					if (((get<0>(*jt))->get_spliced_treenodes()).size () > 0) continue;	
					// Find the intersection of labels between jt and it
					boost::dynamic_bitset<> dest_labels (label_count);
					dest_labels = (get<1>(*jt))->get_used_labels ();
					dest_labels[label_bitset_index[(get<0>(*jt))->get_lhs_label()]] = true;
					boost::dynamic_bitset<> intersection (label_count);
					intersection = src_labels & dest_labels;
					if (intersection.any ()) {
						if (DEBUG) {
							printf ("Intersection is "); 
							print_bitset (label_bitset_index, intersection, label_count);
							printf ("lhs is %s\n", (get<0>(*jt))->get_lhs_label().c_str());
						}
						opt_vec.push_back (jt-computations.begin());
						// Check if dest intersected only because of the accumulation
						intersection[label_bitset_index[lhs_label]] = false;
						if (intersection.any ()) only_accumulation = false;
					}
					dest_labels.clear (); intersection.clear ();
				}
			}
			// First check if all that is common between the computations in opt_vec is just the LHS. In that case,
			// they just are a part of accumulation. Makes no sense to keep them together.
			if (only_accumulation && opt_vec.size() > 0) {
				if (DEBUG) printf ("Clearing up the opt_vec, since all the computations are together only for accumulation\n");
				opt_vec.clear (); 
			}
			// If opt_vec is still empty, no point continuing further
			if (opt_vec.size () == 0) continue;
			// Create a copy of the livein frequency till now, and update it assuming the host statement is executed
			boost::dynamic_bitset<> t_livein (livein);
			t_livein |= (get<1>(*it))->get_used_labels () |  (get<1>(*it))->get_appended_labels ();
			t_livein[label_bitset_index[(get<0>(*it))->get_lhs_label()]] = true;
			unsigned int *t_livein_freq = new unsigned int[label_count] ();
			copyArray (t_livein_freq, livein_freq, label_count);
			unsigned int *t_freq = (get<1>(*it))->get_use_frequency ();
			addArrays (t_livein_freq, t_freq, label_count);
			addArrays (t_livein_freq, (get<1>(*it))->get_appended_frequency(), label_count);
			t_livein_freq[label_bitset_index[(get<0>(*it))->get_lhs_label()]] += 1;
			for (int i=0; i<label_count; i++) {
				if (t_livein[i] && t_livein_freq[i] < label_frequency[i]) t_livein[i] = true;
				if (t_livein[i] && t_livein_freq[i] == label_frequency[i]) t_livein[i] = false;
			}
			if (determine_intra_opt_profitability (computations, opt_vec, t_livein, t_livein_freq, expr_lhs_map, grown_bitset, single_use_labels)) {
				// Put everything in code string, and change the labels of the tree
				for (vector<int>::iterator jt=opt_vec.begin(); jt!=opt_vec.end(); jt++) {
					treenode * &subtree = get<0>(computations[*jt]);
					accnode * &rhs_expr = get<1>(computations[*jt]);
					// Remove the subtrees
					bool not_found = true;
					string lhs_rep = subtree->get_lhs_label ();
					string expr_string = rhs_expr->get_expr_string ();
					if (!subtree->is_accumulation_node ()) {
						if (expr_lhs_map.find (expr_string) != expr_lhs_map.end ()) {
							lhs_rep = get<0>(expr_lhs_map[expr_string]);
							// Reduce the label counts of rhs_expr from necessary bitsets (label_frequency)
							subtractArrays (label_frequency, rhs_expr->get_use_frequency(), label_count);
							if (DEBUG) {
								unsigned int *t_freq = rhs_expr->get_use_frequency ();
								for (int i=0; i<label_count; i++) {
									if (t_freq[i] != 0) 
										printf ("Reduced the label frequency[%d] from %d to %d\n", i, label_frequency[i]+t_freq[i], label_frequency[i]);
								}
							}
							// Increment the label count for the LHS
							label_frequency[label_bitset_index[lhs_rep]] += 1;
							if (DEBUG) printf ("increased the label frequency of %s from %d to %d\n", lhs_rep.c_str(), label_frequency[label_bitset_index[lhs_rep]]-1, label_frequency[label_bitset_index[lhs_rep]]);
							not_found = false;
						}
						else {
							if (rhs_expr->is_asgn_eq_op ()) 
								expr_lhs_map[expr_string] = make_tuple (lhs_rep, subtree->get_lhs ());
							// lhs label must be culled
							if (find (cull_labels.begin(), cull_labels.end(), lhs_rep) == cull_labels.end())
								cull_labels.push_back (lhs_rep);
						}
					}
					if (not_found || subtree->is_accumulation_node ()) {
						stringstream lhs_print;
						(subtree->get_lhs())->print_node (lhs_print);
						treenode *t_node = new treenode (subtree->get_lhs (), lhs_print.str(), lhs_rep, label_bitset_index, label_count, false);
						t_node->add_rhs_expr (rhs_expr, label_bitset_index, label_count);
						//(get<1>(*it))->append_to_code (t_node);
						(get<1>(*it))->update_appended_info (t_node, label_bitset_index, label_count);
						(get<1>(*it))->add_spliced_treenode (t_node);
					}
					// Modify the trees from which the subtree was sliced
					if (!subtree->is_accumulation_node ()) {
						if (not_found) subtree->reset_rhs_accs ();
						else {
							stringstream lhs_print;
							(get<1>(expr_lhs_map[expr_string]))->print_node (lhs_print);
							treenode *t_node = new treenode (get<1>(expr_lhs_map[expr_string]), lhs_print.str(), lhs_rep, label_bitset_index, label_count, false);
							accnode *old_rhs = (subtree->get_rhs_operands ()).front();
							accnode *new_rhs = new accnode (label_count, lhs_rep, t_node, old_rhs->get_assignment_op());
							subtree->reset_rhs_accs (new_rhs);
						}
					}
					else {
						vector<accnode*> &t_rhs_accs = subtree->get_rhs_operands ();
						int pos = 0;
						for (vector<accnode*>::iterator kt=t_rhs_accs.begin(); kt!=t_rhs_accs.end(); kt++, pos++) {
							if (expr_string.compare ((*kt)->get_expr_string()) == 0) 
								break;
						}
						if (DEBUG) assert (pos < t_rhs_accs.size () && "Element to be sliced not found in the rhs operands (intra cluster)");
						t_rhs_accs.erase (t_rhs_accs.begin()+pos);
					}
				}
				tree_changed = true;
				restart_pos++;
			}
			// Free resources
			grown_bitset.clear (); src_labels.clear (); t_livein.clear ();
			delete[] t_livein_freq;
			// Update the livein value, since the treenode it is now visited. Add both used and appended data
			livein |= ((get<1>(*it))->get_used_labels () | (get<1>(*it))->get_appended_labels ());
			livein[label_bitset_index[(get<0>(*it))->get_lhs_label()]] = true;
			addArrays (livein_freq, (get<1>(*it))->get_use_frequency(), label_count);
			addArrays (livein_freq, (get<1>(*it))->get_appended_frequency(), label_count);
			livein_freq[label_bitset_index[(get<0>(*it))->get_lhs_label()]] += 1;
			for (int i=0; i<label_count; i++) {
				if (livein[i] && livein_freq[i] == label_frequency[i]) livein[i] = false;
			}
			if (DEBUG) {
				printf ("live-out after the intra-splice additions at node (%s - %s) is ", lhs_label.c_str(), rhs_string.c_str());
				print_bitset (label_bitset_index, livein, label_count);
			}
			if (tree_changed) break;
		}
	} while (tree_changed);
	// Free resources
	expr_lhs_map.clear ();
	cull_labels.clear ();
}


bool funcdefn::determine_intra_opt_profitability (vector<tuple<treenode*,accnode*>> &computations, vector<int> &opt_vec, boost::dynamic_bitset<> &t_livein, unsigned int *t_livein_freq, map<string,tuple<string,expr_node*>> &expr_lhs_map, boost::dynamic_bitset<>grown_bitset, boost::dynamic_bitset<> single_use_labels) {
	boost::dynamic_bitset<> live_range_removed (label_count);
	boost::dynamic_bitset<> live_range_added (label_count);
	bool removed = false;
	do {
		unsigned int *update_freq = new unsigned int[label_count] ();
		copyArray (update_freq, t_livein_freq, label_count);
		if (DEBUG) {
			printf ("live-in freq after executing source is ");
			print_frequency (label_bitset_index, t_livein_freq, label_count);
		}
		// Update update_freq assuming that the optimizable subtrees are executed
		for (vector<int>::iterator jt=opt_vec.begin(); jt!=opt_vec.end(); jt++) {
			addArrays (update_freq, (get<1>(computations[*jt]))->get_use_frequency(), label_count);
			update_freq[label_bitset_index[get<0>(computations[*jt])->get_lhs_label()]] += 1;
		}
		// Now compute the live ranges removed (not removed by default as a virtue of being single_use_label)
		live_range_removed.reset ();
		for (int i=0; i<label_count; i++) {
			if ((t_livein[i] || grown_bitset[i]) && update_freq[i] == label_frequency[i] && !single_use_labels[i])
				live_range_removed[i] = true;
		}
		delete[] update_freq;
		if (DEBUG) {
			printf ("Temporary live range removed = %lu\n", live_range_removed.count ());
			print_bitset (label_bitset_index, live_range_removed, label_count);
		}
		removed = false;
		// Now remove all those computations from opt_vec that use non-removed values that are not live-in 
		vector<int> rem_vec;
		for (vector<int>::iterator jt=opt_vec.begin(); jt!=opt_vec.end(); jt++) {
			boost::dynamic_bitset<> used_labels = (get<1>(computations[*jt]))->get_used_labels ();
			boost::dynamic_bitset<> a_bitset (label_count);
			a_bitset = used_labels & ~(live_range_removed | single_use_labels | t_livein);
			if (a_bitset.any ())
				rem_vec.push_back (*jt);
		}
		if (DEBUG) printf ("opt size = %d, rem size = %d\n", (int)opt_vec.size(), (int)rem_vec.size());
		for (vector<int>::iterator jt=rem_vec.begin(); jt!=rem_vec.end(); jt++) {
			vector<int>::iterator kt = find (opt_vec.begin(), opt_vec.end(), *jt);
			opt_vec.erase (kt);
			removed = true;
		}
	} while (removed);
	if (DEBUG) {
		printf ("Live range removed = %lu\n", live_range_removed.count ());
		print_bitset (label_bitset_index, live_range_removed, label_count);
	}
	// Compute the live ranges that are added, but only if some live ranges are removed
	if (live_range_removed.any ()) {
		map<string,string> tmp_expr_lhs_map;
		for (vector<int>::iterator jt=opt_vec.begin(); jt!=opt_vec.end(); jt++) {
			string lhs_expr = (get<0>(computations[*jt]))->get_lhs_label ();
			string rhs_expr = (get<1>(computations[*jt]))->get_expr_string ();
			if (get<0>(computations[*jt])->is_accumulation_node ()) { 
				int idx = label_bitset_index[lhs_expr];
				if (t_livein[idx] == false) live_range_added[idx] = true;
			}
			else {
				if (expr_lhs_map.find (rhs_expr) == expr_lhs_map.end () && tmp_expr_lhs_map.find (rhs_expr) == tmp_expr_lhs_map.end ()) {
					if (get<1>(computations[*jt])->is_asgn_eq_op ()) 
						tmp_expr_lhs_map[rhs_expr] = lhs_expr;
					int idx = label_bitset_index[lhs_expr];
					if (t_livein[idx] == false) live_range_added[idx] = true;
				}
				else {
					int idx = (tmp_expr_lhs_map.find (rhs_expr) == tmp_expr_lhs_map.end ()) ? label_bitset_index[get<0>(expr_lhs_map[rhs_expr])] : label_bitset_index[tmp_expr_lhs_map[rhs_expr]];
					if (t_livein[idx] == false) live_range_added[idx] = true;
				}
			}
		}
	}
	if (DEBUG) {
		printf ("Live range added = %lu\n", live_range_added.count());
		print_bitset (label_bitset_index, live_range_added, label_count);
	}
	if (live_range_removed.count () == 0 && live_range_added.count () == 0) 
		return false;
	return (SPLICE_EQUALITY ? (live_range_removed.count () >= live_range_added.count ()) : (live_range_removed.count () > live_range_added.count ()));
}

// Print the forests
void funcdefn::print_forests (stringstream &output, vector<int> tree_sequence, boost::dynamic_bitset<> leaf_labels, boost::dynamic_bitset<>single_use_labels) {
	for (vector<int>::iterator it=tree_sequence.begin(); it!=tree_sequence.end(); it++) {
		vector<treenode*> subtree_vec;
		if (stmt_forest[*it].size () > 1) {
			// put the use label and tree pointer in the map
			for (vector<treenode*>::iterator jt=next(stmt_forest[*it].begin()); jt!=stmt_forest[*it].end(); jt++) {
				if (DEBUG) assert (!(*jt)->is_accumulation_node() && "Cannot have accumulation as a subtree (print_forests)");
				subtree_vec.push_back (*jt);
			}
		}
		stringstream tree_output;
		tree_output << (stmt_forest[*it].front())->print_tree (subtree_vec, leaf_labels, single_use_labels, label_count) << endl;
		vector<treenode*> tmp_subtree_vec;
		for (vector<treenode*>::iterator jt=subtree_vec.begin(); jt!=subtree_vec.end(); jt++) 
			tree_output << (*jt)->print_tree (tmp_subtree_vec, leaf_labels, single_use_labels, label_count) << endl;
		cout << tree_output.str ();
		output << tree_output.str ();
		//if (DEBUG) assert (subtree_vec.empty () && "Subtree not empty");
		subtree_vec.clear ();
	}
}

void funcdefn::copy_propagation (std::vector<int> &tree_sequence) {
	for (vector<int>::iterator it=tree_sequence.begin(); it!=tree_sequence.end(); it++) {
		for (vector<treenode*>::iterator jt=stmt_forest[*it].begin(); jt!=stmt_forest[*it].end(); jt++) {
			map<string, treenode*> asgn_map;	 
			(*jt)->copy_propagation (asgn_map, label_frequency, label_bitset_index);
			vector<string> cull_labels; 
			(*jt)->recompute_tree (label_bitset_index, cull_labels, label_count);
		}
	}
}

void funcdefn::print_forests (stringstream &output, vector<int> tree_sequence) {
	stringstream finalized_tree, regalloc_tree;
	queue<int> avail_regs;
	map<string,int> alloc_map;
	unsigned int *label_use = new unsigned int[label_count] ();
	int reg_count = 0;

	vector<expr_node*> init;
	for (auto in : initial_assignments) {
		init.push_back (in->get_lhs_expr ());
	}
	map<string,string> lhs_init;
	for (vector<int>::iterator it=tree_sequence.begin(); it!=tree_sequence.end(); it++) {
		//stringstream tree_output;
		treenode *tree = stmt_forest[*it].front();
		//tree_output << tree->print_tree () << endl;
		tree->print_finalized_tree (finalized_tree, temp_vars, init, lhs_init, gdata_type);
		//tree->allocate_registers (regalloc_tree, reg_count, avail_regs, alloc_map, label_frequency, label_use, label_bitset_index);
		finalized_tree << endl;
		//cout << tree_output.str ();
		//output << tree_output.str ();
	}
	for (auto s : lhs_init) {
		finalized_tree << s.second << " = " << s.first << ";\n";
	}
	//cout << finalized_tree.str ();
	output << finalized_tree.str ();
	//output << "\n//CLUSTER AFTER REGISTER ALLOCATION (" << reg_count << ")\n";
	//output << regalloc_tree.str ();
}

//// Incorrect: Tries to reduce the number of inter optimizations tried by going backwards
//void funcdefn::fixed_order_inter_forest_optimizations (vector<int> tree_sequence, map<int, tuple<boost::dynamic_bitset<>,unsigned int*>> liveness_map, boost::dynamic_bitset<> single_use_labels) {
//	// Try all combinations to perform AV + CSE optimization
//	vector<int>::iterator it=prev(prev(tree_sequence.end()));
//	bool optimize = true;
//	while (optimize) {
//		if (it == tree_sequence.begin()) optimize = false;
//		// Create a map of expr to lhs for host
//		map<string,tuple<string,expr_node*>> expr_lhs_map;
//		for (vector<treenode*>::iterator jt=stmt_forest[*it].begin(); jt!=stmt_forest[*it].end(); jt++) 
//			(*jt)->create_expr_lhs_map (expr_lhs_map);
//		// Try all the sequences
//		vector<int>::iterator jt = tree_sequence.end ();
//		while (jt != next(it)) {
//			if (DEBUG) printf ("Trying %d to %d\n", *it, *(prev(jt)));
//			// Find the live labels out of it
//			boost::dynamic_bitset<> intersection (label_count);
//			intersection = get<0>(liveness_map[*it]);
//			// Subtract all the labels live from jt onwards
//			for (vector<int>::iterator kt=jt; kt!=tree_sequence.end(); kt++) {
//				treenode * &excluded_tree = (stmt_forest[*kt]).front ();
//				intersection &= ~(excluded_tree->get_used_labels ());
//			}
//			if (DEBUG) {
//				printf ("Intersection is ");
//				print_bitset (label_bitset_index, intersection, label_count);	
//			}
//			// Only proceed further if there are more than one common leafs in intersection
//			if (intersection.count () > 1) {
//				// Now with intersection in arm, we need to find the common subtree amongst 
//				// all the nodes such that the subtree usage is a subset of intersection
//				map<int, vector<tuple<treenode*,accnode*>>> optimizable_subtrees;
//				for (vector<int>::iterator kt=next(it); kt!=jt; kt++) {
//					treenode * &included_tree = (stmt_forest[*kt]).front ();
//					if (SPLICE_TEMP_LABELS)
//						included_tree->identify_optimizable_subtrees (optimizable_subtrees[*kt], intersection, single_use_labels, label_frequency, label_count);
//					else
//						included_tree->identify_optimizable_subtrees (optimizable_subtrees[*kt], intersection, single_use_labels, label_count);
//				}
//				// Perform AV + CSE
//				int subtree_count = 0;
//				for (map<int, vector<tuple<treenode*,accnode*>>>::iterator kt=optimizable_subtrees.begin(); kt!=optimizable_subtrees.end(); kt++) 
//					subtree_count += (int)((kt->second).size ());
//				if (subtree_count > 0)
//					optimize_available_expressions (*it, tree_sequence, optimizable_subtrees, liveness_map, expr_lhs_map, single_use_labels);
//			}
//			jt = prev (jt);
//		}
//		it = prev (it);
//	}
//}

// Try all sequences to find inter-optimization opportunity. This allows moving subtrees that comprise only temp labels.  
void funcdefn::fixed_order_inter_forest_optimizations (vector<int> tree_sequence, map<int, tuple<boost::dynamic_bitset<>,unsigned int*>> liveness_map, boost::dynamic_bitset<> single_use_labels) {
	// Try all combinations to perform AV + CSE optimization
	for (vector<int>::iterator it=tree_sequence.begin(); it!=prev(tree_sequence.end()); it++) {
		// Create a map of expr to lhs for host
		map<string,tuple<string,expr_node*>> expr_lhs_map;
		for (vector<treenode*>::iterator jt=stmt_forest[*it].begin(); jt!=stmt_forest[*it].end(); jt++)
			(*jt)->create_expr_lhs_map (expr_lhs_map);
		// Try all the sequences
		vector<int>::iterator jt = tree_sequence.end ();
		while (jt != next(it)) {
			if (DEBUG) printf ("Trying %d to %d\n", *it, *(prev(jt)));
			// Find the live labels out of it
			boost::dynamic_bitset<> intersection (label_count);
			intersection = get<0>(liveness_map[*it]);
			// Subtract all the labels live from jt onwards
			for (vector<int>::iterator kt=jt; kt!=tree_sequence.end(); kt++) {
				treenode * &excluded_tree = (stmt_forest[*kt]).front ();
				intersection &= ~(excluded_tree->get_used_labels ());
			}
			if (DEBUG) {
				printf ("Intersection is ");
				print_bitset (label_bitset_index, intersection, label_count);	
			}
			// Only proceed further if there are more than one common leafs in intersection
			if (intersection.count () > 1) {
				// Now with intersection in arm, we need to find the common subtree amongst 
				// all the nodes such that the subtree usage is a subset of intersection
				map<int, vector<tuple<treenode*,accnode*>>> optimizable_subtrees;
				for (vector<int>::iterator kt=next(it); kt!=jt; kt++) {
					treenode * &included_tree = (stmt_forest[*kt]).front ();
					if (SPLICE_TEMP_LABELS) 
						included_tree->identify_optimizable_subtrees (optimizable_subtrees[*kt], intersection, single_use_labels, label_frequency, label_count);
					else 
						included_tree->identify_optimizable_subtrees (optimizable_subtrees[*kt], intersection, single_use_labels, label_count);
				}
				// Perform AV + CSE
				int subtree_count = 0;
				for (map<int, vector<tuple<treenode*,accnode*>>>::iterator kt=optimizable_subtrees.begin(); kt!=optimizable_subtrees.end(); kt++) 
					subtree_count += (int)((kt->second).size ());
				if (subtree_count > 0)
					optimize_available_expressions (*it, tree_sequence, optimizable_subtrees, liveness_map, expr_lhs_map, single_use_labels);
			}
			jt = prev (jt);
		}
		expr_lhs_map.clear ();
	}
}

// Same as above, but restricts moving subtrees if they don't have leaf labels
void funcdefn::fixed_order_inter_forest_optimizations (vector<int> tree_sequence, map<int, tuple<boost::dynamic_bitset<>,unsigned int*>> liveness_map, boost::dynamic_bitset<> single_use_labels, boost::dynamic_bitset<> leaf_labels) {
	// Try all combinations to perform AV + CSE optimization
	for (vector<int>::iterator it=tree_sequence.begin(); it!=prev(tree_sequence.end()); it++) {
		// Create a map of expr to lhs for host
		map<string,tuple<string,expr_node*>> expr_lhs_map;
		for (vector<treenode*>::iterator jt=stmt_forest[*it].begin(); jt!=stmt_forest[*it].end(); jt++)
			(*jt)->create_expr_lhs_map (expr_lhs_map);
		// Try all the sequences
		vector<int>::iterator jt = tree_sequence.end ();
		while (jt != next(it)) {
			if (DEBUG) printf ("Trying %d to %d\n", *it, *(prev(jt)));
			// Find the live labels out of it
			boost::dynamic_bitset<> intersection (label_count);
			intersection = get<0>(liveness_map[*it]);
			// Subtract all the labels live from jt onwards
			for (vector<int>::iterator kt=jt; kt!=tree_sequence.end(); kt++) {
				treenode * &excluded_tree = (stmt_forest[*kt]).front ();
				intersection &= ~(excluded_tree->get_used_labels ());
			}
			if (DEBUG) {
				printf ("Intersection is ");
				print_bitset (label_bitset_index, intersection, label_count);	
			}
			// Only proceed further if there are more than one common leafs in intersection
			if (intersection.count () > 1) {
				// Now with intersection in arm, we need to find the common subtree amongst 
				// all the nodes such that the subtree usage is a subset of intersection
				map<int, vector<tuple<treenode*,accnode*>>> optimizable_subtrees;
				for (vector<int>::iterator kt=next(it); kt!=jt; kt++) {
					treenode * &included_tree = (stmt_forest[*kt]).front ();
					if (SPLICE_TEMP_LABELS)
						included_tree->identify_optimizable_subtrees (optimizable_subtrees[*kt], intersection, single_use_labels, leaf_labels, label_frequency, label_count);
					else 
						included_tree->identify_optimizable_subtrees (optimizable_subtrees[*kt], intersection, single_use_labels, leaf_labels, label_count);
				}
				// Perform AV + CSE
				int subtree_count = 0;
				for (map<int, vector<tuple<treenode*,accnode*>>>::iterator kt=optimizable_subtrees.begin(); kt!=optimizable_subtrees.end(); kt++) 
					subtree_count += (int)((kt->second).size ());
				if (subtree_count > 0)
					optimize_available_expressions (*it, tree_sequence, optimizable_subtrees, liveness_map, expr_lhs_map, single_use_labels);
			}
			jt = prev (jt);
		}
		expr_lhs_map.clear ();
	}
}

// Post PLDI: Intra type inter-forest-optimization
void funcdefn::fixed_order_intra_type_inter_forest_optimizations (vector<int> tree_sequence, map<int, tuple<boost::dynamic_bitset<>,unsigned int*>> liveness_map, boost::dynamic_bitset<> single_use_labels, boost::dynamic_bitset<> leaf_labels) {
	// Try all combinations to perform AV + CSE optimization
	for (vector<int>::iterator it=tree_sequence.begin(); it!=prev(tree_sequence.end()); it++) {
		// Create a map of expr to lhs for host
		map<string,tuple<string,expr_node*>> expr_lhs_map;
		for (vector<treenode*>::iterator jt=stmt_forest[*it].begin(); jt!=stmt_forest[*it].end(); jt++)
			(*jt)->create_expr_lhs_map (expr_lhs_map);
		// Try all the sequences
		vector<int>::iterator jt = tree_sequence.end ();
		while (jt != next(it)) {
			if (DEBUG) printf ("Trying %d to %d\n", *it, *(prev(jt)));
			// Find the live labels out of it
			boost::dynamic_bitset<> intersection (label_count);
			intersection = get<0>(liveness_map[*it]);
			// Subtract all the labels live from jt onwards
			for (vector<int>::iterator kt=jt; kt!=tree_sequence.end(); kt++) {
				treenode * &excluded_tree = (stmt_forest[*kt]).front ();
				intersection &= ~(excluded_tree->get_used_labels ());
			}
			if (DEBUG) {
				printf ("Intersection is ");
				print_bitset (label_bitset_index, intersection, label_count);	
			}
			// Only proceed further if there are more than one common leafs in intersection
			if (intersection.count () > 1) {
				// Now with intersection in arm, we need to find the common subtree amongst 
				// all the nodes such that the subtree usage is a subset of intersection
				map<int, vector<tuple<treenode*,accnode*>>> optimizable_subtrees;
				for (vector<int>::iterator kt=next(it); kt!=jt; kt++) {
					treenode * &included_tree = (stmt_forest[*kt]).front ();
					vector<tuple<treenode*,accnode*>> computations;
					included_tree->identify_leafy_computation_subtrees (computations, leaf_labels, label_bitset_index, label_count);
					included_tree->identify_optimizable_leafy_subtrees (optimizable_subtrees[*kt], computations, intersection, label_bitset_index, label_count);
				}
				// Perform AV + CSE
				int subtree_count = 0;
				for (map<int, vector<tuple<treenode*,accnode*>>>::iterator kt=optimizable_subtrees.begin(); kt!=optimizable_subtrees.end(); kt++) 
					subtree_count += (int)((kt->second).size ());
				if (subtree_count > 0)
					optimize_intra_type_available_expressions (*it, tree_sequence, optimizable_subtrees, liveness_map, expr_lhs_map, single_use_labels);
			}
			jt = prev (jt);
		}
		expr_lhs_map.clear ();
	}
}

// This first tries to put everything in the host node if there is a reduction in the number of live 
// values going out of the cluster. After optimization, recompute the labelset and livemap of the touched trees.
void funcdefn::optimize_available_expressions (int host_cluster, vector<int> tree_sequence, map<int, vector<tuple<treenode*,accnode*>>> &optimizable_subtrees, map<int, tuple<boost::dynamic_bitset<>,unsigned int*>> &liveness_map, map<string, tuple<string,expr_node*>> &expr_lhs_map, boost::dynamic_bitset<> single_use_labels) {
	bool removed = false;
	boost::dynamic_bitset<> host_live_labels = get<0>(liveness_map[host_cluster]);
	boost::dynamic_bitset<> live_range_removed (label_count);
	boost::dynamic_bitset<> live_range_added (label_count);
	do {
		live_range_removed.reset ();
		//live_range_removed.resize (label_count);
		unsigned int *use_frequency = new unsigned int[label_count] ();
		copyArray (use_frequency, get<1>(liveness_map[host_cluster]), label_count);
		// Increment use_frequency count for all the subtrees in optimizable_subtrees
		for (map<int, vector<tuple<treenode*,accnode*>>>::iterator it=optimizable_subtrees.begin(); it!=optimizable_subtrees.end(); it++) {
			int cluster_num = it->first;
			if (cluster_num == host_cluster) continue;
			vector<tuple<treenode*,accnode*>> subtrees = it->second;
			for (vector<tuple<treenode*,accnode*>>::iterator jt=subtrees.begin(); jt!=subtrees.end(); jt++) {
				accnode *rhs_expr = get<1>(*jt);
				addArrays (use_frequency, rhs_expr->get_use_frequency(), label_count);
				string lhs_label = get<0>(*jt)->get_lhs_label ();
				use_frequency[label_bitset_index[lhs_label]] += 1;
			}
		}
		// Count the number of uses that have reached their limit. Discount labels that weren't live-out in the host cluster
		for (int i=0; i<label_count; i++) {
			if (host_live_labels[i] && label_frequency[i] == use_frequency[i] && !single_use_labels[i]) 
				live_range_removed[i] = true;
		}
		delete[] use_frequency;
		if (DEBUG) {
			printf ("Temporary live range removed = %lu\n", live_range_removed.count ());
			print_bitset (label_bitset_index, live_range_removed, label_count);
		}
		removed = false;
		// Remove all those subtrees with rhs_expr having a label whose live range is not removed
		for (map<int, vector<tuple<treenode*,accnode*>>>::iterator it=optimizable_subtrees.begin(); it!=optimizable_subtrees.end(); it++) {
			int cluster_num = it->first;
			if (cluster_num == host_cluster) continue;
			vector<tuple<treenode*,accnode*>> &subtrees = it->second;
			vector<tuple<treenode*,accnode*>> rem_vec;
			for (vector<tuple<treenode*,accnode*>>::iterator jt=subtrees.begin(); jt!=subtrees.end(); jt++) {
				accnode *rhs_expr = get<1>(*jt);
				boost::dynamic_bitset<> a_bitset (label_count);
				if (SPLICE_TEMP_LABELS) {
					boost::dynamic_bitset<> temp_labels (label_count);
					unsigned int *u_freq = rhs_expr->get_use_frequency ();
					for (int i=0; i<label_count; i++) {
						if (u_freq[i] == label_frequency[i]) temp_labels[i] = true;
					}
					a_bitset = rhs_expr->get_used_labels () & ~(live_range_removed | single_use_labels | temp_labels | host_live_labels);
				}
				else 
					a_bitset = rhs_expr->get_used_labels () & ~(live_range_removed | single_use_labels | host_live_labels);
				if (a_bitset.any ())
					rem_vec.push_back (*jt);
			}
			for (vector<tuple<treenode*,accnode*>>::iterator jt=rem_vec.begin(); jt!=rem_vec.end(); jt++) {
				vector<tuple<treenode*,accnode*>>::iterator kt = find ((it->second).begin(), (it->second).end(), *jt);
				(it->second).erase (kt);
				removed = true;
			}
		}
	} while (removed);
	host_live_labels.clear ();
	if (DEBUG) {
		printf ("Live range removed = %lu\n", live_range_removed.count ());
		print_bitset (label_bitset_index, live_range_removed, label_count); 
	}
	// Now count the values made live for other subtrees
	if (live_range_removed.any ()) {
		// Compute the live-out values till now
		boost::dynamic_bitset<> t_livein (label_count);
		t_livein = get<0>(liveness_map[host_cluster]) & ~live_range_removed;
		// Compute the live ranges added
		map<string,string> tmp_expr_lhs_map;
		for (map<int, vector<tuple<treenode*,accnode*>>>::iterator it=optimizable_subtrees.begin(); it!=optimizable_subtrees.end(); it++) {
			int cluster_num = it->first;
			if (cluster_num == host_cluster) continue;
			vector<tuple<treenode*,accnode*>> subtrees = it->second;
			for (vector<tuple<treenode*,accnode*>>::iterator jt=subtrees.begin(); jt!=subtrees.end(); jt++) {
				string rhs_expr = (get<1>(*jt))->get_expr_string ();
				string lhs_expr = (get<0>(*jt))->get_lhs_label ();
				bool is_accumulation_node = (get<0>(*jt))->is_accumulation_node ();
				if (is_accumulation_node) {
					int idx = label_bitset_index[lhs_expr]; 
					if (t_livein[idx] == false) live_range_added[idx] = true;
				}
				else {
					if (expr_lhs_map.find (rhs_expr) == expr_lhs_map.end () && tmp_expr_lhs_map.find (rhs_expr) == tmp_expr_lhs_map.end ()) {
						if (get<1>(*jt)->is_asgn_eq_op ())
							tmp_expr_lhs_map[rhs_expr] = lhs_expr;
						int idx = label_bitset_index[lhs_expr]; 
						if (t_livein[idx] == false) live_range_added[idx] = true;
					}
					else {
						int idx = (tmp_expr_lhs_map.find (rhs_expr) == tmp_expr_lhs_map.end ()) ? label_bitset_index[get<0>(expr_lhs_map[rhs_expr])] : label_bitset_index[tmp_expr_lhs_map[rhs_expr]];
						if (t_livein[idx] == false) live_range_added[idx] = true;
					}
				}
			}
		}
	}
	if (DEBUG) {
		printf ("Live range added = %lu\n", live_range_added.count());
		print_bitset (label_bitset_index, live_range_added, label_count); 
	}
	// If more live ranges are removed than added, update the clusters
	if (!(live_range_removed.count () == 0 && live_range_added.count () == 0)) {
		if ((SPLICE_EQUALITY && (live_range_removed.count () >= live_range_added.count ())) || (!SPLICE_EQUALITY && (live_range_removed.count () > live_range_added.count ()))) {
			update_forests (host_cluster, optimizable_subtrees, expr_lhs_map);
			compute_liveness_map (liveness_map, tree_sequence);
		}
	}
}

// Add all the subtrees in optimizable_subtrees to host cluster
void funcdefn::update_forests (int host_cluster, map<int, vector<tuple<treenode*,accnode*>>> &optimizable_subtrees, map<string, tuple<string,expr_node*>> &expr_lhs_map) {
	if (DEBUG) printf ("In update_forests for host cluster %d\n", host_cluster);
	vector<treenode*> &host_tree = stmt_forest[host_cluster];
	map<int, vector<string>> cull_labels;
	for (map<int, vector<tuple<treenode*,accnode*>>>::iterator it=optimizable_subtrees.begin(); it!=optimizable_subtrees.end(); it++) {
		int cluster_num = it->first;
		vector<string> t_vec;
		cull_labels[cluster_num] = t_vec;
		if (cluster_num == host_cluster) continue;
		// Put all the subtrees in the host cluster
		vector<tuple<treenode*,accnode*>> &subtrees = it->second;
		for (vector<tuple<treenode*,accnode*>>::iterator jt=subtrees.begin(); jt!=subtrees.end(); jt++) {
			treenode * &subtree = get<0>(*jt);
			accnode * &rhs_expr = get<1>(*jt);
			string expr_string = rhs_expr->get_expr_string ();
			// Look for the leading lhs_expr for the rhs expr_string
			string lhs_rep = subtree->get_lhs_label ();
			if (DEBUG) printf ("Slicing tree %s : %s\n", lhs_rep.c_str(), expr_string.c_str());
			bool not_found = true;
			if (!subtree->is_accumulation_node ()) {
				if (expr_lhs_map.find (expr_string) != expr_lhs_map.end ()) {
					lhs_rep = get<0>(expr_lhs_map[expr_string]);
					// Reduce the label counts of rhs_expr from necessary bitsets (label_frequency)
					subtractArrays (label_frequency, rhs_expr->get_use_frequency(), label_count);
					if (DEBUG) {
						unsigned int *t_freq = rhs_expr->get_use_frequency ();	
						for (int i=0; i<label_count; i++) 
							if (t_freq[i] != 0) printf ("Reduced the label frequency[%d] from %d to %d\n", i, label_frequency[i]+t_freq[i], label_frequency[i]);
					}
					// Increment the label count for the LHS
					label_frequency[label_bitset_index[lhs_rep]] += 1;
					if (DEBUG) printf ("increased the label frequency of %s from %d to %d\n", lhs_rep.c_str(), label_frequency[label_bitset_index[lhs_rep]]-1, label_frequency[label_bitset_index[lhs_rep]]);
					not_found = false;
				}
				else { 
					if (rhs_expr->is_asgn_eq_op ()) 
						expr_lhs_map[expr_string] = make_tuple (lhs_rep, subtree->get_lhs());
					// lhs label must be culled
					if (find (cull_labels[cluster_num].begin(), cull_labels[cluster_num].end(), lhs_rep) == cull_labels[cluster_num].end ()) 
						cull_labels[cluster_num].push_back (lhs_rep);
				}
			}
			if (not_found || subtree->is_accumulation_node ()) {
				stringstream lhs_print;
				(subtree->get_lhs())->print_node (lhs_print);
				treenode *t_node = new treenode (subtree->get_lhs (), lhs_print.str(), lhs_rep, label_bitset_index, label_count, false);
				t_node->add_rhs_expr (rhs_expr, label_bitset_index, label_count);
				host_tree.push_back (t_node);
			}
			// Modify the trees from which the node was sliced
			if (!subtree->is_accumulation_node ()) {
				if (not_found) subtree->reset_rhs_accs ();
				else {
					stringstream lhs_print;
					(get<1>(expr_lhs_map[expr_string]))->print_node (lhs_print);
					treenode *t_node = new treenode (get<1>(expr_lhs_map[expr_string]), lhs_print.str(), lhs_rep, label_bitset_index, label_count, false);
					accnode *old_rhs = (subtree->get_rhs_operands ()).front();
					accnode *new_rhs = new accnode (label_count, lhs_rep, t_node, old_rhs->get_assignment_op());
					subtree->reset_rhs_accs (new_rhs);
				}
			}
			else {
				vector<accnode*> &t_rhs_accs = subtree->get_rhs_operands ();
				int pos = 0;
				for (vector<accnode*>::iterator kt=t_rhs_accs.begin(); kt!=t_rhs_accs.end(); kt++, pos++) {
					if (expr_string.compare ((*kt)->get_expr_string()) == 0) 
						break; 
				}
				if (DEBUG) assert (pos < t_rhs_accs.size () && "Element to be sliced not found in the rhs operands (update_forests)");
				t_rhs_accs.erase (t_rhs_accs.begin()+pos);
			}
		}
	}
	// Recompute the modified trees
	for (map<int, vector<tuple<treenode*,accnode*>>>::iterator it=optimizable_subtrees.begin(); it!=optimizable_subtrees.end(); it++) {
		int cluster_num = it->first;
		treenode * &mod_tree = (stmt_forest[cluster_num]).front();
		mod_tree->recompute_tree (label_bitset_index, cull_labels[cluster_num], label_count);
	}
	// Since some subtrees are added to the host cluster, update the host costs as well
	treenode * &target_tree = host_tree.front ();
	vector<string> temp_cull_labels;
	target_tree->recompute_tree (label_bitset_index, temp_cull_labels, label_count);
	if (DEBUG) printf ("for %d, modified tree size = %d\n", host_cluster, (int)host_tree.size());
	for (vector<treenode*>::iterator it=next(host_tree.begin()); it!=host_tree.end(); it++)
		target_tree->update_host_tree (*it, label_count);
	if (DEBUG) {
		printf ("Printing use frequency for %s\n", target_tree->get_lhs_label().c_str());
		print_frequency (label_bitset_index, target_tree->get_use_frequency (), label_count);	
	}
}

// Do an intra-like maximal splicing 
void funcdefn::optimize_intra_type_available_expressions (int host_cluster, vector<int> tree_sequence, map<int, vector<tuple<treenode*,accnode*>>> &optimizable_subtrees, map<int, tuple<boost::dynamic_bitset<>,unsigned int*>> &liveness_map, map<string, tuple<string,expr_node*>> &expr_lhs_map, boost::dynamic_bitset<> single_use_labels) {
	bool removed = false;
	boost::dynamic_bitset<> host_live_labels = get<0>(liveness_map[host_cluster]);
	boost::dynamic_bitset<> live_range_removed (label_count);
	boost::dynamic_bitset<> live_range_added (label_count);
	do {
		live_range_removed.reset ();
		//live_range_removed.resize (label_count);
		unsigned int *use_frequency = new unsigned int[label_count] ();
		copyArray (use_frequency, get<1>(liveness_map[host_cluster]), label_count);
		// Increment use_frequency count for all the subtrees in optimizable_subtrees
		for (map<int, vector<tuple<treenode*,accnode*>>>::iterator it=optimizable_subtrees.begin(); it!=optimizable_subtrees.end(); it++) {
			int cluster_num = it->first;
			if (cluster_num == host_cluster) continue;
			vector<tuple<treenode*,accnode*>> subtrees = it->second;
			for (vector<tuple<treenode*,accnode*>>::iterator jt=subtrees.begin(); jt!=subtrees.end(); jt++) {
				accnode *rhs_expr = get<1>(*jt);
				addArrays (use_frequency, rhs_expr->get_use_frequency(), label_count);
				string lhs_label = get<0>(*jt)->get_lhs_label ();
				use_frequency[label_bitset_index[lhs_label]] += 1;
			}
		}
		// Count the number of uses that have reached their limit. Discount labels that weren't live-out in the host cluster
		for (int i=0; i<label_count; i++) {
			if (host_live_labels[i] && label_frequency[i] == use_frequency[i] && !single_use_labels[i]) 
				live_range_removed[i] = true;
		}
		delete[] use_frequency;
		if (DEBUG) {
			printf ("Temporary live range removed = %lu\n", live_range_removed.count ());
			print_bitset (label_bitset_index, live_range_removed, label_count);
		}
		removed = false;
		// Remove all those subtrees with rhs_expr having a label whose live range is not removed
		for (map<int, vector<tuple<treenode*,accnode*>>>::iterator it=optimizable_subtrees.begin(); it!=optimizable_subtrees.end(); it++) {
			int cluster_num = it->first;
			if (cluster_num == host_cluster) continue;
			vector<tuple<treenode*,accnode*>> rem_vec;
			vector<tuple<treenode*,accnode*>> &subtrees = it->second;
			for (vector<tuple<treenode*,accnode*>>::iterator jt=subtrees.begin(); jt!=subtrees.end(); jt++) {
				accnode *rhs_expr = get<1>(*jt);
				boost::dynamic_bitset<> a_bitset (label_count);
				a_bitset = rhs_expr->get_used_labels () & ~(live_range_removed | single_use_labels | host_live_labels);
				if (a_bitset.any ())
					rem_vec.push_back (*jt);
			}
			for (vector<tuple<treenode*,accnode*>>::iterator jt=rem_vec.begin(); jt!=rem_vec.end(); jt++) {
				vector<tuple<treenode*,accnode*>>::iterator kt = find ((it->second).begin(), (it->second).end(), *jt);
				(it->second).erase (kt);
				removed = true;
			}
		}
	} while (removed);
	host_live_labels.clear ();
	if (DEBUG) {
		printf ("Live range removed = %lu\n", live_range_removed.count ());
		print_bitset (label_bitset_index, live_range_removed, label_count); 
	}
	// Now count the values made live for other subtrees
	if (live_range_removed.any ()) {
		// Compute the live-out values till now
		boost::dynamic_bitset<> t_livein (label_count);
		t_livein = get<0>(liveness_map[host_cluster]) & ~live_range_removed;
		// Compute the live ranges added
		map<string,string> tmp_expr_lhs_map;
		for (map<int, vector<tuple<treenode*,accnode*>>>::iterator it=optimizable_subtrees.begin(); it!=optimizable_subtrees.end(); it++) {
			int cluster_num = it->first;
			if (cluster_num == host_cluster) continue;
			vector<tuple<treenode*,accnode*>> subtrees = it->second;
			for (vector<tuple<treenode*,accnode*>>::iterator jt=subtrees.begin(); jt!=subtrees.end(); jt++) {
				string rhs_expr = (get<1>(*jt))->get_expr_string ();
				string lhs_expr = (get<0>(*jt))->get_lhs_label ();
				bool is_accumulation_node = (get<0>(*jt))->is_accumulation_node ();
				if (is_accumulation_node) {
					int idx = label_bitset_index[lhs_expr]; 
					if (t_livein[idx] == false) live_range_added[idx] = true;
				}
				else {
					if (expr_lhs_map.find (rhs_expr) == expr_lhs_map.end () && tmp_expr_lhs_map.find (rhs_expr) == tmp_expr_lhs_map.end ()) {
						if (get<1>(*jt)->is_asgn_eq_op ())
							tmp_expr_lhs_map[rhs_expr] = lhs_expr;
						int idx = label_bitset_index[lhs_expr]; 
						if (t_livein[idx] == false) live_range_added[idx] = true;
					}
					else {
						int idx = (tmp_expr_lhs_map.find (rhs_expr) == tmp_expr_lhs_map.end ()) ? label_bitset_index[get<0>(expr_lhs_map[rhs_expr])] : label_bitset_index[tmp_expr_lhs_map[rhs_expr]];
						if (t_livein[idx] == false) live_range_added[idx] = true;
					}
				}
			}
		}
	}
	if (DEBUG) {
		printf ("Live range added = %lu\n", live_range_added.count());
		print_bitset (label_bitset_index, live_range_added, label_count); 
	}
	// If more live ranges are removed than added, update the clusters
	if (!(live_range_removed.count () == 0 && live_range_added.count () == 0)) {
		if ((SPLICE_EQUALITY && (live_range_removed.count () >= live_range_added.count ())) || (!SPLICE_EQUALITY && (live_range_removed.count () > live_range_added.count ()))) {
			update_forests (host_cluster, optimizable_subtrees, expr_lhs_map);
			compute_liveness_map (liveness_map, tree_sequence);
		}
	}
}

void funcdefn::compute_cluster_dependences (void) {
	vector<stmtnode*> stmts = stmt_list->get_stmt_list ();
	vector<string> i_label;
	vector<string> j_label;
	bool dependence;
	// Clear the cluster dependence graph, start from scratch
	cluster_dependence_graph.clear ();
	for (vector<stmtnode*>::const_iterator i=stmts.begin(); i!=stmts.end(); i++) {
		int i_cluster_num = (*i)->get_orig_stmt_num ();
		// Go over all previous statements to figure out dependences. 
		for (vector<stmtnode*>::const_iterator j=stmts.begin(); j!=i; j++) {
			int j_cluster_num = (*j)->get_orig_stmt_num ();
			// Only compute dependences if the statements belong to different clusters
			if (i_cluster_num == j_cluster_num) continue;
			// 1. WAW dependence is only if either of them is an assignment
			dependence = false;

			bool init_asgn = (*i)->get_op_type() == ST_EQ || (*j)->get_op_type() == ST_EQ;
			for (vector<stmtnode*>::iterator it=initial_assignments.begin(); it!=initial_assignments.end(); it++) 
				init_asgn |= (((*it)->get_lhs_expr() == (*i)->get_lhs_expr()) || ((*it)->get_lhs_expr() == (*j)->get_lhs_expr()));
			if (init_asgn) {
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
					if (DEBUG) printf ("Found WAW dependence between source cluster %d and dest cluster %d\n", j_cluster_num, i_cluster_num);
					if (cluster_dependence_graph.find (i_cluster_num) == cluster_dependence_graph.end ()) {
						vector<int> dep_stmts;
						dep_stmts.push_back (j_cluster_num);
						cluster_dependence_graph[i_cluster_num] = dep_stmts;
					}
					else {
						if (find ((cluster_dependence_graph[i_cluster_num]).begin(), (cluster_dependence_graph[i_cluster_num]).end(), j_cluster_num) == (cluster_dependence_graph[i_cluster_num]).end())
							(cluster_dependence_graph[i_cluster_num]).push_back (j_cluster_num); 
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
			//i_label = (*i)->get_lhs_names ();
			//j_label = (*j)->get_rhs_names ();
			//for (vector<string>::iterator s1=i_label.begin(); s1!=i_label.end(); s1++) {
			//	for (vector<string>::iterator s2=j_label.begin(); s2!=j_label.end(); s2++) {
			//		if ((*s1).compare (*s2) == 0) {
			//			dependence = true;
			//			break;
			//		}
			//	}
			//}
			if (dependence) {
				if (DEBUG) printf ("Found WAR dependence between source cluster %d and dest cluster %d\n", j_cluster_num, i_cluster_num);
				if (cluster_dependence_graph.find (i_cluster_num) == cluster_dependence_graph.end ()) {
					vector<int> dep_stmts;
					dep_stmts.push_back (j_cluster_num);
					cluster_dependence_graph[i_cluster_num] = dep_stmts;
				}
				else {
					if (find ((cluster_dependence_graph[i_cluster_num]).begin(), (cluster_dependence_graph[i_cluster_num]).end(), j_cluster_num) == (cluster_dependence_graph[i_cluster_num]).end())
						(cluster_dependence_graph[i_cluster_num]).push_back (j_cluster_num); 
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
				if (DEBUG) printf ("Found RAW dependence between source cluster %d and dest cluster %d\n", j_cluster_num, i_cluster_num);
				if (cluster_dependence_graph.find (i_cluster_num) == cluster_dependence_graph.end ()) {
					vector<int> dep_stmts;
					dep_stmts.push_back (j_cluster_num);
					cluster_dependence_graph[i_cluster_num] = dep_stmts;
				}
				else {
					if (find ((cluster_dependence_graph[i_cluster_num]).begin(), (cluster_dependence_graph[i_cluster_num]).end(), j_cluster_num) == (cluster_dependence_graph[i_cluster_num]).end())
						(cluster_dependence_graph[i_cluster_num]).push_back (j_cluster_num); 
				}
			}
		}
	}
}

void funcdefn::print_cluster_dependence_graph (string name) {
	cout << "\nCluster dependence graph for function " << name << " : " << endl;
	for (map<int, vector<int>>::iterator j=cluster_dependence_graph.begin(); j!=cluster_dependence_graph.end(); j++) {
		int lhs = j->first;
		printf ("%d - ", lhs);
		vector<int> rhs = j->second;
		for (vector<int>::iterator k=rhs.begin(); k!=rhs.end(); k++) {
			printf ("%d ", *k);
		}
		printf ("\n");
	}
}
