#include "exprnode.hpp"
using namespace std;

DATA_TYPE expr_node::get_type (void) {
	return type;
}

void expr_node::set_name (string s) {
	name = s;
}

void expr_node::set_name (char *s) { 
	name = string (s); 
}

string expr_node::get_name (void) {
	return name;
}

bool expr_node::is_data_type (DATA_TYPE gdata_type) {
	return expr_type == T_DATATYPE;
}

bool expr_node::is_data_type (void) {
	return expr_type == T_DATATYPE;
}

bool expr_node::is_id_type (DATA_TYPE gdata_type) {
	if (expr_type == T_ID) {
		type = gdata_type;
		return true;
	}
	return false;
}

bool expr_node::is_id_type (void) {
	return expr_type == T_ID;
}

bool expr_node::simple_nondecomposable_expr (void) {
	return true;
}

bool expr_node::is_shiftvec_type (DATA_TYPE gdata_type) {
	if (expr_type == T_SHIFTVEC) {
		type = gdata_type;
		return true;
	}
	return false;
}

bool expr_node::is_shiftvec_type (void) {
	return expr_type == T_SHIFTVEC;
}

void id_node::print_node (stringstream &out) {
	// out << name;
	out << label;
}

void id_node::print_node (stringstream &out, vector<string> &initialized_labels, vector<string> &iters, bool perform_load, bool is_lhs) {
	if (PRINT_INTRINSICS && find (initialized_labels.begin(), initialized_labels.end(), label) == initialized_labels.end ()) {
		if (type == DOUBLE) out << "_mm256_set1_pd (" << label << ")";
		else if (type == FLOAT) out << "_mm256_set1_ps (" << label << ")";
		else out << "_mm256_set1_epi64 (" << label << ")";
	}
	//else out << label;
	else out << name;
}

void id_node::print_initializations (stringstream &header_output, vector<string> &initialized_labels, vector<string> iters, bool perform_load, bool is_lhs) {
	if (!PRINT_INTRINSICS) {
		if (is_lhs || perform_load) {	
			if (find (initialized_labels.begin(), initialized_labels.end(), label) == initialized_labels.end ()) {
				if (type == DOUBLE) header_output << "double " << label;
				else if (type == FLOAT) header_output << "float " << label;
				else if (type == INT) header_output << "int " << label; 
				else if (type == BOOL) header_output << "bool " << label;
				header_output << ";\n";
				initialized_labels.push_back (label);
			}
		}
	}
	else if (is_lhs || perform_load) {	
		if (find (initialized_labels.begin(), initialized_labels.end(), label) == initialized_labels.end ()) {
			if (type == DOUBLE) header_output << "__m256d " << label;
			else if (type == FLOAT) header_output << "__m256 " << label;
			else  header_output << "__m256i " << label;
			header_output << ";\n";
			initialized_labels.push_back (label);
		}
	}
}

void id_node::print_node (map<string, string> &reg_map, stringstream &out) {
	if (DEBUG) assert (reg_map.find (label) != reg_map.end ());
	out << reg_map[label];
}

void id_node::create_labels (map<string, expr_node*> &label_map) {
	label = name;
	if (label_map.find (label) == label_map.end ())
		label_map[label] = this;
}

void id_node::create_labels (map<string, int> &lassign_map, map<string, expr_node*> &label_map, bool is_asgn) {
	if (is_asgn) {
		if (lassign_map.find (name) == lassign_map.end ()) 
			lassign_map[name] = 0;
		else 
			lassign_map[name] = lassign_map[name] + 1;
	}
	if (lassign_map[name] != 0) 
		label = name + "_" + to_string (lassign_map[name]) + "_";	
	else label = name;
	if (label_map.find (label) == label_map.end ())
		label_map[label] = this;
}

//void id_node::stringify_accesses (vector<string> &labels, map<string, int> &lassign_map, bool is_asgn) {
//	string lbl;
//	if (is_asgn) {
//		if (lassign_map.find (name) == lassign_map.end ()) 
//			lassign_map[name] = 0;
//		else 
//			lassign_map[name] = lassign_map[name] + 1;
//	}
//	if (lassign_map[name] != 0) 
//		lbl = name + "_" + to_string (lassign_map[name]) + "_";	
//	else lbl = name;
//	labels.push_back (lbl);
//}
//
//void id_node::stringify_accesses (vector<string> &labels, string &expr_label, map<string, int> &lassign_map, bool is_asgn) {
//	string lbl;
//	if (is_asgn) {
//		if (lassign_map.find (name) == lassign_map.end ()) 
//			lassign_map[name] = 0;
//		else 
//			lassign_map[name] = lassign_map[name] + 1;
//	}
//	if (lassign_map[name] != 0) 
//		lbl = name + "_" + to_string (lassign_map[name]) + "_";	
//	else lbl = name;
//	labels.push_back (lbl);
//	expr_label = expr_label + lbl;
//}

void id_node::stringify_accesses (vector<string> &labels) {
	string lbl = name;
	labels.push_back (lbl);
}

void id_node::stringify_accesses (vector<string> &labels, string &expr_label) {
	string lbl = name;
	labels.push_back (lbl);
	expr_label = expr_label + lbl;
}

void id_node::array_access_info (string &id, int &offset) {
	id = id + name;
}

void id_node::gather_participating_labels (vector<string> &labels, vector<string> &names, vector<string> coefficients) {
	if (find (labels.begin(), labels.end(), label) == labels.end()) {
		if (!DROP_COEFS || (find (coefficients.begin(), coefficients.end(), label) == coefficients.end())) 
			labels.push_back (label);
	}
	if (find (names.begin(), names.end(), name) == names.end()) {
		if (!DROP_COEFS || (find (coefficients.begin(), coefficients.end(), label) == coefficients.end())) 
			names.push_back (name);
	}
}

expr_node *id_node::unroll_expr (string s, int val, vector<string> coefficients, map<string,int> &scalar_count, bool is_lhs) {
	expr_node *temp1;
	if (is_lhs) {
		if (scalar_count.find (name) == scalar_count.end()) 
			scalar_count[name] = 1;
		else
			scalar_count[name] = scalar_count[name] + 1;
	}
	if (scalar_count[name] == 0)
		temp1 = new id_node (name);
	else {
		string new_name = name + "_" + to_string (scalar_count[name]) + "_";
		temp1 = new id_node (new_name); 
	}
	if (name.compare (s) == 0 && val!=0) {
		expr_node *temp2 = new datatype_node<int> (val, INT);
		return new binary_node (T_PLUS, temp1, temp2);	
	}
	else return temp1;
}

expr_node *id_node::deep_copy (void) {
	return new id_node (name, type, nested, label);
}

void id_node::populate_tree (accnode *acc_node, map <string, treenode*> &tree_map, map<string, int> label_bitset_index, int label_count, bool is_uminus) {
	// Check if the node is already in tree_map
	if (tree_map.find (label) != tree_map.end ()) {
		treenode *tree_node = tree_map[label];
		tree_node->set_uminus_val (is_uminus);
		acc_node->add_operand (tree_node, label_bitset_index, label_count);
		tree_map.erase (label);
	}
	// Otherwise create a new leaf node
	else {
        stringstream lhs_print;
        this->print_node (lhs_print);
		treenode *node = new treenode (this, lhs_print.str(), label, label_bitset_index, label_count, is_uminus);
		acc_node->add_operand (node, label_bitset_index, label_count);
	}
	// Now modify the expr string
	acc_node->append_to_expr_string (label);	
	acc_node->append_to_rhs_printing (label);
}

void id_node::decompose_node (vector<tuple<expr_node*, expr_node*, STMT_OP>> &tstmt, vector<tuple<expr_node*, expr_node*, STMT_OP>> &init, vector<expr_node*> &temp_vars, expr_node *alhs, STMT_OP cur_op, int &id, DATA_TYPE gdata_type, bool &local_assigned, bool &global_assigned, bool flip) {
	if (DEBUG) {
		printf ("For me (%s), cur_op is %s", name.c_str(), (print_stmt_op (cur_op)).c_str());
		printf (", and flip is %d\n", flip);
	}
	if (!local_assigned) cur_op = acc_start_op (cur_op); 
	cur_op = get_cur_op (cur_op, flip);
	// Create a node of the form a += b;
	if (!global_assigned && cur_op != ST_EQ)
		init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
	tstmt.push_back (make_tuple (alhs, this, cur_op));
	local_assigned = true;
	global_assigned = true;
	alhs->set_type (gdata_type);
}

expr_node *uminus_node::get_base_expr (void) {
	return base_expr;
}	

bool uminus_node::simple_nondecomposable_expr (void) {
	type = base_expr->get_type ();
	bool ret = base_expr->get_expr_type () != T_BINARY && base_expr->get_expr_type () != T_FUNCTION;
	return (ret && base_expr->simple_nondecomposable_expr ());
}

bool uminus_node::is_data_type (DATA_TYPE gdata_type) {
	if (base_expr->is_data_type (gdata_type)) {
		type = gdata_type;
		return true;
	}
	return false;
}

bool uminus_node::is_data_type (void) {
	type = base_expr->get_type ();
	return (base_expr->is_data_type ()); 
}

bool uminus_node::is_id_type (DATA_TYPE gdata_type) {
	if (base_expr->is_id_type (gdata_type)) {
		type = gdata_type;
		return true;
	}
	return false;
}

bool uminus_node::is_id_type (void) {
	type = base_expr->get_type ();
	return (base_expr->is_id_type ());
}

bool uminus_node::is_shiftvec_type (DATA_TYPE gdata_type) {
	if (base_expr->is_shiftvec_type (gdata_type)) {
		type = base_expr->get_type ();
		return true;
	}
	return false;
}

bool uminus_node::is_shiftvec_type (void) {
	type = base_expr->get_type ();
	return (base_expr->is_shiftvec_type ()); 
}

void uminus_node::print_node (stringstream &out) {
	out << "-";
	base_expr->print_node (out); 
}

void uminus_node::print_node (stringstream &out, vector<string> &initialized_labels, vector<string> &iters, bool perform_load, bool is_lhs) {
	out << "-";
	base_expr->print_node (out, initialized_labels, iters, perform_load, is_lhs);
}

void uminus_node::print_initializations (stringstream &header_output, vector<string> &initialized_labels, vector<string> iters, bool perform_load, bool is_lhs) {
	base_expr->print_initializations (header_output, initialized_labels, iters, perform_load, is_lhs);
}

void uminus_node::print_node (map<string, string> &reg_map, stringstream &out) {
	out << "-";
	base_expr->print_node (reg_map, out); 
}

void uminus_node::create_labels (map<string, expr_node*> &label_map) {
	base_expr->create_labels (label_map);
}

void uminus_node::create_labels (map<string, int> &lassign_map, map<string, expr_node*> &label_map, bool is_asgn) {
	base_expr->create_labels (lassign_map, label_map, is_asgn);
}

//void uminus_node::stringify_accesses (vector<string> &labels, string &expr_label, map<string, int> &lassign_map, bool is_asgn) {
//	expr_label = expr_label + "-";
//	base_expr->stringify_accesses (labels, expr_label, lassign_map, is_asgn);
//}

void uminus_node::stringify_accesses (vector<string> &labels, string &expr_label) {
	expr_label = expr_label + "-";
	base_expr->stringify_accesses (labels, expr_label);
}

void uminus_node::gather_participating_labels (vector<string> &labels, vector<string> &names, vector<string> coefficients) {
	base_expr->gather_participating_labels (labels, names, coefficients);
}

expr_node *uminus_node::unroll_expr (string s, int val, vector<string> coefficients, map<string,int> &scalar_count, bool is_lhs) {
	expr_node *temp = base_expr->unroll_expr (s, val, coefficients, scalar_count, is_lhs);
	return new uminus_node (temp);	
}

expr_node *uminus_node::deep_copy (void) {
	expr_node *temp = base_expr->deep_copy ();
	return new uminus_node (temp, name, type, nested);
}

void uminus_node::populate_tree (accnode *acc_node, map <string, treenode*> &tree_map, map<string, int> label_bitset_index, int label_count, bool is_uminus) {
	acc_node->append_to_expr_string ("-");
	acc_node->append_to_rhs_printing ("-");
	base_expr->populate_tree (acc_node, tree_map, label_bitset_index, label_count, true);
}

void uminus_node::decompose_node (vector<tuple<expr_node*, expr_node*, STMT_OP>> &tstmt, vector<tuple<expr_node*, expr_node*, STMT_OP>> &init, vector<expr_node*> &temp_vars, expr_node *alhs, STMT_OP cur_op, int &id, DATA_TYPE gdata_type, bool &local_assigned, bool &global_assigned, bool flip) {
	cur_op = get_cur_op (cur_op, flip);
	if (base_expr->is_id_type (gdata_type) || base_expr->is_data_type (gdata_type) || base_expr->is_shiftvec_type (gdata_type)) {
		// Simply create a node like a += -b or a += -b[];
		if (!local_assigned) cur_op = acc_start_op (cur_op);
		if (!global_assigned && cur_op != ST_EQ) 
			init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
		tstmt.push_back (make_tuple (alhs, this, cur_op));
		local_assigned = true;
		global_assigned = true;
		// Determine the type of base expr and this node
		type = base_expr->get_type ();
		alhs->set_type (gdata_type);
	}
	else {
		// Otherwise explore base_expr with a different tlhs 
		string name_t = "_t_" + to_string (id++) + "_";
		expr_node *temp1 = new id_node (name_t);
		bool nested_local_assigned = false;
		bool nested_global_assigned = false;
		base_expr->decompose_node (tstmt, init, temp_vars, temp1, ST_EQ, id, gdata_type, nested_local_assigned, nested_global_assigned, false);
		expr_node *new_rhs = new uminus_node (temp1);
		if (!local_assigned) cur_op = acc_start_op (cur_op);
		if (!global_assigned && cur_op != ST_EQ) 
			init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
		tstmt.push_back (make_tuple (alhs, new_rhs, cur_op));
		local_assigned = true;
		global_assigned = true;
		// Now determine the type of node; this will be the type of rhs decomposition
		type = temp1->get_type ();
		new_rhs->set_type (type);
		alhs->set_type (gdata_type);
		temp_vars.push_back (temp1);
		temp_vars.push_back (new_rhs);
	}
}

void uminus_node::array_access_info (string &id, int &offset) {
	string temp_id = "";
	int temp_offset = 0;
	base_expr->array_access_info (temp_id, temp_offset);
	if (temp_id.length() > 0) 
		id = id + "-" + temp_id;
	offset += -1*temp_offset; 
}

bool binary_node::is_data_type (DATA_TYPE gdata_type) {
	if (lhs->is_data_type (gdata_type) && rhs->is_data_type (gdata_type)) {
		type = gdata_type;
		return true;
	}
	return false;
}

bool binary_node::is_data_type (void) {
	type = infer_data_type (lhs->get_type (), rhs->get_type ());
	return (lhs->is_data_type () || rhs->is_data_type ());
}

bool binary_node::is_id_type (DATA_TYPE gdata_type) {
	if ((lhs->is_data_type (gdata_type) && rhs->is_id_type (gdata_type)) || (rhs->is_data_type (gdata_type) && lhs->is_id_type (gdata_type))) {
		type = gdata_type;
		return true;
	}
	return false;
}

bool binary_node::is_id_type (void) {
	type = infer_data_type (lhs->get_type (), rhs->get_type ());
	return ((lhs->is_data_type () && rhs->is_id_type ()) || (lhs->is_id_type () && rhs->is_data_type ()));
}

bool binary_node::is_shiftvec_type (DATA_TYPE gdata_type) {
	if ((lhs->is_data_type (gdata_type) && rhs->is_shiftvec_type (gdata_type)) || (rhs->is_data_type (gdata_type) && lhs->is_shiftvec_type (gdata_type))) {
		type = gdata_type;
		return true;
	}
	return false;
}

bool binary_node::is_shiftvec_type (void) {
	type = infer_data_type (lhs->get_type (), rhs->get_type ());
	return ((lhs->is_data_type () && rhs->is_shiftvec_type ()) || (rhs->is_data_type () && lhs->is_shiftvec_type ()));
}

// Returns true of the expression has just two nodes, and no nestings
bool binary_node::simple_nondecomposable_expr (void) {
	type = infer_data_type (lhs->get_type (), rhs->get_type ());
	// Check that lhs and rhs are simple types, not binary or function
	bool ret = op != T_PLUS && op != T_MINUS;
	ret &= lhs->get_expr_type () != T_BINARY && rhs->get_expr_type () != T_BINARY;
	ret &= lhs->get_expr_type () != T_FUNCTION && lhs->get_expr_type () != T_FUNCTION;
	return (ret && lhs->simple_nondecomposable_expr () && rhs->simple_nondecomposable_expr ());
}

void binary_node::print_node (stringstream &out) {
	if (!PRINT_INTRINSICS) {
		lhs->print_node (out);
		out << print_operator (op);
		rhs->print_node (out);
	}
	else {
		type = infer_data_type (lhs->get_type (), rhs->get_type ());
		out << print_operator (op, type);
		out << " (";
		lhs->print_node (out);
		out << ", ";
		rhs->print_node (out);
		out << ")";
	}
}

void binary_node::print_node (stringstream &out, vector<string> &initialized_labels, vector<string> &iters, bool perform_load, bool is_lhs) {
	if (!PRINT_INTRINSICS) {
		lhs->print_node (out, initialized_labels, iters, perform_load, is_lhs);
		out << print_operator (op);
		rhs->print_node (out, initialized_labels, iters, perform_load, is_lhs);
	}
	else {
		type = infer_data_type (lhs->get_type (), rhs->get_type ());
		out << print_operator (op, type);
		out << " (";
		lhs->print_node (out, initialized_labels, iters, perform_load, is_lhs);
		out << ", ";
		rhs->print_node (out, initialized_labels, iters, perform_load, is_lhs);
		out << ")";
	}
}

void binary_node::print_initializations (stringstream &header_output, vector<string> &initialized_labels, vector<string> iters, bool perform_load, bool is_lhs) {
	lhs->print_initializations (header_output, initialized_labels, iters, perform_load, is_lhs);
	rhs->print_initializations (header_output, initialized_labels, iters, perform_load, is_lhs);
}      

void binary_node::print_node (map<string, string> &reg_map, stringstream &out) {
	lhs->print_node (reg_map, out);
	out << print_operator (op);
	rhs->print_node (reg_map, out);
}

expr_node *binary_node::unroll_expr (string s, int val, vector<string> coefficients, map<string,int> &scalar_count, bool is_lhs) {
	expr_node *temp1 = lhs->unroll_expr (s, val, coefficients, scalar_count, is_lhs);
	expr_node *temp2 = rhs->unroll_expr (s, val, coefficients, scalar_count, is_lhs);
	return new binary_node (op, temp1, temp2);
}

expr_node *binary_node::deep_copy (void) {
	expr_node *temp1 = lhs->deep_copy ();
	expr_node *temp2 = rhs->deep_copy ();
	return new binary_node (op, temp1, temp2, name, type, nested);
}

void binary_node::populate_tree (accnode *acc_node, map <string, treenode*> &tree_map, map<string, int> label_bitset_index, int label_count, bool is_uminus) {
	acc_node->set_rhs_operator (op);
	lhs->populate_tree (acc_node, tree_map, label_bitset_index, label_count, false);
	acc_node->append_to_expr_string (print_operator (op));
	acc_node->append_to_rhs_printing (print_operator (op));
	rhs->populate_tree (acc_node, tree_map, label_bitset_index, label_count, false);
}

void binary_node::decompose_node (vector<tuple<expr_node*, expr_node*, STMT_OP>> &tstmt, vector<tuple<expr_node*, expr_node*, STMT_OP>> &init, vector<expr_node*> &temp_vars, expr_node *alhs, STMT_OP cur_op, int &id, DATA_TYPE gdata_type, bool &local_assigned, bool &global_assigned, bool flip) {
	if (is_id_type (gdata_type) || is_shiftvec_type (gdata_type)) {
		if (!local_assigned) cur_op = acc_start_op (cur_op);
		cur_op = get_cur_op (cur_op, flip);
		if (!global_assigned && cur_op != ST_EQ)
			init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
		tstmt.push_back (make_tuple (alhs, this, cur_op));
		local_assigned = true;
		global_assigned = true;
		alhs->set_type (gdata_type);
		return;
	}
	if (op == T_PLUS || op == T_MINUS) {
		if (RETAIN_SIMPLE_OPS) {
			// Use this only for tree register allocation 
			if ((lhs->is_id_type (gdata_type) || lhs->is_data_type (gdata_type) || lhs->is_shiftvec_type (gdata_type)) &&
					(rhs->is_id_type (gdata_type) || rhs->is_data_type (gdata_type) || rhs->is_shiftvec_type (gdata_type))) {
				if (!local_assigned) cur_op = acc_start_op (cur_op);
				cur_op = get_cur_op (cur_op, flip);
				// If both lhs and rhs are simple (no further investiagion required), use the same node
				if (!global_assigned && cur_op != ST_EQ)
					init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
				tstmt.push_back (make_tuple (alhs, this, cur_op));
				local_assigned = true;
				global_assigned = true;
				// Now determine the type. First set the types of children, and then the parent
				type = infer_data_type (lhs->get_type (), rhs->get_type());
				alhs->set_type (gdata_type);
			}
			else {
				// If the node is a = b + c, then modify it simply as a = init_val; a += b; a += c;
				if (lhs->is_nested ()) {
					bool nested_local_assigned = false;
					lhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, nested_local_assigned, global_assigned, flip);
					local_assigned |= nested_local_assigned;
				}
				else lhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip);
				if (rhs->is_nested ()) {
					// Flip RHS signs only if the op is "-" or flip is not the same as nested_flip 
					bool nested_flip = (op == T_MINUS);
					if (nested_flip == flip) nested_flip = false;
					bool nested_local_assigned = false;
					rhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, nested_local_assigned, global_assigned, nested_flip);
					local_assigned |= nested_local_assigned;
				}
				else rhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip); 
			}
		}
		else {
			// If the node is a = b + c, then modify it simply as a = init_val; a += b; a += c;
			if (lhs->is_nested ()) { 
				bool nested_local_assigned = false;
				lhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, nested_local_assigned, global_assigned, flip);
				local_assigned |= nested_local_assigned;
			}
			else lhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip);
			if (rhs->is_nested ()) {
				// Flip RHS signs only if the op is "-" or flip is not the same as nested_flip
				bool nested_flip = (op == T_MINUS);
				if (nested_flip == flip) nested_flip = false;
				bool nested_local_assigned = false;
				rhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, nested_local_assigned, global_assigned, nested_flip);
				local_assigned |= nested_local_assigned;
			}
			else rhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip); 
		}
	}
	else if (ASSOC_MULT && (op == T_MULT || op == T_AND || op == T_OR)) {
		if (lhs->is_data_type (gdata_type)) {
			// rhs is complicated, so visit it
			string name_t2 = "_t_" + to_string (id++) + "_";
			expr_node *temp2 = new id_node (name_t2);
			bool nested_local_assigned2 = false;
			bool nested_global_assigned2 = false;
			bool nested_flip2 = false;
			rhs->decompose_node (tstmt, init, temp_vars, temp2, ST_EQ, id, gdata_type, nested_local_assigned2, nested_global_assigned2, nested_flip2);
			expr_node *new_rhs = new binary_node (op, lhs, temp2);
			if (!local_assigned) cur_op = acc_start_op (cur_op);
			cur_op = get_cur_op (cur_op, flip);
			if (!global_assigned && cur_op != ST_EQ)
				init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
			tstmt.push_back (make_tuple (alhs, new_rhs, cur_op));
			local_assigned = true;
			global_assigned = true;
			// Now determine the type. First set the types of children, and then the parent
			type = infer_data_type (lhs->get_type (), temp2->get_type());
			new_rhs->set_type (type);
			alhs->set_type (gdata_type);
			temp_vars.push_back (temp2);
			temp_vars.push_back (new_rhs);
		}
		else if (rhs->is_data_type (gdata_type)) {
			// lhs is not simple, visit it
			string name_t1 = "_t_" + to_string (id++) + "_";
			expr_node *temp1 = new id_node (name_t1);
			bool nested_local_assigned1 = false;
			bool nested_global_assigned1 = false;
			bool nested_flip1 = false;
			lhs->decompose_node (tstmt, init, temp_vars, temp1, ST_EQ, id, gdata_type, nested_local_assigned1, nested_global_assigned1, nested_flip1);
			expr_node *new_rhs = new binary_node (op, temp1, rhs);
			if (!local_assigned) cur_op = acc_start_op (cur_op);
			cur_op = get_cur_op (cur_op, flip);
			if (!global_assigned && cur_op != ST_EQ) 
				init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
			tstmt.push_back (make_tuple (alhs, new_rhs, cur_op));
			local_assigned = true;
			global_assigned = true;
			// Now determine the type. First set the types of children, and then the parent
			type = infer_data_type (temp1->get_type (), rhs->get_type());
			new_rhs->set_type (type);
			alhs->set_type (gdata_type);
			temp_vars.push_back (temp1);
			temp_vars.push_back (new_rhs);
		}
		else if ((lhs->is_id_type (gdata_type) || lhs->is_shiftvec_type (gdata_type)) && (rhs->is_id_type (gdata_type) || rhs->is_shiftvec_type (gdata_type))) {
			// If the cur_op is the same as my op, it simplifies a lot of things
			if (cur_op == convert_op_to_stmt_op (op)) {
				lhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip);
				rhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip);
			}
			else {
				// If both lhs and rhs are simple (no further investigation required), use the same node
				string name_t1 = "_t_" + to_string (id++) + "_";
				expr_node *temp1 = new id_node (name_t1);
				if (!local_assigned) cur_op = acc_start_op (cur_op);
				cur_op = get_cur_op (cur_op, flip);
				if (cur_op != ST_EQ) 
					init.push_back (make_tuple (temp1, new datatype_node<int> (get_init_val (convert_op_to_stmt_op (op)), INT), ST_EQ));
				tstmt.push_back (make_tuple (temp1, lhs, convert_op_to_stmt_op (op)));
				tstmt.push_back (make_tuple (temp1, rhs, convert_op_to_stmt_op (op)));
				if (!global_assigned && cur_op != ST_EQ) 
					init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
				tstmt.push_back (make_tuple (alhs, temp1, cur_op));
				local_assigned = true;
				global_assigned = true;
				// Now determine the type. First set the types of children, and then the parent
				type = infer_data_type (lhs->get_type (), rhs->get_type());
				temp1->set_type (type);
				alhs->set_type (gdata_type);
				temp_vars.push_back (temp1);
			}
		}
		else if (lhs->is_id_type (gdata_type) || lhs->is_shiftvec_type (gdata_type)) {
			// If the cur_op is the same as my op, it simplifies a lot of things
			if (cur_op == convert_op_to_stmt_op (op)) {
				lhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip);
				rhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip);
			}
			else {
				// lhs is simple, so only explore rhs
				string name_t1 = "_t_" + to_string (id++) + "_";
				expr_node *new_rhs = new id_node (name_t1);
				string name_t2 = "_t_" + to_string (id++) + "_";
				expr_node *temp2 = new id_node (name_t2);
				bool nested_local_assigned2 = false;
				bool nested_global_assigned2 = false;
				bool nested_flip2 = false;
				rhs->decompose_node (tstmt, init, temp_vars, temp2, ST_EQ, id, gdata_type, nested_local_assigned2, nested_global_assigned2, nested_flip2);
				if (!local_assigned) cur_op = acc_start_op (cur_op);
				cur_op = get_cur_op (cur_op, flip);
				if (cur_op != ST_EQ) 
					init.push_back (make_tuple (new_rhs, new datatype_node<int> (get_init_val (convert_op_to_stmt_op (op)), INT), ST_EQ));
				tstmt.push_back (make_tuple (new_rhs, lhs, convert_op_to_stmt_op (op)));
				tstmt.push_back (make_tuple (new_rhs, temp2, convert_op_to_stmt_op (op)));
				if (!global_assigned && cur_op != ST_EQ) 
					init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
				tstmt.push_back (make_tuple (alhs, new_rhs, cur_op));
				local_assigned = true;
				global_assigned = true;
				// Now infer the type of lhs and the new rhs
				type = infer_data_type (lhs->get_type(), temp2->get_type());
				new_rhs->set_type (type);
				alhs->set_type (gdata_type);
				temp_vars.push_back (new_rhs);
				temp_vars.push_back (temp2);
			}
		}
		else if (rhs->is_id_type (gdata_type) || rhs->is_shiftvec_type (gdata_type)) {
			// If the cur_op is the same as my op, it simplifies a lot of things
			if (cur_op == convert_op_to_stmt_op (op)) {
				lhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip);
				rhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip);
			}
			else {
				// Explore each child with a separate lhs
				string name_t2 = "_t_" + to_string (id++) + "_";
				expr_node *new_rhs = new id_node (name_t2);
				string name_t1 = "_t_" + to_string (id++) + "_";
				expr_node *temp1 = new id_node (name_t1);
				bool nested_local_assigned1 = false;
				bool nested_global_assigned1 = false;
				bool nested_flip1 = false;
				lhs->decompose_node (tstmt, init, temp_vars, temp1, ST_EQ, id, gdata_type, nested_local_assigned1, nested_global_assigned1, nested_flip1);
				if (!local_assigned) cur_op = acc_start_op (cur_op);
				cur_op = get_cur_op (cur_op, flip);
				if (cur_op != ST_EQ) 
					init.push_back (make_tuple (new_rhs, new datatype_node<int> (get_init_val (convert_op_to_stmt_op (op)), INT), ST_EQ));
				tstmt.push_back (make_tuple (new_rhs, temp1, convert_op_to_stmt_op (op)));
				tstmt.push_back (make_tuple (new_rhs, rhs, convert_op_to_stmt_op (op)));
				if (!global_assigned && cur_op != ST_EQ) 
					init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
				tstmt.push_back (make_tuple (alhs, new_rhs, cur_op));
				local_assigned = true;
				global_assigned = true;
				// Now infer the type of lhs and the new rhs
				type = infer_data_type (temp1->get_type(), rhs->get_type());
				new_rhs->set_type (type);
				alhs->set_type (gdata_type);
				temp_vars.push_back (temp1);
				temp_vars.push_back (new_rhs);
			}
		}
		else {
			// If the cur_op is the same as my op, it simplifies a lot of things
			if (cur_op == convert_op_to_stmt_op (op)) {
				lhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip);
				rhs->decompose_node (tstmt, init, temp_vars, alhs, convert_op_to_stmt_op (op), id, gdata_type, local_assigned, global_assigned, flip);
			}
			else {
				// Explore each child with a separate lhs
				string name_t0 = "_t_" + to_string (id++) + "_";
				string name_t1 = "_t_" + to_string (id++) + "_";
				string name_t2 = "_t_" + to_string (id++) + "_";
				expr_node *new_rhs = new id_node (name_t0);
				expr_node *temp1 = new id_node (name_t1);
				expr_node *temp2 = new id_node (name_t2);
				bool nested_local_assigned1 = false, nested_local_assigned2 = false;
				bool nested_global_assigned1 = false, nested_global_assigned2 = false;
				bool nested_flip1 = false, nested_flip2 = false;
				lhs->decompose_node (tstmt, init, temp_vars, temp1, ST_EQ, id, gdata_type, nested_local_assigned1, nested_global_assigned1, nested_flip1);
				rhs->decompose_node (tstmt, init, temp_vars, temp2, ST_EQ, id, gdata_type, nested_local_assigned2, nested_global_assigned2, nested_flip2);
				if (!local_assigned) cur_op = acc_start_op (cur_op);
				cur_op = get_cur_op (cur_op, flip);
				if (cur_op != ST_EQ) 
					init.push_back (make_tuple (new_rhs, new datatype_node<int> (get_init_val (convert_op_to_stmt_op (op)), INT), ST_EQ));
				tstmt.push_back (make_tuple (new_rhs, temp1, convert_op_to_stmt_op (op)));
				tstmt.push_back (make_tuple (new_rhs, temp2, convert_op_to_stmt_op (op)));
				if (!global_assigned && cur_op != ST_EQ) 
					init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
				tstmt.push_back (make_tuple (alhs, new_rhs, cur_op));
				local_assigned = true;
				global_assigned = true;
				// Now infer the type of lhs and the new rhs
				type = infer_data_type (temp1->get_type(), temp2->get_type());
				new_rhs->set_type (type);
				alhs->set_type (gdata_type);
				temp_vars.push_back (temp1);
				temp_vars.push_back (temp2);
				temp_vars.push_back (new_rhs);
			}
		}
	}
	else {
		if ((lhs->is_id_type (gdata_type) || lhs->is_data_type (gdata_type) || lhs->is_shiftvec_type (gdata_type)) &&
				(rhs->is_id_type (gdata_type) || rhs->is_data_type (gdata_type) || rhs->is_shiftvec_type (gdata_type))) {
			if (!local_assigned) cur_op = acc_start_op (cur_op);
			cur_op = get_cur_op (cur_op, flip);
			// If both lhs and rhs are simple (no further investiagion required), use the same node
			if (!global_assigned && cur_op != ST_EQ) 
				init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
			tstmt.push_back (make_tuple (alhs, this, cur_op));
			local_assigned = true;
			global_assigned = true;
			// Now determine the type. First set the types of children, and then the parent
			type = infer_data_type (lhs->get_type (), rhs->get_type());
			alhs->set_type (gdata_type);
		}
		else if (lhs->is_id_type (gdata_type) || lhs->is_data_type (gdata_type) || lhs->is_shiftvec_type (gdata_type)) {
			// lhs is simple, so only explore rhs
			string name_t2 = "_t_" + to_string (id++) + "_";
			expr_node *temp2 = new id_node (name_t2);
			bool nested_local_assigned2 = false;
			bool nested_global_assigned2 = false;
			bool nested_flip2 = false;
			rhs->decompose_node (tstmt, init, temp_vars, temp2, ST_EQ, id, gdata_type, nested_local_assigned2, nested_global_assigned2, nested_flip2);
			expr_node *new_rhs = new binary_node (op, lhs, temp2);
			if (!local_assigned) cur_op = acc_start_op (cur_op);
			cur_op = get_cur_op (cur_op, flip);
			if (!global_assigned && cur_op != ST_EQ) 
				init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
			tstmt.push_back (make_tuple (alhs, new_rhs, cur_op));
			local_assigned = true;
			global_assigned = true;
			// Now infer the type of lhs and the new rhs
			type = infer_data_type (lhs->get_type(), temp2->get_type());
			new_rhs->set_type (type);
			alhs->set_type (gdata_type);
			temp_vars.push_back (temp2);
			temp_vars.push_back (new_rhs);
		}
		else if (rhs->is_id_type (gdata_type) || rhs->is_data_type (gdata_type) || rhs->is_shiftvec_type (gdata_type)) {
			// Explore each child with a separate lhs
			string name_t1 = "_t_" + to_string (id++) + "_";
			expr_node *temp1 = new id_node (name_t1);
			bool nested_local_assigned1 = false;
			bool nested_global_assigned1 = false;
			bool nested_flip1 = false;
			lhs->decompose_node (tstmt, init, temp_vars, temp1, ST_EQ, id, gdata_type, nested_local_assigned1, nested_global_assigned1, nested_flip1);
			expr_node *new_rhs = new binary_node (op, temp1, rhs);
			if (!local_assigned) cur_op = acc_start_op (cur_op);
			cur_op = get_cur_op (cur_op, flip);
			if (!global_assigned && cur_op != ST_EQ) 
				init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
			tstmt.push_back (make_tuple (alhs, new_rhs, cur_op));
			local_assigned = true;
			global_assigned = true;
			// Now infer the type of lhs and the new rhs
			type = infer_data_type (temp1->get_type(), rhs->get_type());
			new_rhs->set_type (type);
			alhs->set_type (gdata_type);
			temp_vars.push_back (temp1);
			temp_vars.push_back (new_rhs);
		}
		else {
			// Explore each child with a separate lhs
			string name_t1 = "_t_" + to_string (id++) + "_";
			string name_t2 = "_t_" + to_string (id++) + "_";
			expr_node *temp1 = new id_node (name_t1);
			expr_node *temp2 = new id_node (name_t2);
			bool nested_local_assigned1 = false, nested_local_assigned2 = false;
			bool nested_global_assigned1 = false, nested_global_assigned2 = false;
			bool nested_flip1 = false, nested_flip2 = false;
			lhs->decompose_node (tstmt, init, temp_vars, temp1, ST_EQ, id, gdata_type, nested_local_assigned1, nested_global_assigned1, nested_flip1);
			rhs->decompose_node (tstmt, init, temp_vars, temp2, ST_EQ, id, gdata_type, nested_local_assigned2, nested_global_assigned2, nested_flip2);
			expr_node *new_rhs = new binary_node (op, temp1, temp2);
			if (!local_assigned) cur_op = acc_start_op (cur_op);
			cur_op = get_cur_op (cur_op, flip);
			if (!global_assigned && cur_op != ST_EQ) 
				init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
			tstmt.push_back (make_tuple (alhs, new_rhs, cur_op));
			local_assigned = true;
			global_assigned = true;
			// Now infer the type of lhs and the new rhs
			type = infer_data_type (temp1->get_type(), temp2->get_type());
			new_rhs->set_type (type);
			alhs->set_type (gdata_type);
			temp_vars.push_back (temp1);
			temp_vars.push_back (temp2);
			temp_vars.push_back (new_rhs);
		}
	}
}

void binary_node::create_labels (map<string, expr_node*> &label_map) {
	lhs->create_labels (label_map);
	rhs->create_labels (label_map);
}

void binary_node::create_labels (map<string, int> &lassign_map, map<string, expr_node*> &label_map, bool is_asgn) {
	lhs->create_labels (lassign_map, label_map, is_asgn);
	rhs->create_labels (lassign_map, label_map, is_asgn);
}

//void binary_node::stringify_accesses (vector<string> &labels, string &expr_label, map<string, int> &lassign_map, bool is_asgn) {
//	lhs->stringify_accesses (labels, expr_label, lassign_map, is_asgn);
//	expr_label = expr_label + print_operator (op);
//	rhs->stringify_accesses (labels, expr_label, lassign_map, is_asgn);
//}

void binary_node::stringify_accesses (vector<string> &labels, string &expr_label) {
	lhs->stringify_accesses (labels, expr_label);
	expr_label = expr_label + print_operator (op);
	rhs->stringify_accesses (labels, expr_label);
}

void binary_node::array_access_info (string &id, int &offset) {
	string l_id = "", r_id = "";
	int l_val = 0, r_val = 0;
	lhs->array_access_info (l_id, l_val);
	rhs->array_access_info (r_id, r_val);
	if (op == T_DIV) {
		if (DEBUG) assert ((l_id.length () == 0 || l_val == 0) && "Complicated division of array accesses unhandled (array_access_info)");
		if (DEBUG) assert ((r_id.length () == 0 || r_val == 0) && "Complicated division of array accesses unhandled (array_access_info)");
		if (l_id.length () == 0 && r_id.length () == 0)
			offset += floor (l_val / r_val);
		else if (l_id.length () != 0) {
			if (r_id.length () == 0) 
				id = id + l_id + "/" + to_string (r_val);
			else
				id = id + l_id + "/" + r_id; 
		}
		else if (r_id.length () != 0) 
			id = id + to_string (l_val) + "/" + r_id;
	}
	else if (op == T_MULT) {
		// Arrange IDs on left, and val on right.
		if (DEBUG) assert ((l_id.length () == 0 || l_val == 0) && "Complicated multiplication of array accesses unhandled (array_access_info)");
		if (DEBUG) assert ((r_id.length () == 0 || r_val == 0) && "Complicated multiplication of array accesses unhandled (array_access_info)");
		if (l_id.length () == 0 && r_id.length () == 0)
			offset += l_val * r_val;
		else if (l_id.length () != 0) {
			if (r_id.length () == 0) 
				id = id + l_id + "*" + to_string (r_val);
			else {
				string first = (l_id.compare (r_id) <= 0) ? l_id : r_id;
				string second = (l_id.compare (r_id) > 0) ? l_id : r_id;
				id = id + first + "*" + second;
			}
		}
		else if (r_id.length () != 0) 
			id = id + r_id + "*" + to_string (l_val);
	}
	else if (op == T_PLUS) {
		if (l_id.length () > 0 && r_id.length () > 0) { 
			string first = (l_id.compare (r_id) <= 0) ? l_id : r_id;
			string second = (l_id.compare (r_id) > 0) ? l_id : r_id;
			id = id + first + "+" + second;
		}
		else if (l_id.length () > 0)
			id = id + l_id;
		else if (r_id.length () > 0)
			id = id + r_id;
		offset += l_val + r_val;
	}
	else if (op == T_MINUS) {
		if (l_id.length () > 0 && r_id.length () > 0) { 
			string first = (l_id.compare (r_id) <= 0) ? l_id : r_id;
			string second = (l_id.compare (r_id) > 0) ? l_id : r_id;
			id = id + first + "-" + second;
		}
		else if (l_id.length () > 0)
			id = id + l_id;
		else if (r_id.length () > 0)
			id = id + "-" + r_id;
		offset += l_val - r_val;
	}
}

void binary_node::gather_participating_labels (vector<string> &labels, vector<string> &names, vector<string> coefficients) {
	lhs->gather_participating_labels (labels, names, coefficients);
	rhs->gather_participating_labels (labels, names, coefficients);
}

vector<expr_node *>& shiftvec_node::get_indices (void) {
	return indices;
}

int shiftvec_node::get_index_size (void) {
	return indices.size ();
}

void shiftvec_node::gather_participating_labels (vector<string> &labels, vector<string> &names, vector<string> coefficients) {
	if (find (labels.begin(), labels.end(), label) == labels.end()) {
		if (!DROP_COEFS || (find (coefficients.begin(), coefficients.end(), name) == coefficients.end())) 
			labels.push_back (label);
	}
	if (find (names.begin(), names.end(), name) == names.end()) {
		if (!DROP_COEFS || (find (coefficients.begin(), coefficients.end(), name) == coefficients.end())) 
			names.push_back (name);
	}
}

void shiftvec_node::print_node (stringstream &out) {
	out << print_array ();
}

void shiftvec_node::print_node (stringstream &out, vector<string> &initialized_labels, vector<string> &iters, bool perform_load, bool is_lhs) {
	if (perform_load || is_lhs || (find (initialized_labels.begin(), initialized_labels.end(), label) != initialized_labels.end())) 
		out << label;
	else {
		if (!PRINT_INTRINSICS) 
			out << print_array ();
		else {
			// Get the last dimension
			string arr_id = "";
			int offset = 0;
			vector<expr_node*>::reverse_iterator it = indices.rbegin();
			(*it)->array_access_info (arr_id, offset);
			if (DEBUG) assert (!iters.empty () && "Forgot to specify the iterators in the DSL? (print_node ())");
			bool splat = (arr_id.compare (iters.front()) != 0);
			if (splat) {
				if (type == DOUBLE) out << "_mm256_set1_pd (" << print_array () << ")";
				else if (type == FLOAT) out << "_mm256_set1_ps (" << print_array () << ")";
				else out << "_mm256_set1_epi64 (" << print_array () << ")";
			}
			else {
				if (type == DOUBLE) out << "_mm256_loadu_pd (&" << print_array () << ")";
				else if (type == FLOAT) out << "_mm256_loadu_ps (&" << print_array () << ")";
				else out << "_mm256_loadu_epi64 (&" << print_array () << ")";
			}
		}
	}
}

string shiftvec_node::print_array (void) {
	stringstream out;
	out << name;
	for (vector<expr_node*>::const_iterator i=indices.begin(); i!=indices.end(); i++) {
		string id = "";
		int offset = 0;
		(*i)->array_access_info (id, offset);
		out << "[";
		if (id.compare ("") == 0) 
			out << offset;		
		else {
			out << id;
			if (offset > 0) out << "+";
			else if (offset < 0) out << "-";
			if (offset != 0) out << abs (offset);
		}
		out << "]";
	}
	return out.str ();
}

void shiftvec_node::print_node (map<string, string> &reg_map, stringstream &out) {
	if (DEBUG) assert (reg_map.find (label) != reg_map.end ());
	out << reg_map[label];
}

void shiftvec_node::print_initializations (stringstream &header_output, vector<string> &initialized_labels, vector<string> iters, bool perform_load, bool is_lhs) {
	if (!PRINT_INTRINSICS) {
		if (is_lhs || perform_load) {
			if (find (initialized_labels.begin(), initialized_labels.end(), label) == initialized_labels.end ()) {
				if (type == DOUBLE) header_output << "double " << label;
				else if (type == FLOAT) header_output << "float " << label;
				else if (type == INT) header_output << "int " << label;
				else if (type == BOOL) header_output << "bool " << label;
				if (perform_load) header_output << " = " << print_array (); 
				header_output << ";\n";
				initialized_labels.push_back (label);
			}
		}
	}
	else if (is_lhs || perform_load) {
		if (DEBUG) assert (!iters.empty () && "Forgot to specify the iterators in the DSL? (print_initializations ())"); 
		// Get the last dimension
		string arr_id = "";
		int offset = 0;
		vector<expr_node*>::reverse_iterator it = indices.rbegin();
		(*it)->array_access_info (arr_id, offset);
		if (find (initialized_labels.begin(), initialized_labels.end(), label) == initialized_labels.end ()) {
			if (type == DOUBLE) header_output << "__m256d " << label;
			else if (type == FLOAT) header_output << "__m256 " << label;
			else  header_output << "__m256i " << label;
			// Print either load or splat
			if (perform_load) {
				bool splat = (arr_id.compare (iters.front()) != 0);
				header_output << " = ";
				if (splat) {
					if (type == DOUBLE) header_output << "_mm256_set1_pd (" << print_array () << ")";
					else if (type == FLOAT) header_output << "_mm256_set1_ps (" << print_array () << ")";
					else header_output << "_mm256_set_epi64 (" << print_array () << ")";
				}
				else {
					if (type == DOUBLE) header_output << "_mm256_loadu_pd (&" << print_array () << ")";
					else if (type == FLOAT) header_output << "_mm256_loadu_ps (&" << print_array () << ")";
					else header_output << "_mm256_loadu_epi64 (&" << print_array () << ")";
				}
			}
			header_output << ";\n";
			initialized_labels.push_back (label);
		}
	}
}

expr_node *shiftvec_node::unroll_expr (string s, int val, vector<string> coefficients, map<string,int> &scalar_count, bool is_lhs) {
	shiftvec_node *ret = new shiftvec_node (name);
	bool is_coef = find (coefficients.begin(), coefficients.end(), name) != coefficients.end ();
	for (vector<expr_node*>::const_iterator i=indices.begin(); i!=indices.end(); i++) {
		expr_node *temp = is_coef ? (*i)->unroll_expr (s, 0, coefficients, scalar_count, false) : (*i)->unroll_expr (s, val, coefficients, scalar_count, false);
		ret->push_index (temp);
	}
	return ret;
}

expr_node *shiftvec_node::deep_copy (void) {
	shiftvec_node *ret = new shiftvec_node (name, type, nested, label);
	for (vector<expr_node*>::const_iterator i=indices.begin(); i!=indices.end(); i++) {
		expr_node *temp = (*i)->deep_copy ();
		ret->push_index (temp);
	}
	return ret;
}

void shiftvec_node::populate_tree (accnode *acc_node, map <string, treenode*> &tree_map, map<string, int> label_bitset_index, int label_count, bool is_uminus) {
	// Check if the node is already in tree_map
	if (tree_map.find (label) != tree_map.end ()) {
		treenode *tree_node = tree_map[label];
		tree_node->set_uminus_val (is_uminus);
		acc_node->add_operand (tree_node, label_bitset_index, label_count);
		tree_map.erase (label);
	}
	// Otherwise create a new leaf node
	else {
        stringstream lhs_print;
        this->print_node (lhs_print);
		treenode *node = new treenode (this, lhs_print.str(), label, label_bitset_index, label_count, is_uminus);
		acc_node->add_operand (node, label_bitset_index, label_count);
	}
	// Now modify the expr string
	acc_node->append_to_expr_string (label); 
	acc_node->append_to_rhs_printing (print_array());
}

void shiftvec_node::decompose_node(vector<tuple<expr_node*, expr_node*, STMT_OP>> &tstmt, vector<tuple<expr_node*, expr_node*, STMT_OP>> &init, vector<expr_node*> &temp_vars, expr_node *alhs, STMT_OP cur_op, int &id, DATA_TYPE gdata_type, bool &local_assigned, bool &global_assigned, bool flip) {
	if (DEBUG) {
		printf ("For me (%s), cur_op is %s", name.c_str(), (print_stmt_op (cur_op)).c_str());
		printf (", and flip is %d\n", flip);
	}
	if (!local_assigned) cur_op = acc_start_op (cur_op);
	cur_op = get_cur_op (cur_op, flip);
	if (!global_assigned && cur_op != ST_EQ) 
		init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
	tstmt.push_back (make_tuple (alhs, this, cur_op));
	local_assigned = true;
	global_assigned = true;
	// Now infer the types
	alhs->set_type (gdata_type);
}

void shiftvec_node::create_labels (map<string, expr_node*> &label_map) {
	// A bit involved
	label = name;
	for (vector<expr_node*>::const_iterator i=indices.begin(); i!=indices.end(); i++) {
		string id = "";
		int offset = 0;
		(*i)->array_access_info (id, offset);
		if (id.length () > 1)
			label = label + "(" + id + ")";
		else label = label + id;
		if (offset == 0) label = label + "c";
		else if (offset < 0) label = label + "m";
		else label = label + "p";
		label = label + to_string (abs(offset));
	}
	if (label_map.find (label) == label_map.end ())
		label_map[label] = this;
}

void shiftvec_node::create_labels (map<string, int> &lassign_map, map<string, expr_node*> &label_map, bool is_asgn) {
	label = name;
	// A bit involved
	for (vector<expr_node*>::const_iterator i=indices.begin(); i!=indices.end(); i++) {
		string id = "";
		int offset = 0;
		(*i)->array_access_info (id, offset);
		if (id.length () > 1) 
			label = label + "(" + id + ")";
		else label = label + id;
		if (offset == 0) label = label + "c";
		else if (offset < 0) label = label + "m";
		else label = label + "p";
		label = label + to_string (abs(offset));
	}
	// Now modify if the label is lhs of an assignment
	if (is_asgn) {
		if (lassign_map.find (label) == lassign_map.end ()) 
			lassign_map[label] = 0;
		else 
			lassign_map[label] = lassign_map[label] + 1;
	}
	if (lassign_map[label] != 0) 
		label = label + "_" + to_string (lassign_map[label]) + "_";

	if (label_map.find (label) == label_map.end ())
		label_map[label] = this;
}

//void shiftvec_node::stringify_accesses (vector<string> &labels, map<string, int> &lassign_map, bool is_asgn) {
//	string lbl = name;
//	// A bit involved
//	for (vector<expr_node*>::const_iterator i=indices.begin(); i!=indices.end(); i++) {
//		string id = "";
//		int offset = 0;
//		(*i)->array_access_info (id, offset);
//		if (id.length () > 1) 
//			lbl = lbl + "(" + id + ")";
//		else lbl = lbl + id;
//		if (offset == 0) lbl = lbl + "c";
//		else if (offset < 0) lbl = lbl + "m";
//		else lbl = lbl + "p";
//		lbl = lbl + to_string (abs(offset));
//	}
//	// Now modify if the lbl is lhs of an assignment
//	if (is_asgn) {
//		if (lassign_map.find (lbl) == lassign_map.end ()) 
//			lassign_map[lbl] = 0;
//		else 
//			lassign_map[lbl] = lassign_map[lbl] + 1;
//	}
//	if (lassign_map[lbl] != 0) 
//		lbl = lbl + "_" + to_string (lassign_map[lbl]) + "_";
//	labels.push_back (lbl);
//}
//
//void shiftvec_node::stringify_accesses (vector<string> &labels, string &expr_label, map<string, int> &lassign_map, bool is_asgn) {
//	string lbl = name;
//	// A bit involved
//	for (vector<expr_node*>::const_iterator i=indices.begin(); i!=indices.end(); i++) {
//		string id = "";
//		int offset = 0;
//		(*i)->array_access_info (id, offset);
//		if (id.length () > 1) 
//			lbl = lbl + "(" + id + ")";
//		else lbl = lbl + id;
//		if (offset == 0) lbl = lbl + "c";
//		else if (offset < 0) lbl = lbl + "m";
//		else lbl = lbl + "p";
//		lbl = lbl + to_string (abs(offset));
//	}
//	// Now modify if the lbl is lhs of an assignment
//	if (is_asgn) {
//		if (lassign_map.find (lbl) == lassign_map.end ()) 
//			lassign_map[lbl] = 0;
//		else 
//			lassign_map[lbl] = lassign_map[lbl] + 1;
//	}
//	if (lassign_map[lbl] != 0) 
//		lbl = lbl + "_" + to_string (lassign_map[lbl]) + "_";
//	labels.push_back (lbl);
//	expr_label = expr_label + lbl;
//}

void shiftvec_node::stringify_accesses (vector<string> &labels) {
	string lbl = name;
	// A bit involved
	for (vector<expr_node*>::const_iterator i=indices.begin(); i!=indices.end(); i++) {
		string id = "";
		int offset = 0;
		(*i)->array_access_info (id, offset);
		if (id.length () > 1) 
			lbl = lbl + "(" + id + ")";
		else lbl = lbl + id;
		if (offset == 0) lbl = lbl + "c";
		else if (offset < 0) lbl = lbl + "m";
		else lbl = lbl + "p";
		lbl = lbl + to_string (abs(offset));
	}
	labels.push_back (lbl);
}

void shiftvec_node::stringify_accesses (vector<string> &labels, string &expr_label) {
	string lbl = name;
	// A bit involved
	for (vector<expr_node*>::const_iterator i=indices.begin(); i!=indices.end(); i++) {
		string id = "";
		int offset = 0;
		(*i)->array_access_info (id, offset);
		if (id.length () > 1) 
			lbl = lbl + "(" + id + ")";
		else lbl = lbl + id;
		if (offset == 0) lbl = lbl + "c";
		else if (offset < 0) lbl = lbl + "m";
		else lbl = lbl + "p";
		lbl = lbl + to_string (abs(offset));
	}
	labels.push_back (lbl);
	expr_label = expr_label + lbl;
}

void shiftvec_node::lexical_index_offsets (vector<int> &v) {
	for (vector<expr_node*>::const_iterator i=indices.begin(); i!=indices.end(); i++) {
		string id = "";
		int offset = 0;
		(*i)->array_access_info (id, offset);
		v.push_back (offset);
	}
}

void function_node::create_labels (map<string, expr_node*> &label_map) {
	arg->create_labels (label_map);
}

void function_node::create_labels (map<string, int> &lassign_map, map<string, expr_node*> &label_map, bool is_asgn) {
	arg->create_labels (lassign_map, label_map, is_asgn);
}

//void function_node::stringify_accesses (vector<string> &labels, map<string, int> &lassign_map, bool is_asgn) {
//	arg->stringify_accesses (labels, lassign_map, is_asgn);
//}
//
//void function_node::stringify_accesses (vector<string> &labels, string &expr_label, map<string, int> &lassign_map, bool is_asgn) {
//	expr_label = expr_label + name + "(";
//	arg->stringify_accesses (labels, expr_label, lassign_map, is_asgn);
//	expr_label = expr_label + ")";
//}

void function_node::stringify_accesses (vector<string> &labels) {
	arg->stringify_accesses (labels);
}

void function_node::stringify_accesses (vector<string> &labels, string &expr_label) {
	expr_label = expr_label + name + "(";
	arg->stringify_accesses (labels, expr_label);
	expr_label = expr_label + ")";
}

void function_node::gather_participating_labels (vector<string> &labels, vector<string> &names, vector<string> coefficients) {
	arg->gather_participating_labels (labels, names, coefficients);
}

bool function_node::is_data_type (DATA_TYPE gdata_type) {
	if (arg->is_data_type (gdata_type)) {
		type = gdata_type;
		return true;
	}
	return false;
}

bool function_node::is_data_type (void) {
	return arg->is_data_type ();
}

bool function_node::is_id_type (DATA_TYPE gdata_type) {
	if (arg->is_id_type (gdata_type)) {
		type = arg->get_type ();
		return true;
	}
	return false;
}

bool function_node::is_shiftvec_type (DATA_TYPE gdata_type) {
	if (arg->is_shiftvec_type (gdata_type)) {
		type = arg->get_type ();
		return true;
	}
	return false;
}

void function_node::print_node (stringstream &out) {
	out << name << "(";  
	arg->print_node (out);
	out << ")";
}

void function_node::print_node (stringstream &out, vector<string> &initialized_labels, vector<string> &iters, bool perform_load, bool is_lhs) {
	out << name << "(";
	arg->print_node (out, initialized_labels, iters, perform_load, is_lhs);
	out << ")";
}

void function_node::print_initializations (stringstream &header_output, vector<string> &initialized_labels, vector<string> iters, bool perform_load, bool is_lhs) {
	arg->print_initializations (header_output, initialized_labels, iters, perform_load, is_lhs);
}

void function_node::print_node (map<string, string> &reg_map, stringstream &out) {
	out << name << "(";  
	arg->print_node (reg_map, out);
	out << ")";
}

expr_node *function_node::unroll_expr (string s, int val, vector<string> coefficients, map<string,int> &scalar_count, bool is_lhs) {
	expr_node *new_arg = arg->unroll_expr (s, val, coefficients, scalar_count, is_lhs);
	return new function_node (name, new_arg);
}

expr_node *function_node::deep_copy (void) {
	expr_node *new_arg = arg->deep_copy ();
	return new function_node (name, new_arg, type, nested);
}

void function_node::populate_tree (accnode *acc_node, map <string, treenode*> &tree_map, map<string, int> label_bitset_index, int label_count, bool is_uminus) {
	acc_node->append_to_expr_string (name + "(");
	acc_node->append_to_rhs_printing (name + "(");
	arg->populate_tree (acc_node, tree_map, label_bitset_index, label_count, is_uminus);
	acc_node->append_to_expr_string (")"); 
	acc_node->append_to_rhs_printing (")"); 
}

void function_node::decompose_node(vector<tuple<expr_node*, expr_node*, STMT_OP>> &tstmt, vector<tuple<expr_node*, expr_node*, STMT_OP>> &init, vector<expr_node*> &temp_vars, expr_node *alhs, STMT_OP cur_op, int &id, DATA_TYPE gdata_type, bool &local_assigned, bool &global_assigned, bool flip) {
	// Visit the argument with a separate lhs
	string name_t = "_t_" + to_string (id++) + "_";
	expr_node *new_arg = new id_node (name_t);
	bool nested_local_assigned = false; 
	bool nested_global_assigned = false;
	bool nested_flip = false;
	arg->decompose_node (tstmt, init, temp_vars, new_arg, ST_EQ, id, gdata_type, nested_local_assigned, nested_global_assigned, nested_flip);
	expr_node *new_rhs = new function_node (name, new_arg);
	if (!local_assigned) cur_op = acc_start_op (cur_op);
	cur_op = get_cur_op (cur_op, flip);
	if (!global_assigned && cur_op != ST_EQ) 
		init.push_back (make_tuple (alhs, new datatype_node<int> (get_init_val (cur_op), INT), ST_EQ));
	tstmt.push_back (make_tuple (alhs, new_rhs, cur_op));
	local_assigned = true;
	global_assigned = true;
	// Now infer the types
	new_rhs->set_type (gdata_type);
	alhs->set_type (gdata_type);
	temp_vars.push_back (new_arg);
	temp_vars.push_back (new_rhs);
}
