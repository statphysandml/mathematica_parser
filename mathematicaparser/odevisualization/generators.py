import sys


def flow_equation_generator(folder, filename):
    with open(folder + "/" + filename, 'r') as f:
        lines = f.readlines()[0]
        # Crop new line command if present
        if lines[-1:] == "\n":
            lines = lines[:-1]
        equation_sets = lines[10:-1].split(", Equal")
        for equality in equation_sets:
            if equality[:11] == "[Derivative":
                yield "Equality", equality
            else:
                yield "InitialCondition", equality


def jacobian_generator(folder, filename):
    import re
    import numpy as np
    from mathematicaparser.core.flow_equation_parser import FlowEquationParser

    with open(folder + "/" + filename, 'r') as f:
        lines = f.readlines()[0]
        # Crop new line command
        if lines[-1:] == "\n":
            lines = lines[:-1]
        equation_sets = lines[10:-1].split(", List[")
        number_of_equations = len(equation_sets)
        for row_idx, row in enumerate(equation_sets):
            lbr, rbr = FlowEquationParser.find_actual_left_and_right_brace(row[:-1])
            starting_indices = np.array([m.start(0) for m in re.finditer("\[", row[lbr+1:-1])])
            ending_indices = np.array([m.end(0) for m in re.finditer("]", row[lbr+1:-1])])
            offset_indices = ending_indices[:-1][np.greater(starting_indices, ending_indices[:-1])]
            offset_indices = [-3 - lbr] + list(offset_indices) + [-2 - lbr]

            # Evaluate start and indices of the different expressions of the given list of equations
            expr_block_start_indices = list(np.array(offset_indices[:-1]) + lbr + 3)
            expr_block_end_indices = list(np.array(offset_indices[1:]) + lbr + 1)

            c = 0
            while len(expr_block_start_indices) < number_of_equations and c <= number_of_equations:
                actual_expr_block_start_indices = []
                actual_expr_block_end_indices = []
                for start_index, end_index in zip(expr_block_start_indices, expr_block_end_indices):
                    proposal_expr = row[start_index:end_index]
                    next_comma = proposal_expr.find(",")
                    last_comma = proposal_expr.rfind(",")
                    proposal_lbr, proposal_rbr = FlowEquationParser.find_actual_left_and_right_brace(proposal_expr)
                    if next_comma < proposal_lbr:
                        # There is another block element at the beginning of the proposal_expr
                        # Append new block element
                        actual_expr_block_start_indices.append(start_index)
                        actual_expr_block_end_indices.append(start_index + next_comma)
                        # Append adapted start index for considered proposal_expr
                        actual_expr_block_start_indices.append(start_index + next_comma + 2)
                        actual_expr_block_end_indices.append(end_index)
                    elif last_comma > proposal_rbr:
                        # There is another block element at the end of the proposal_expr
                        # Append adapted start index for considered proposal_expr
                        actual_expr_block_start_indices.append(start_index)
                        actual_expr_block_end_indices.append(start_index + last_comma)
                        # Append new block element
                        actual_expr_block_start_indices.append(start_index + last_comma + 2)
                        actual_expr_block_end_indices.append(end_index)
                    else:
                        actual_expr_block_start_indices.append(start_index)
                        actual_expr_block_end_indices.append(end_index)

                expr_block_start_indices = actual_expr_block_start_indices.copy()
                expr_block_end_indices = actual_expr_block_end_indices.copy()
                c += 1

            if c > number_of_equations:
                assert False, "Something went wrong in the jacobian_generator function - wrong number of extracted epxressions!"

            for col_idx, (start_index, end_index) in enumerate(zip(expr_block_start_indices, expr_block_end_indices)):
                yield str(row_idx) + ", " + str(col_idx), row[start_index:end_index]



def generate_equations(theory_name, equation_path=None, max_num_of_terms=9, max_num_of_term_operations=40):
    from mathematicaparser.core.flow_equation_parser import FlowEquationParser
    from mathematicaparser.odevisualization.flow_equation_meta_programmer import FlowEquationMetaProgrammer
    from mathematicaparser.odevisualization.jacobian_equation_meta_programmer import JacobianEquationMetaProgrammer

    FlowEquationParser.clear_static_variables()

    ''' Extract time variable and couplings and store results as static variables in FlowEquationParser '''

    raw_flow_equation_generator = flow_equation_generator(folder=equation_path, filename="flow_equations.txt")
    dim = 0
    for equality_kind, equality_content in raw_flow_equation_generator:
        print(equality_kind, equality_content)
        if equality_kind == "Equality":
            flow_equation_parser = FlowEquationParser(
                expression_kind="flow_equation",
                equality_content=equality_content
            )
            flow_equation_parser.extract_time_variable_and_couplings()
            dim += 1

    ''' Write thrust modules for the computation of the evaluation of the flow equations '''

    time_variable, couplings = FlowEquationParser.get_time_variable_and_couplings()

    FlowEquationMetaProgrammer.reset_counters()
    flow_equation_meta_programmer = FlowEquationMetaProgrammer(
        project_path=equation_path,
        theory_name=theory_name,
        dim=dim,
        time_variable=time_variable,
        coupling_names=couplings,
        base_struct_name="FlowEquation"
    )

    flow_equation_meta_programmer.write_hpp_header()
    flow_equation_meta_programmer.write_cu_header()

    dim_index = 0
    raw_flow_equation_generator = flow_equation_generator(folder=equation_path, filename="/flow_equations.txt")
    for equality_kind, equality_content in raw_flow_equation_generator:
        print(equality_kind, equality_content)
        if equality_kind == "Equality":
            flow_equation_parser = FlowEquationParser(
                expression_kind="flow_equation",
                equality_content=equality_content,
                max_num_of_terms=max_num_of_terms,
                max_num_of_term_operations=max_num_of_term_operations
            )
            flow_equation_parser.generated_binary_tree()
            flow_equation_parser.contract_operations()
            flow_equation_meta_programmer.write_flow_equation(
                dim_index=dim_index,
                flow_equation_parser=flow_equation_parser
            )
        dim_index += 1

    flow_equation_meta_programmer.write_hpp_footer()

    ''' Write thrust modules for the computation of the Jacobian matrix '''

    JacobianEquationMetaProgrammer.reset_counters()
    jacobian_equation_meta_programmer = JacobianEquationMetaProgrammer(
        project_path=equation_path,
        theory_name=theory_name,
        dim=dim,
        base_struct_name="JacobianEquation"
    )

    jacobian_equation_meta_programmer.write_hpp_header()
    jacobian_equation_meta_programmer.write_cu_header()

    raw_jacobian_generator = jacobian_generator(folder=equation_path, filename="jacobian.txt")
    for matrix_idx, (jacobian_idx, jacobian_element) in enumerate(raw_jacobian_generator):
        print(jacobian_idx, jacobian_element)
        jacobian_parser = FlowEquationParser(
            expression_kind="jacobian",
            jacobian_element=jacobian_element,
            max_num_of_terms=max_num_of_terms,
            max_num_of_term_operations=max_num_of_term_operations
        )
        jacobian_parser.generated_binary_tree()
        jacobian_parser.contract_operations()
        jacobian_equation_meta_programmer.write_flow_equation(
            dim_index=matrix_idx,
            flow_equation_parser=jacobian_parser
        )

    jacobian_equation_meta_programmer.write_hpp_footer()