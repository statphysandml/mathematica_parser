import numpy as np
import pandas as pd
import os

from operator import itemgetter

from abc import ABCMeta, abstractmethod

from flow_equation_parser import FlowEquationParser


class IntermediateVectorManager:
    def __init__(self, couplings):
        self.couplings = couplings

        self.num_intermediate_vectors = 0
        self.intermediate_vectors = np.array([])
        self.global_iterator_map = dict()
        for coupling_index, key in enumerate(self.couplings):
            self.global_iterator_map[key] = "variables[" + str(coupling_index) + "]"
        self.num_couplings = len(self.couplings)

    def get_intermediate_vector(self):
        # Generate new intermediate vector
        if len(self.intermediate_vectors) == len(np.array(list(self.global_iterator_map.values()))) - self.num_couplings:
            self.intermediate_vectors = np.append(
                self.intermediate_vectors, "inter_med_vec" + str(self.num_intermediate_vectors))
            self.num_intermediate_vectors += 1
            return self.intermediate_vectors[-1]
        # Return existing intermediate vector
        else:
            used_intermediate_vectors = list(itemgetter(*self.get_actual_intermediate_vector_keys())(
                self.global_iterator_map))
            available_intermediate_vectors = [item for item in self.intermediate_vectors
                                              if item not in used_intermediate_vectors]
            return available_intermediate_vectors[-1]

    def get_actual_intermediate_vector_keys(self):
        keys = list(self.global_iterator_map.keys())
        return [item for item in keys if item not in self.couplings]

    def free_intermediate_vector(self, name):
        self.global_iterator_map.pop(name)


class ThrustMetaProgrammer:
    __metaclass__ = ABCMeta

    comp_functor_counter = 0

    def __init__(self, **kwargs):
        self.theory_name = kwargs.pop("theory_name")
        self.dim = kwargs.pop("dim")
        self.time_variable = kwargs.pop("time_variable", None)
        self.coupling_names = kwargs.pop("coupling_names", None)

        self.base_struct_name = kwargs.pop("base_struct_name")

        self.class_name = ''.join([item.capitalize() for item in self.theory_name.split(sep="_")])

        self.intermediate_vector_manager = None
        self.couplings = None

        self.header_file = None
        self.source_file = None

    def write_flow_equation(self, dim_index, flow_equation_parser):
        constant_expressions = flow_equation_parser.operation_tree_dataframe.query("type == 'constant expression'").value
        unique_constant_expressions = pd.unique(constant_expressions)
        self.couplings = np.array(flow_equation_parser.couplings)
        self.intermediate_vector_manager = IntermediateVectorManager(couplings=self.couplings)

        #  generate const expressions
        const_expr_map = dict()
        for idx, const_expr in enumerate(unique_constant_expressions):
            const_expr_map[const_expr] = 'const_expr' + str(idx) + "_"

        # generate transformations
        tree_root_index = flow_equation_parser.operation_tree_dataframe.query("depth == 0").iloc[0].name
        transformations, comp_functors = self.write_transformations(
            tree_index=tree_root_index,
            flow_equation_parser=flow_equation_parser,
            const_expr_map=const_expr_map)

        with open(self.header_file, "a") as f:
            # Struct name + constructor
            f.write('\nstruct ' + self.class_name + self.base_struct_name + str(dim_index) + ' : public odesolver::flowequations::' + self.base_struct_name + '\n'
                    '{\n'
                    '\t' + self.class_name + self.base_struct_name + str(dim_index) + '(const cudaT k) : k_(k)')

            for idx, const_expr in enumerate(unique_constant_expressions):
                f.write(',\n\t\tconst_expr' + str(idx) + "_" + const_expr)
            f.write('\n\t{}\n\n')

            # void operator()
            f.write('\tvoid operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;\n\n')

            # private variables
            f.write('private:\n'
                    '\tconst cudaT k_;')
            for idx, _ in enumerate(unique_constant_expressions):
                f.write('\n\tconst cudaT const_expr' + str(idx) + '_;')
            f.write('\n};\n\n')        # replace values of constant expressions by corresponding variable name

        with open(self.source_file, "a") as f:
            # write computation functors
            for comp_functor in reversed(comp_functors):
                f.write(comp_functor)

            # void operator()
            f.write('\nvoid ' + self.class_name + self.base_struct_name + str(dim_index) + '::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)\n'
                    '{\n')

            # write initialization of intermediate vectors
            for intermediate_vector in self.intermediate_vector_manager.intermediate_vectors:
                # f.write('\t\tauto * ' + intermediate_vector + ' = new dev_vec(derivatives.size());\n')
                f.write('\tdev_vec ' + intermediate_vector + '(derivatives.size());\n')

            # write transformations
            for transformation in reversed(transformations):
                f.write(transformation)

            f.write('}\n\n')

    def write_transformations(self, tree_index, flow_equation_parser, const_expr_map):
        current_entry = flow_equation_parser.operation_tree_dataframe.loc[tree_index]

        # Refer to the variables it will be iterated over in the transformation
        iterators = current_entry.eff_terms
        # Keys refer the keys of the local_iterator_map,
        if current_entry.type == "coupling":
            # Identity
            coupling = current_entry.value[:-len(FlowEquationParser.coupling_appendix)]
            local_iterator_map = {coupling: "val"}
            keys = [coupling]
            iterators = [coupling]
        elif current_entry.type in ["numeric constant", "constant expression"]:
            local_iterator_map = None
            keys = None
            iterators = []
        else:
            # The local_iterator_map contains a dictionary for the assigment of keys for the repsective iterators
            # to variables in the comp_functors or in the lambda expressions
            local_iterator_map, keys = self.prepare_transformation(iterators)
        # Determine the output_iterators of the resulting transformation
        if current_entry.value is None or current_entry.type in ["coupling", "numeric constant", "constant expression"]:
            output_iterator = 'derivatives.begin()'
            output_end_iterator = 'derivatives.end()'
        else:
            output_iterator = self.intermediate_vector_manager.global_iterator_map[current_entry.value] + '.begin()'
            output_end_iterator = self.intermediate_vector_manager.global_iterator_map[current_entry.value] + '.end()'

        # Determine the return expression of the resulting lambda expression
        return_expression, const_expressions, open_tree_ids = self.write_return_expression(
            tree_index=tree_index,
            flow_equation_parser=flow_equation_parser,
            local_iterator_map=local_iterator_map,
            const_expr_map=const_expr_map,
            initial=True
        )

        if current_entry.type not in ["coupling", "numeric constant", "constant expression"]:
            # strip ()-braces
            return_expression = return_expression[1:-1] + ";"

        # ToDo: Check if this if condition has any implications on other outputs!!!
        if return_expression == "val":
            return_expression = return_expression + ";"

        # Determine attributes of the resulting lambda expression
        if current_entry.type in ["numeric constant", "constant expression"]:
            attributes = ''
        elif len(iterators) <= 2:
            attributes = 'const cudaT &' + ', const cudaT &'.join(local_iterator_map.values())
        else:
            attributes = 'Tuple t'

        # Determine captures of the resulting lambda expression
        if len(const_expressions) > 0:
            # const_expressions = [const_expr + "=this." + const_expr for const_expr in const_expressions]
            # captures = ', &'.join(const_expressions)
            # captures = '&' + captures
            # captures = 'this'
            captures = const_expressions
        else:
            captures = ''

        # Write the actual transformation
        transformation, comp_functor = self.get_and_write_transformation(len(iterators))(
            output_iterator=output_iterator,
            output_end_iterator=output_end_iterator,
            captures=captures,
            attributes=attributes,
            return_expression=return_expression,
            keys=keys,
        )

        if current_entry.value in self.intermediate_vector_manager.get_actual_intermediate_vector_keys():
            self.intermediate_vector_manager.free_intermediate_vector(current_entry.value)
        # remove evaluated keys from global iterator map

        transformations = [transformation]
        comp_functors = [comp_functor] if comp_functor is not None else []
        for open_tree_id in open_tree_ids:
            transformation, comp_functor = self.write_transformations(
                tree_index=open_tree_id,
                flow_equation_parser=flow_equation_parser,
                const_expr_map=const_expr_map
            )
            transformations += transformation
            comp_functors += comp_functor
        return transformations, comp_functors

        # f.write('\t\tthrust::transform(')
        # variables[' + str(dim_index) + '].begin(), '
        #        'variables[' + str(dim_index) + '].end(), '
        # f.write('derivatives.begin(), )[](const cudaT &vertex) { return vertex; });\n')

    def prepare_transformation(self, iterators):
        if len(iterators) <= 2:
            local_iterator_map = {key: "val" + str(idx + 1) for idx, key in enumerate(iterators)}
        else:
            local_iterator_map = {key: "thrust::get<" + str(idx) + ">(t)" for idx, key in enumerate(iterators)}
        for key in iterators:
            if key not in self.couplings:
                self.intermediate_vector_manager.global_iterator_map[key] = \
                    self.intermediate_vector_manager.get_intermediate_vector()
        # assert np.equal(self.intermediate_vector_manager.global_iterator_map.keys(), local_iterator_map.keys()),\
        #     "Keys do not coincide"
        return local_iterator_map, list(local_iterator_map.keys())

    @staticmethod
    def write_comp_functor(theory_name, captures, attributes, return_expression, return_index=None):
        if len(captures) == 0:
            string = str('\nstruct comp_func_' + theory_name + str(ThrustMetaProgrammer.comp_functor_counter) + '\n'
                         '{\n'
                         '\tcomp_func_' + theory_name + str(ThrustMetaProgrammer.comp_functor_counter) + '()\n'
                         '\t{}\n\n')
        else:
            string = str('\nstruct comp_func_' + theory_name + str(ThrustMetaProgrammer.comp_functor_counter) + '\n'
                         '{\n'
                         '\tconst cudaT ' + ';\n\tconst cudaT '.join(captures) + ';\n\n'
                         '\tcomp_func_' + theory_name + str(ThrustMetaProgrammer.comp_functor_counter) + '(const cudaT ' + ', const cudaT '.join([const_expr[:-1] for const_expr in captures]) + ')\n'
                         '\t\t: ' + ', '.join([const_expr + '(' + const_expr[:-1] + ')' for const_expr in captures]) + ' {}\n\n')
        if return_index is not None:
            string += '\ttemplate <typename Tuple>\n'
            string += str('\t__host__ __device__\n'
                          '\tvoid operator()(' + attributes + ')\n'
                          '\t{\n')
        else:
            string += str('\t__host__ __device__\n'
                          '\tcudaT operator()(' + attributes + ')\n'
                          '\t{\n')
        if return_index is not None:
            string += str('\t\tthrust::get<' + str(return_index) + '>(t) = ' + return_expression + '\n')
        else:
            string += str('\t\treturn ' + return_expression + '\n')
        string += str('\t}\n'
                      '};\n\n')
        comp_functor_name = 'comp_func_' + theory_name + str(ThrustMetaProgrammer.comp_functor_counter)
        ThrustMetaProgrammer.comp_functor_counter += 1
        return comp_functor_name, string

    def write_scalar_transformation(self, output_iterator, output_end_iterator, captures, attributes,
                                   return_expression, keys):
        comp_functor_string = None
        string = str('\tthrust::fill(thrust::device, ' + output_iterator + ", " + output_end_iterator + ", " + return_expression + ');\n')
        return string, comp_functor_string

    def write_unary_transformation(self, output_iterator, output_end_iterator, captures, attributes,
                                   return_expression, keys):
        # identity operator
        # f.write('\t\tthrust::transform(variables[' + str(dim_index) + '].begin(), '
        #         'variables[' + str(dim_index) + '].end(), '
        #         'derivatives.begin(), [](const cudaT &vertex) { return vertex; });\n')
        if len(captures) == 0:
            comp_functor_string = None
            string = str('\tthrust::transform(' +
                         self.intermediate_vector_manager.global_iterator_map[keys[0]] + '.begin(), ' +
                         self.intermediate_vector_manager.global_iterator_map[keys[0]] + '.end(), ' +
                         output_iterator +
                         ', [] __host__ __device__ (' + attributes + ') { return ' + return_expression + ' });\n')
        else:
            comp_functor_name, comp_functor_string = ThrustMetaProgrammer.write_comp_functor(self.theory_name, captures, attributes, return_expression)
            string = str('\tthrust::transform(' +
                         self.intermediate_vector_manager.global_iterator_map[keys[0]] + '.begin(), ' +
                         self.intermediate_vector_manager.global_iterator_map[keys[0]] + '.end(), ' +
                         output_iterator +
                         ', ' + comp_functor_name + '(' + ', '.join(captures) + '));\n')
        return string, comp_functor_string

    def write_binary_transformation(self, output_iterator, output_end_iterator, captures, attributes,
                                    return_expression, keys):
        if len(captures) == 0:
            comp_functor_string = None
            string = str('\tthrust::transform(' +
                         self.intermediate_vector_manager.global_iterator_map[keys[0]] + '.begin(), ' +
                         self.intermediate_vector_manager.global_iterator_map[keys[0]] + '.end(), ' +
                         self.intermediate_vector_manager.global_iterator_map[keys[1]] + '.begin(), ' +
                         output_iterator +
                         ', [] __host__ __device__ (' + attributes + ') { return ' + return_expression + ' });\n')
        else:
            comp_functor_name, comp_functor_string = ThrustMetaProgrammer.write_comp_functor(self.theory_name, captures, attributes, return_expression)
            string = str('\tthrust::transform(' +
                         self.intermediate_vector_manager.global_iterator_map[keys[0]] + '.begin(), ' +
                         self.intermediate_vector_manager.global_iterator_map[keys[0]] + '.end(), ' +
                         self.intermediate_vector_manager.global_iterator_map[keys[1]] + '.begin(), ' +
                         output_iterator +
                         ', ' + comp_functor_name + '(' + ', '.join(captures) + '));\n')
        return string, comp_functor_string

    def write_arbitrary_transformation(self, output_iterator, output_end_iterator, captures, attributes,
                                       return_expression, keys):
        comp_functor_name, comp_functor_string = ThrustMetaProgrammer.write_comp_functor(self.theory_name, captures, attributes, return_expression, len(keys))
        string = str('\tthrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(')
        string += ('.begin(), '.join(self.intermediate_vector_manager.global_iterator_map[key] for key in keys) +
                   '.begin(), ' + output_iterator + ')),')
        string += 'thrust::make_zip_iterator(thrust::make_tuple('
        string += ('.end(), '.join(self.intermediate_vector_manager.global_iterator_map[key] for key in keys) +
                   '.end(), ' + output_end_iterator + '))'
                                                       ', ' + comp_functor_name + '(' + ', '.join(captures) + '));\n')
        return string, comp_functor_string

    def get_and_write_transformation(self, number_of_terms):
        if number_of_terms == 0:
            return self.write_scalar_transformation
        if number_of_terms == 1:
            return self.write_unary_transformation
        elif number_of_terms == 2:
            return self.write_binary_transformation
        else:
            return self.write_arbitrary_transformation

    def write_return_expression(self, tree_index, flow_equation_parser, local_iterator_map, const_expr_map, initial=False):
        const_expressions = []
        open_tree_ids = []

        current_entry = flow_equation_parser.operation_tree_dataframe.loc[tree_index]
        operation, child_ids, value, type = current_entry[["operation", "child_ids", "value", "type"]].values

        # Found leaf
        if value is not None:
            if type == "coupling":
                return local_iterator_map[value[:-len(FlowEquationParser.coupling_appendix)]], const_expressions, open_tree_ids
            elif type == "numeric constant":
                return value, const_expressions, open_tree_ids
            elif type == "constant expression":
                const_expressions.append(const_expr_map[value])
                return const_expr_map[value], const_expressions, open_tree_ids
            elif type == "intermediate expression" and value in local_iterator_map.keys():
                open_tree_ids.append(tree_index)
                return local_iterator_map[value], const_expressions, open_tree_ids

        args = []
        for child_id in child_ids:
            content, new_const_expressions, new_open_tree_ids = self.write_return_expression(
                tree_index=child_id,
                flow_equation_parser=flow_equation_parser,
                local_iterator_map=local_iterator_map,
                const_expr_map=const_expr_map,
                initial=False
            )
            args.append(content)
            const_expressions += new_const_expressions
            open_tree_ids += new_open_tree_ids
        return FlowEquationParser.plain_operations[operation](args), list(np.unique(const_expressions)), open_tree_ids


class FlowEquationMetaProgrammer(ThrustMetaProgrammer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.custom = kwargs.pop("custom")
        if self.custom:
            self.header_file = kwargs.get("project_path") + "/flow_equations/" +\
                               self.theory_name + "/" + self.theory_name + "_flow_equation.hpp"
            self.source_file = kwargs.get("project_path") + "/flow_equations/" +\
                               self.theory_name + "/" + self.theory_name + "_flow_equation.cu"
            self.flow_equations_directory = kwargs.get("project_path") + "/flow_equations/"
        else:
            self.header_file = "../flow_equations/" + self.theory_name + "_flow_equation.hpp"
            self.source_file = "../flow_equations/" + self.theory_name + "_flow_equation.cu"
            self.flow_equations_directory = kwargs.get("project_path") + "/flow_equations/"

        self.ode_solver_path = kwargs.pop("ode_solver_path")

    def write_hpp_header(self):
        ifndef_name = ''.join([item.upper() for item in self.theory_name.split(sep="_")])

        with open(self.header_file, "w") as f:
            f.write('#ifndef PROJECT_' + ifndef_name + 'FLOWEQUATION_HPP\n'
                    '#define PROJECT_' + ifndef_name + 'FLOWEQUATION_HPP\n\n'
                    '#include <math.h>\n'
                    '#include <tuple>\n\n'
                    '#include <odesolver/flow_equations/flow_equation.hpp>\n\n')

    def write_hpp_footer(self):
        ifndef_name = ''.join([item.upper() for item in self.theory_name.split(sep="_")])

        with open(self.header_file, "a") as f:
            f.write('\nclass ' + self.class_name + self.base_struct_name + 's : public odesolver::flowequations::' + self.base_struct_name + 'sWrapper\n'
                    '{\n'
                    'public:\n'
                    '\t' + self.class_name + self.base_struct_name + 's(const cudaT k) : k_(k)\n'
                    '\t{\n'
                    '\t\tflow_equations_ = std::vector<std::shared_ptr<odesolver::flowequations::' + self.base_struct_name + '>> {\n')
            for dim_index in range(self.dim-1):
                f.write('\t\t\tstd::make_shared<' + self.class_name + self.base_struct_name + str(dim_index) + '>(k_),\n')
            f.write('\t\t\tstd::make_shared<' + self.class_name + self.base_struct_name + str(self.dim-1) + '>(k_)\n\t\t};\n')
            f.write('\t}\n\n'
                    '\tvoid operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, '
                    'const int dim_index) override\n'
                    '\t{\n'
                    '\t\t(*flow_equations_[dim_index])(derivatives, variables);\n'
                    '\t}\n\n'
                    '\tsize_t get_dim() override\n'
                    '\t{\n'
                    '\t\treturn dim_;\n'
                    '\t}\n\n'
                    '\tstatic std::string model_;\n'
                    '\tstatic size_t dim_;\n'
                    '\tstatic std::string explicit_variable_;\n'
                    '\tstatic std::vector<std::string> explicit_functions_;\n'
                    '\n'
                    '\tjson get_json() const override\n'
                    '\t{\n'
                    '\t\treturn json {\n'
                    '\t\t\t{"model", model_},\n'
                    '\t\t\t{"dim", dim_},\n'
                    '\t\t\t{"explicit_variable", explicit_variable_},\n'
                    '\t\t\t{"explicit_functions", explicit_functions_}\n'
                    '\t\t};\n'
                    '\t}\n\n'
                    'private:\n'
                    '\tconst cudaT k_;\n'
                    '\tstd::vector<std::shared_ptr<odesolver::flowequations::' + self.base_struct_name + '>> flow_equations_;\n'
                    '};\n\n'
                    '#endif //PROJECT_' + ifndef_name + 'FLOWEQUATION_HPP\n')

    def write_cu_header(self):
        with open(self.source_file, "w") as f:
            f.write('#include "' + self.theory_name + '_flow_equation.hpp"\n\n')
            f.write('std::string ' + self.class_name + self.base_struct_name + 's::model_ = "' + self.theory_name + '";\n'
                    'size_t ' + self.class_name + self.base_struct_name + 's::dim_ = ' + str(self.dim) + ';\n'
                    'std::string ' + self.class_name + self.base_struct_name + 's::explicit_variable_ = "' + self.time_variable + '";\n'
                    'std::vector<std::string> ' + self.class_name + self.base_struct_name + 's::explicit_functions_ = {"' + '", "'.join(self.coupling_names) + '"};\n\n')


class JacobianEquationMetaProgrammer(ThrustMetaProgrammer):

    def __init__(self, **kwargs):
        super(JacobianEquationMetaProgrammer, self).__init__(**kwargs)

        self.custom = kwargs.pop("custom")
        if self.custom:
            self.header_file = kwargs.get("project_path") + "/flow_equations/" +\
                               self.theory_name + "/" + self.theory_name + "_jacobian_equation.hpp"
            self.source_file = kwargs.get("project_path") + "/flow_equations/" +\
                               self.theory_name + "/" + self.theory_name + "_jacobian_equation.cu"
            self.jacobian_equations_directory = kwargs.get("project_path") + "/flow_equations/"
        else:
            self.header_file = "../flow_equations/source/" + self.theory_name + "_jacobian_equation.hpp"
            self.source_file = "../flow_equations/source/" + self.theory_name + "_jacobian_equation.cu"
            self.jacobian_equations_directory = kwargs.get("project_path") + "/flow_equations/"

        self.ode_solver_path = kwargs.pop("ode_solver_path")

    def write_hpp_header(self):
        ifndef_name = ''.join([item.upper() for item in self.theory_name.split(sep="_")])

        with open(self.header_file, "w") as f:
            f.write('#ifndef PROJECT_' + ifndef_name + 'JACOBIAN_HPP\n'
                    '#define PROJECT_' + ifndef_name + 'JACOBIAN_HPP\n\n'
                    '#include <math.h>\n'
                    '#include <tuple>\n\n'
                    '#include <odesolver/flow_equations/jacobian_equation.hpp>\n\n')

    def write_hpp_footer(self):
        ifndef_name = ''.join([item.upper() for item in self.theory_name.split(sep="_")])

        with open(self.header_file, "a") as f:
            f.write('\nclass ' + self.class_name + self.base_struct_name + 's : public odesolver::flowequations::' + self.base_struct_name + 'sWrapper\n'
                    '{\n'
                    'public:\n'
                    '\t' + self.class_name + self.base_struct_name + 's(const cudaT k) : k_(k)\n'
                    '\t{\n'
                    '\t\tjacobian_equations_ = std::vector<std::shared_ptr<odesolver::flowequations::' + self.base_struct_name + '>> {\n')
            for dim_index in range(pow(self.dim, 2)-1):
                f.write('\t\t\tstd::make_shared<' + self.class_name + self.base_struct_name + str(dim_index) + '>(k),\n')
            f.write('\t\t\tstd::make_shared<' + self.class_name + self.base_struct_name + str(pow(self.dim, 2)-1) + '>(k)\n\t\t};\n')
            f.write('\t}\n\n'
                    '\tvoid operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, '
                    'const int row_idx, const int col_idx) override\n'
                    '\t{\n'
                    '\t\t(*jacobian_equations_[row_idx * dim_ + col_idx])(derivatives, variables);\n'
                    '\t}\n\n'
                    '\tvoid operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, '
                    'const int matrix_idx) override\n'
                    '\t{\n'
                    '\t\t(*jacobian_equations_[matrix_idx])(derivatives, variables);\n'
                    '\t}\n\n'
                    '\tsize_t get_dim() override\n'
                    '\t{\n'
                    '\t\treturn dim_;\n'
                    '\t}\n\n'
                    '\tstatic std::string model_;\n'
                    '\tstatic size_t dim_;\n'
                    '\n'
                    'private:\n'
                    '\tconst cudaT k_;\n'
                    '\tstd::vector<std::shared_ptr<odesolver::flowequations::' + self.base_struct_name + '>> jacobian_equations_;\n'
                    '};\n\n'
                    '#endif //PROJECT_' + ifndef_name + 'JACOBIAN_HPP\n')

    def write_cu_header(self):
        with open(self.source_file, "w") as f:
            f.write('#include "' + self.theory_name + '_jacobian_equation.hpp"\n\n')
            f.write('std::string ' + self.class_name + self.base_struct_name + 's::model_ = "' + self.theory_name + '";\n'
                    'size_t ' + self.class_name + self.base_struct_name + 's::dim_ = ' + str(self.dim) + ';\n\n')
