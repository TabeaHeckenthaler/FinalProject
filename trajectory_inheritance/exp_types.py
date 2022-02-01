
exp_types = {'SPT': {'ant': ['XL', 'L', 'M', 'S'], 'human': ['Large', 'Medium', 'Small Far', 'Small Near'],
                     'ps_simulation': ['XL', 'L', 'M', 'S']},
             'H': {'ant': ['XL', 'SL', 'L', 'M', 'S', 'XS']},
             'I': {'ant': ['XL', 'SL', 'L', 'M', 'S', 'XS']},
             'T': {'ant': ['XL', 'SL', 'L', 'M', 'S', 'XS']}}


def is_exp_valid(shape, solver, size):
    error_msg = 'Shape ' + shape + ', Solver ' + solver + ', Size ' + size + ' is not valid.'
    if shape not in exp_types.keys():
        raise ValueError(error_msg)
    if solver not in exp_types[shape].keys():
        raise ValueError(error_msg)
    if size not in exp_types[shape][solver]:
        raise ValueError(error_msg)
