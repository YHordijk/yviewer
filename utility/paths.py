import os
from yutility import pathfunc

j = os.path.join
dn = os.path.dirname

base = dn(dn(__file__))
data = j(base, 'data')
resources = j(data, 'resources')
test_molecules = j(resources, 'xyz')
basis_set_dir = j(resources, 'basissets')


__all__ = [base,
           test_molecules,
           ]

if __name__ == '__main__':
    pathfunc.check_paths(__all__)
    pathfunc.print_paths(__all__)
