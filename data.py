import os
import periodictable as pt
from yviewer.utility import paths

j = os.path.join


def csv_to_dict(file):
    d = {}
    with open(file, 'r') as f:
        for row in f.readlines():
            row = row.strip('\n')
            row = row.split(', ')
            d[int(row[0])] = row[1:]

    return d


MAX_PRINCIPAL_QUANTUM_NUMBER = csv_to_dict(j(paths.resources, 'elements', 'max_quantum_number.csv'))
MAX_VALENCE = csv_to_dict(j(paths.resources, 'elements', 'max_valence.csv'))
ATOM_COLOURS = csv_to_dict(j(paths.resources, 'elements', 'colours.csv'))
ATOM_COLOURS = {i: (int(c[0]), int(c[1]), int(c[2])) for i, c in ATOM_COLOURS.items()}
IONISATION_ENERGIES = csv_to_dict(j(paths.resources, 'elements', 'ionisation_energies.csv'))
COVALENT_RADII = {i: pt.elements[i].covalent_radius for i in range(96)}
ELECTRONEGATIVITIES = csv_to_dict(j(paths.resources, 'elements', 'electronegativities.csv'))


# forcefield parameters
def load_uff_parameters(path):
    uff_params = {}
    uff_params['valence_bond'] = {}
    uff_params['valence_angle'] = {}
    uff_params['nonbond_distance'] = {}
    uff_params['nonbond_scale'] = {}
    uff_params['nonbond_energy'] = {}
    uff_params['effective_charge'] = {}
    uff_params['sp3_torsional_barrier_params'] = {}
    uff_params['sp2_torsional_barrier_params'] = {}
    uff_params['electro_negativity'] = {}

    with open(path, 'r') as f:
        for row in f.readlines():
            parts = row.split()

            if len(parts) == 0:
                continue

            if parts[0] == 'param':
                el = parts[1]
                uff_params['valence_bond'][el] = float(parts[2])
                uff_params['valence_angle'][el] = float(parts[3])
                uff_params['nonbond_distance'][el] = float(parts[4])
                uff_params['nonbond_scale'][el] = float(parts[6])
                uff_params['nonbond_energy'][el] = float(parts[5])
                uff_params['effective_charge'][el] = float(parts[7])
                uff_params['sp3_torsional_barrier_params'][el] = float(parts[8])
                uff_params['sp2_torsional_barrier_params'][el] = float(parts[9])
                uff_params['electro_negativity'][el] = float(parts[10])

    return uff_params


FF_UFF_PARAMETERS = load_uff_parameters(j(paths.resources, 'forcefields', 'UFF', 'UFF.prm'))


def get_basis_file(name, extension='json'):
    f = rf'{paths.basis_set_dir}\{name}.{extension}'
    assert (os.path.exists(f))
    return f
