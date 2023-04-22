import os
import screen as screen
import pygame as pg
from scm import plams
pg.display.set_icon(pg.image.load('icon.tiff'))
import matplotlib.pyplot as plt
import numpy as np
# import pdb
# pdb.set_trace() 

j = os.path.join


def show(mols=[], molinfo=None, simple=False, debug=False, background_color=(25, 25, 25), hide_hydrogens=False):
    if not isinstance(mols, list):
        mols = [mols]
    scr = screen.Screen(size=(1600, 900), background_color=background_color, hide_hydrogens=hide_hydrogens)
    if debug:
        scr.draw_mode = 'debug'
    else:
        if simple:
            scr.draw_mode = 'simple'
    scr.draw_molecules(mols, molinfo=molinfo)


# def show_results(results, simple=False):
#     # mols = [r.get_aligned_mol() for r in results]
#     mols = [r.get_mol() for r in results]
#     # [print(mol) for mol in mols]
#     mols = [molecule.load_plams_molecule(m)[0] for m in mols]
#     scr = screen.Screen(size=(1600, 900))
#     if simple:
#         scr.draw_mode = 'simple'
#     scr.draw_molecules(mols, loop=False)


def screen_shot_mols(mols, simple=False, background_color=(25, 25, 25), files=None, overwrite=True):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'
    if not isinstance(mols, list):
        mols = [mols]
    scr = screen.Screen(
        size=(
            1600,
            900),
        background_color=background_color,
        headless=True)
    if simple:
        scr.draw_mode = 'simple'
    for i, mol in enumerate(mols):
        try:
            if not overwrite and os.path.exists(files[i]):
                continue
            scr.draw_molecules([mol], loop=False, no_text=True)
            pg.image.save(scr.molecule_surf, files[i])
        except BaseException:
            print(mol)
            raise


def read_cub(file):
    with open(file) as cub:
        lines = [line.strip() for line in cub.readlines()]

    natoms, origin = int(lines[2].split()[0]), np.array(lines[2].split()[1:]).astype(float) * 0.52918
    xvec, yvec, zvec = np.array([line.split()[1:] for line in lines[3:6]]).astype(float) * 0.52918
    xn, yn, zn = np.array([line.split()[0] for line in lines[3:6]]).astype(int)
    atomcoords = []
    for line in lines[6:6+natoms]:
        atomcoords.append(np.array(line.split()[2:]).astype(float) * 0.52918)

    atomcoords = np.array(atomcoords)
    points = []
    for line in lines[6+natoms:]:
        points.extend(line.split())
    points = np.array(points).astype(float)

    limit = .03
    keep_idx = []
    colors = []
    for i, point in enumerate(points):
        if point >= limit:
            colors.append((255, 0, 0))
            keep_idx.append(i)
        elif point < -limit:
            colors.append((0, 0, 255))
            keep_idx.append(i)
        else:
            colors.append((0, 0, 0))

    pos = []
    for x in range(xn):
        for y in range(yn):
            for z in range(zn):
                pos.append(origin + x*xvec + y*yvec + z*zvec - np.mean(atomcoords, axis=0))
    pos = np.array(pos)
    pos[:, 1] *= -1

    colors = np.array(colors)
    pos = pos[keep_idx]
    colors = colors[keep_idx]
    return [pos, colors]


if __name__ == '__main__':
    mol = plams.Molecule(r"D:\Users\Yuman\Desktop\PhD\LewisAcid_coordination\calculations_final\EDA_vacuum\I2_N_pi\full\output.xyz")
    # mol = plams.Molecule()
    # mol.add_atom(plams.Atom(symbol='H', coords=[1, 1, 1]))
    mol.guess_bonds()

    # cube = [np.array([[1,1,1], [1,-1,1], [1,-1,-1], [1,1,-1], [-1,1,1], [-1,-1,1], [-1,-1,-1], [-1,1,-1]]), 
    #         np.array([[255, 255, 255]]*8)]
    # cube[0] = cube[0] + np.array([1, 1, 1])
    # cube[0] = cube[0] - np.mean(cube[0], axis=0)
    cube = read_cub(r"C:\Users\Yuman\Downloads\test%SCF_A%69.cub")

    show(mol, molinfo=[{'cub': cube}], simple=False, background_color=(25, 25, 25))
