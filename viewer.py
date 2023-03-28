import os
from yviewer.utility import paths
from yutility import molecule, timer
import yviewer.screen as screen
import pygame as pg

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
    # [mol.center() for mol in mols]
    scr.draw_molecules(mols, molinfo=molinfo)


def show_results(results, simple=False):
    # mols = [r.get_aligned_mol() for r in results]
    mols = [r.get_mol() for r in results]
    # [print(mol) for mol in mols]
    mols = [molecule.load_plams_molecule(m)[0] for m in mols]
    scr = screen.Screen(size=(1600, 900))
    if simple:
        scr.draw_mode = 'simple'
    scr.draw_molecules(mols, loop=False)


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


if __name__ == '__main__':
    files = [j(paths.test_molecules, 'batrachotoxin.xyz')]
    xyz = [molecule.load(file)['molecule'] for file in files]
    [mol.guess_bonds() for mol in xyz]
    inf = [molecule.load(file) for file in files]
    inf[0]['reaction'] = 'testReaction'
    show(simple=False, background_color=(25, 25, 25))
    timer.print_timings()
