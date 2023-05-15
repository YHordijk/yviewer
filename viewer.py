import os
import yviewer.screen as screen
from yviewer.utility import paths
import pygame as pg

# import pdb
# pdb.set_trace() 

j = os.path.join

pg.display.set_icon(pg.image.load(j(paths.base, 'icon.tiff')))


def show(mols=[], molinfo=None, simple=False, debug=False, background_color=(25, 25, 25), hide_hydrogens=False, update_funcs=[]):
    if not isinstance(mols, list):
        mols = [mols]
    scr = screen.Screen(size=(1600, 900), background_color=background_color, hide_hydrogens=hide_hydrogens)
    if debug:
        scr.draw_mode = 'debug'
    else:
        if simple:
            scr.draw_mode = 'simple'
    scr.draw_molecules(mols, molinfo=molinfo, update_funcs=update_funcs)


# def show_results(results, simple=False):
#     # mols = [r.get_aligned_mol() for r in results]
#     mols = [r.get_mol() for r in results]
#     # [print(mol) for mol in mols]
#     mols = [molecule.load_plams_molecule(m)[0] for m in mols]
#     scr = screen.Screen(size=(1600, 900))
#     if simple:
#         scr.draw_mode = 'simple'
#     scr.draw_molecules(mols, loop=False)


def screen_shot_mols(mols, files=None, simple=False, background_color=(25, 25, 25), overwrite=True, zoom=None, rotation=None):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'
    if not isinstance(mols, list):
        mols = [mols]

    scr = screen.Screen(size=(1600, 900), background_color=background_color, headless=True, rotation=rotation, zoom=zoom)
    
    if simple:
        scr.draw_mode = 'simple'
    for i, mol in enumerate(mols):
        try:
            if not overwrite and os.path.exists(files[i]):
                continue
            scr.draw_molecules([mol], loop=False, no_text=True)
            pg.image.save(scr.main_display, files[i])
        except BaseException:
            raise
    pg.display.quit()
    pg.quit()



if __name__ == '__main__':
    show(simple=False, background_color=(25, 25, 25))
