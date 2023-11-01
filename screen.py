import pygame as pg
import pygame.locals
import numpy as np
from yviewer import data
import periodictable as pt
import skimage.draw
import matplotlib.pyplot as plt
from time import perf_counter
from math import cos, sin
from yutility import geometry, molecule, timer
import pyperclip
from scm import plams


def l2_norm(u, v): return np.linalg.norm(u - v)

# atom_img = './data/images/atom_default.png'


class Screen:
    def __init__(self, *args, **kwargs):
        self.size = kwargs.get('size', (500, 300))
        self.background_color = kwargs.get('background_color', (25, 25, 25))
        self.headless = kwargs.get('headless', False)
        self.projection_mode = kwargs.get('projection_mode', 'perspective')
        pg.display.init()
        # if not self.headless:
        self.main_display = pg.display.set_mode(self.size, pg.locals.HWSURFACE | pg.locals.DOUBLEBUF | pg.locals.RESIZABLE)
        self.molecule_surf = pg.surface.Surface(self.size, pg.locals.HWSURFACE | pg.locals.DOUBLEBUF | pg.locals.RESIZABLE)
        # else:
        #     self.main_display = pg.display.set_mode(self.size, pg.locals.SRCALPHA)
        #     self.molecule_surf = pg.surface.Surface(self.size, pg.locals.SRCALPHA).convert()
        self.camera_position = [0, 0]
        self.camera_orientation = [0, 0, 0]
        self.camera_z = 6
        self.time = 0
        self.screen_center = (self.size[0] / 2, self.size[1] / 2)
        self.draw_mode = 'normal'
        self.texts = kwargs.get('texts', [])
        self.hide_hydrogens = kwargs.get('hide_hydrogens', False)
        self.init_options = {'zoom': kwargs.get('zoom'), 'rotation': kwargs.get('rotation')}
        self.set_projection_plane()

    def add_mol(self, mol, molinfo=None):
        self.prepare_atom_bonds_imgs(mol)
        self.positions[mol] = np.array([atom.coords for atom in mol.atoms])
        self.original_positions[mol] = np.array([atom.coords for atom in mol.atoms])
        self.bond_tuples[mol] = self.get_bond_tuples(mol)
        if molinfo is None:
            molinfo = {}
        self.molinfo.append(molinfo)
        self.mols.append(mol)

    def delete_mol(self, molidx):
        mol = self.mols[molidx]
        self.positions.pop(mol)
        self.original_positions.pop(mol)
        self.bond_tuples.pop(mol)
        self.molinfo.pop(molidx)
        self.mols.pop(molidx)

    @timer.Time
    def prepare_atom_bonds_imgs(self, mol, pos=(.3, .3)):
        def gaussian(size, pos, m, one_dim=False):
            x, y = np.meshgrid(np.linspace(0, 1, size[1]) - pos[0], np.linspace(0, 1, size[0]) - pos[1])
            if one_dim:
                dst = np.sqrt(y * y)
            else:
                dst = np.sqrt(x * x + y * y)
            gauss = np.exp(-((dst) / m)**2) + 0.1
            return gauss

        def generate_atom_img(n, size=500):
            r = size
            c = data.ATOM_COLOURS[n]
            atom_img = np.zeros((r, r))

            # draw circle
            rr, cc = skimage.draw.disk((int(r / 2), int(r / 2)), int(r / 2), shape=(r, r))
            atom_img[rr, cc] = 1

            # add gaussians as highlight
            gauss1 = gaussian((r, r), pos, .25)
            gauss2 = 0.5 * gaussian((r, r), pos, .75)
            gauss = (gauss1 + gauss2) / np.max(gauss1 + gauss2)  # double gaussian highlight

            # define surface to draw atom_img to
            surf = pg.surface.Surface((r, r))
            a = gauss * atom_img
            a = (a / a.max() * 255).astype(int)  # normalize
            # code for rgb in pygame internal
            pg.pixelcopy.array_to_surface(surf, a + a * 256 + a * 256 * 256)
            atom_img = surf

            # atom_img.set_colorkey(self.background_color)
            atom_img.set_colorkey((0, 0, 0))
            atom_img.fill(c, special_flags=pg.locals.BLEND_RGB_MULT)

            return atom_img

        def generate_single_bond_img(n1, n2, size=20):
            size = (size, 20)
            half_size = (size[0], size[1] // 2)

            c1 = data.ATOM_COLOURS[n1]
            c2 = data.ATOM_COLOURS[n2]

            # generate bond_img
            bond_img = np.ones(half_size)

            gauss1 = gaussian(half_size, (.5 / 2, .5), .25, one_dim=True)
            gauss2 = 0.5 * gaussian(half_size, (.5 / 2, .5), .75, one_dim=True)
            gauss = (gauss1 + gauss2) / np.max(gauss1 + gauss2)  # double gaussian highlight

            surf = pg.surface.Surface(size, pg.locals.SRCALPHA)
            # surf.fill(self.background_color)
            surf_half = pg.surface.Surface(half_size, pg.locals.SRCALPHA)
            a = gauss * bond_img
            a = ((a - a.min()) / a.max() * 255).astype(int)  # normalize

            pg.pixelcopy.array_to_surface(surf_half, a + a * 256 + a * 256 * 256 + a * 256 * 256 * 256)
            surf1 = surf_half.copy()
            surf2 = surf_half.copy()

            surf1.fill(c1, special_flags=pg.locals.BLEND_RGB_ADD)
            surf2.fill(c2, special_flags=pg.locals.BLEND_RGB_ADD)

            surf.blit(surf1, (0, 0))
            surf.blit(surf2, (0, size[1] // 2))

            return surf

        def generate_aromatic_bond_img(
                n1, n2, size=40, spacing=4, sub_len=4, offset=4):
            size = (size, 20)
            sbn_size = ((size[0]) // 2, 20)
            sbn_img = generate_single_bond_img(n1, n2, size[0])

            surf = pg.surface.Surface(size, pg.locals.SRCALPHA)
            sbn_img1 = pg.transform.scale(sbn_img.copy(), sbn_size)
            sbn_img2 = pg.transform.scale(sbn_img.copy(), (size[0], sub_len))

            surf.blit(sbn_img1, (0, 0))
            for i in range(size[0] // (sub_len + offset)):
                surf.blit(sbn_img2, (sbn_size[0],
                          offset * (i + 1) + sub_len * i))

            return surf

        def generate_double_bond_img(n1, n2, size=40, spacing=4):
            size = (size, 20)
            sbn_size = ((size[0]) // 2, 20)
            sbn_img = generate_single_bond_img(n1, n2, size[0])

            surf = pg.surface.Surface(size, pg.locals.SRCALPHA)
            sbn_img = pg.transform.scale(sbn_img, sbn_size)

            surf.blit(sbn_img, (0, 0))
            surf.blit(sbn_img, (sbn_size[0], 0))

            return surf

        def generate_triple_bond_img(n1, n2, size=60, spacing=4):
            size = (size, 20)
            sbn_size = ((size[0]) // 3, 20)

            sbn_img = generate_single_bond_img(n1, n2, size[0])

            surf = pg.surface.Surface(size, pg.locals.SRCALPHA)
            sbn_img = pg.transform.scale(sbn_img, sbn_size)

            surf.blit(sbn_img, (0, 0))
            surf.blit(sbn_img, (sbn_size[0], 0))
            surf.blit(sbn_img, (sbn_size[0] * 2, 0))

            return surf

        anum = set(atom.atnum for atom in mol.atoms)

        at_imgs = {}
        for n in anum:
            im = generate_atom_img(n)
            at_imgs[n] = im

        single_bn_imgs = {}
        aromatic_bn_imgs = {}
        double_bn_imgs = {}
        triple_bn_imgs = {}
        for n1 in anum:
            for n2 in anum:
                single_bn_imgs[(n1, n2)] = generate_single_bond_img(
                    n1, n2, 50)
                aromatic_bn_imgs[(n1, n2)] = generate_aromatic_bond_img(
                    n1, n2, 50)
                double_bn_imgs[(n1, n2)] = generate_double_bond_img(
                    n1, n2, 50)
                triple_bn_imgs[(n1, n2)] = generate_triple_bond_img(
                    n1, n2, 50)

        self.atom_imgs.append(at_imgs)
        self.single_bond_imgs.append(single_bn_imgs)
        self.aromatic_bond_imgs.append(aromatic_bn_imgs)
        self.double_bond_imgs.append(double_bn_imgs)
        self.triple_bond_imgs.append(triple_bn_imgs)

    def get_rotation_matrix(self, x=None, y=None, z=None):
        R = np.eye(3)

        if not x is None:
            c = cos(x)
            s = sin(x)
            R = R @ np.array(([1, 0, 0],
                              [0, c, -s],
                              [0, s, c]))

        if not y is None:
            c = cos(y)
            s = sin(y)
            R = R @ np.array(([c, 0, s],
                              [0, 1, 0],
                              [-s, 0, c]))

        if not z is None:
            c = cos(z)
            s = sin(z)
            R = R @ np.array(([c, -s, 0],
                              [s, c, 0],
                              [0, 0, 1]))

        return R

    def project(self, points, settings={}):
        if self.projection_mode == 'orthographic':
            if len(points.shape) == 1:
                points = points[0:2]
            else:
                points = points[:, 0:2]
            return np.hstack(((points * 100 + self.camera_position).T,
                             (points * 100 + self.camera_position).T)).T.astype(int)

        if self.projection_mode == 'perspective':
            d = self.get_rotation_matrix(*self.camera_orientation) @ (points - np.asarray([*self.camera_position, self.camera_z])).T
            f = self.projection_plane @ d
            x, y = f[0] / f[2], f[1] / f[2]
            return np.vstack([self.size[0] - x, y]).T.astype(int)

    def set_projection_plane(self):
        e = np.array([self.size[0] / 2, self.size[1] / 2, 600])
        self.projection_plane = np.array([[1, 0, e[0] / e[2]], [0, 1, e[1] / e[2]], [0, 0, 1 / e[2]]])

    def get_bond_tuples(self, mol):
        tuples = []
        # mol.guess_bonds()
        B = mol.bond_matrix()
        for i, a1 in enumerate(B):
            for j, a2 in enumerate(a1[i + 1:]):
                if a2 > 0:
                    tuples.append((i, i + j + 1))
        return tuples

    def draw_molecules(self, mols, molinfo=None,
                       settings={}, loop=True, no_text=False, update_funcs=[]):
        self.mols = []
        self.atom_imgs = []
        self.single_bond_imgs = []
        self.aromatic_bond_imgs = []
        self.double_bond_imgs = []
        self.triple_bond_imgs = []
        self.positions = {}
        self.original_positions = {}
        self.bond_tuples = {}
        self.molinfo = []
        self.no_text = no_text
        self.update_funcs = update_funcs
        state = {}
        state['self'] = self
        state['run'] = True
        state['molidx'] = 0
        state['mols'] = self.mols
        state['molinfo'] = self.molinfo

        if molinfo is None:
            molinfo = [None for _ in mols]

        [self.add_mol(mol, molinfo=inf) for mol, inf in zip(mols, molinfo)]

        md = self.main_display
        ms = self.molecule_surf

        self.init_loop(state)

        while state['run']:
            md.fill(self.background_color)
            self.pre_update(state)
            self.update(state)
            self.post_update(state)

            with timer.Timer('Screen.draw_molecules.blit'):
                md.blit(ms, (0, 0))
            with timer.Timer('Screen.draw_molecules.update'):
                pg.display.update()

            if not loop:
                return

        pg.display.quit()
        pg.quit()

    def _prepare_molecule_surf(self, molidx, smooth_bonds=False, atom_radius_factor=.5):
        if len(self.mols) == 0:
            return
        mol = self.mols[molidx]

        def rotate_image(image, pos, originPos, angle):
            image_rect = image.get_rect(
                topleft=(
                    pos[0] - originPos[0],
                    pos[1] - originPos[1]))
            offset_center_to_pivot = pg.math.Vector2(
                list(pos)) - image_rect.center
            rotated_offset = offset_center_to_pivot.rotate(-angle)
            rotated_image_center = (
                pos[0] - rotated_offset.x,
                pos[1] - rotated_offset.y)
            rotated_image = pg.transform.rotate(image, angle)
            rotated_image_rect = rotated_image.get_rect(
                center=rotated_image_center)
            return rotated_image, rotated_image_rect

        if self.draw_mode.lower() == 'normal':
            draw_bond_rect = False
            draw_bond_center = False
            draw_bond_through_atom = False
            simple_atoms = False
            simple_bonds = False
        if self.draw_mode.lower() == 'simple':
            draw_bond_rect = False
            draw_bond_center = False
            draw_bond_through_atom = False
            simple_atoms = True
            simple_bonds = True
        if self.draw_mode.lower() == 'debug':
            draw_bond_rect = True
            draw_bond_center = True
            draw_bond_through_atom = True
            simple_atoms = True
            simple_bonds = True

        blits = []
        pos = self.positions[mol].copy()
        # pos[:, 1] *= -1
        dist_to_cam = np.sqrt(np.sum((pos - (*self.camera_position, self.camera_z))**2, axis=1))
        idx = np.argsort(dist_to_cam)[::-1]
        atn = [atom.atnum for atom in mol.atoms]
        atcolours = [data.ATOM_COLOURS[n] for n in atn]

        # zero = self.project(np.array([0,0,0]))
        # r = (mol.radii/dist_to_cam * 800).astype(int)
        radii = np.array([pt.elements[n].covalent_radius for n in atn])
        rad = np.zeros((len(radii), 3))
        rad[:, 0] = 1
        rad = rad * radii.reshape(-1, 1) * atom_radius_factor
        offset_pos = rad + pos

        mapped_positions = self.project(pos)
        mapped_radii = self.project(offset_pos)
        r = abs((mapped_radii - mapped_positions)[:, 0])

        for i in idx:
            n = atn[i]
            a = mapped_positions[i]

            if n == 1 and self.hide_hydrogens:
                blits.append(None)
                continue

            if simple_atoms:
                pg.draw.circle(self.molecule_surf, atcolours[i], a, r[i])
            else:
                atom_img = self.atom_imgs[molidx][n]
                if int(r[i] * 2) * int(r[i] * 2) > 300_000:
                    continue
                if int(r[i] * 2) * int(r[i] * 2) > 100_000:
                    f = 1 - (int(r[i] * 2) * int(r[i] * 2) - 100_000)/200_000
                    atom_img.set_alpha(f*255)
                
                atom_img = pg.transform.scale(atom_img, (int(r[i] * 2), int(r[i] * 2)))
                blits.append((atom_img, a - r[i]))

        # moving on to bonds
        tuples = np.asarray(self.bond_tuples[mol])
        if len(tuples) > 0:
            ra = mapped_positions[tuples[:, 0]]
            rb = mapped_positions[tuples[:, 1]]
            rabn = (rb - ra)
            B = mol.bond_matrix()

            atom_indices_in_blits = [i for i in idx]
            for i, (a, b) in enumerate(self.bond_tuples[mol]):
                n1 = atn[a]
                n2 = atn[b]

                if self.hide_hydrogens:
                    if n1 == 1 or n2 == 1:
                        continue

                radius = int((r[b] + r[a]) / 5)

                pab = (pos[b] - pos[a])
                npab = pab / np.linalg.norm(pab)
                X = pos[a] + npab * radii[a] * atom_radius_factor
                mapped_on_sphere1 = self.project(X)[0]
                X = pos[b] - npab * radii[b] * atom_radius_factor
                mapped_on_sphere2 = self.project(X)[0]
                mapped_bond_dist = np.linalg.norm(mapped_on_sphere1 - mapped_on_sphere2)
                bond_len = max(0, mapped_bond_dist)

                if bond_len > 0:
                    bond_center = mapped_positions[a] + (mapped_positions[b] - mapped_positions[a]) / 2
                    new_scale = (int(radius * B[a, b]), int(bond_len))
                    if B[a, b] == 1:
                        bond_img = self.single_bond_imgs[molidx][(
                            n1, n2)].copy()
                    if B[a, b] == 1.5:
                        bond_img = self.aromatic_bond_imgs[molidx][(
                            n1, n2)].copy()
                    if B[a, b] == 2:
                        bond_img = self.double_bond_imgs[molidx][(
                            n1, n2)].copy()
                    if B[a, b] == 3:
                        bond_img = self.triple_bond_imgs[molidx][(
                            n1, n2)].copy()

                    new_scale = (max(1, new_scale[0]), max(1, new_scale[1]))
                    if new_scale[0] * new_scale[1] > 300_000:
                        continue
                    if new_scale[0] * new_scale[1] > 100_000:
                        f = 1 - (new_scale[0] * new_scale[1] - 100_000)/200_000
                        bond_img.set_alpha(f*255)
                    if smooth_bonds:
                        bond_img = pg.transform.smoothscale(bond_img, new_scale)
                    else:
                        bond_img = pg.transform.scale(bond_img, new_scale)

                    # insert the bond_img in the right place (after
                    # furthest atom)
                    ai = atom_indices_in_blits.index(a)
                    bi = atom_indices_in_blits.index(b)
                    index = ai if ai < bi else bi
                    atom_indices_in_blits.insert(index, None)

                    angle = pg.math.Vector2([0, 1]).angle_to(rabn[i])
                    im, p = rotate_image(
                        bond_img, mapped_on_sphere1, (bond_img.get_width() / 2, 0), -angle)
                    if draw_bond_rect:
                        pg.draw.rect(
                            self.molecule_surf, (0, 255, 0), p, width=2)

                    if self.show_fig:
                        plt.imshow(pg.PixelArray(im).transpose())
                        plt.show()

                    if not simple_bonds:
                        blits.insert(index + 1, (im, p.topleft))
                    else:
                        pg.draw.line(
                            self.molecule_surf, (255, 255, 255), ra[i], rb[i], width=2)

                if draw_bond_through_atom:
                    pg.draw.line(
                        self.molecule_surf, (255, 0, 255), ra[i], mapped_on_sphere1, width=5)
                    pg.draw.line(
                        self.molecule_surf, (255, 0, 255), rb[i], mapped_on_sphere2, width=5)
                if draw_bond_center:
                    pg.draw.circle(
                        self.molecule_surf, (255, 255, 255), bond_center, 10)

        self.molecule_surf.blits([blit for blit in blits if blit is not None])

    def init_loop(self, state):
        pg.init()
        pg.font.init()
        state['rot'] = np.array(self.init_options['rotation'] or [0, 0])
        state['rotation'] = np.array(self.init_options['zoom'] or [0, 0])
        state['zoom'] = 0
        state['fpss'] = []
        state['fps_num'] = 100
        state['dT'] = 0
        state['show_fps'] = True
        state['time'] = 0
        state['prev_keys'] = pg.key.get_pressed()
        state['normalmode_animation'] = False
        state['normalmode_displacement'] = 0
        state['normalmode_animation_start_time'] = 0

    @timer.Time
    def pre_update(self, state):
        state['start_time'] = perf_counter()
        state['keys'] = pg.key.get_pressed()
        state['events'] = pg.event.get()
        self.molecule_surf.fill(self.background_color)
        self.handle_events(state)

        if state['normalmode_animation']:
            state['normalmode_displacement'] = np.sin(7 * (state['normalmode_animation_start_time'] - state['time']))
            nm = self.molinfo[state['molidx']]['normalmode']
            self.positions[self.mols[state['molidx']]] = list(self.original_positions.values())[
                state['molidx']] + state['normalmode_displacement'] * nm

    @timer.Time
    def update(self, state):
        # if hasattr(state['main_mol'], 'frames'):
        #   i = state.get('mol_frame_i', 0)
        #   i = i % len(state['main_mol'].frames)
        #   state['main_mol'].positions = state['main_mol'].frames[i]
        #   state['mol_frame_i'] = i + 1
        self.positions = {mol: geometry.rotate(coords, x=state['rot'][0], y=state['rot'][1]) for mol, coords in self.positions.items()}
        self.original_positions = {mol: geometry.rotate(coords, x=state['rot'][0], y=state['rot'][1]) for mol, coords in self.original_positions.items()}
        for inf in self.molinfo:
            if 'normalmode' in inf:
                inf['normalmode'] = geometry.rotate(inf['normalmode'], x=state['rot'][0], y=state['rot'][1])
            if 'cub' in inf:
                inf['cub'][0] = geometry.rotate(inf['cub'][0], x=state['rot'][0], y=state['rot'][1])
        with timer.Timer('Screen.update.draw_pixels'):
            if len(self.molinfo) > 0 and 'cub' in self.molinfo[state['molidx']]:
                self.draw_pixels(*self.molinfo[state['molidx']]['cub'])
        self._prepare_molecule_surf(state['molidx'])

        # draw some text
        if not self.no_text:
            try:
                font = pg.font.SysFont(None, 50)
                rxn = self.molinfo[state['molidx']]['reaction']
                name = self.molinfo[state['molidx']]['name']
                text = font.render(f"{rxn} | {name}", True, (255, 255, 255, 255))
                self.molecule_surf.blit(text, (20, 20))
            except BaseException:
                pass
            try:
                font = pg.font.SysFont(None, 24)
                text = font.render(
                    "[CTRL + C] to copy coordinates", True, (255, 255, 255, 255))
                self.molecule_surf.blit(text, (20, self.size[1] - 40))
            except BaseException:
                pass
            try:
                font = pg.font.SysFont(None, 24)
                subss = self.molinfo[state['molidx']]['substituents']
                i = 0
                rct_len = max(len(s) for s in subss.keys())
                for rct, subs in subss.items():
                    for R, s in subs.items():
                        text = font.render(
                            f'{rct.rjust(rct_len)}:{R} = {s}', True, (255, 255, 255, 255))
                        self.molecule_surf.blit(text, (20, 70 + i * 30))
                        i += 1
            except BaseException:
                pass
            try:
                if 'normalmode' in self.molinfo[state['molidx']]:
                    if 'frequency' in self.molinfo[state['molidx']]:
                        freq = f" ({self.molinfo[state['molidx']]['frequency']:.1f} cm^-1)"
                    else:
                        freq = ''
                    text = font.render(
                        f'Press [SPACE] to visualize lowest mode {freq}', True, (255, 255, 255, 255))
                    self.molecule_surf.blit(
                        text, (self.size[0] - 500, self.size[1] - 40))
            except BaseException:
                pass
            try:
                if len(self.molinfo) > 1:
                    font = pg.font.SysFont(None, 50)
                    text = font.render(
                        f'( {state["molidx"] + 1} / {len(self.molinfo)} )', True, (255, 255, 255, 255))
                    self.molecule_surf.blit(text, (self.size[0] - 130, 20))
            except BaseException:
                pass

        [func(state) for func in self.update_funcs]


    @timer.Time
    def post_update(self, state):
        state['rotation'] = state['rotation'] + state['rot']
        state['rot'] = state['rot'] * 0.8
        state['zoom'] = 0
        state['fpss'].append(1 / (perf_counter() - state['start_time']))
        state['dT'] = perf_counter() - state['start_time']
        state['time'] += state['dT']

        if len(state['fpss']) > state['fps_num']:
            state['fpss'].pop(0)
        # if state['show_fps']: print(f"fps (avg. over {state['fps_num']}) = {sum(state['fpss'])/state['fps_num'] :.0f}")

        # self.camera_orientation = (0, state['time']/2, 0)
        # self.camera_position = [sin(state['time']*10), cos(state['time']*10)]
        state['prev_keys'] = state['keys']


    @timer.Time
    def handle_events(self, state):
        def start_mode_animation():
            state['normalmode_animation'] = True
            state['normalmode_animation_start_time'] = state['time']

        def stop_mode_animation():
            state['normalmode_animation'] = False
            self.positions[self.mols[state['molidx']]] = self.original_positions[self.mols[state['molidx']]]
            state['normalmode_displacement'] = 0

        def copy_atoms():
            if len(self.mols) == 0:
                return
            p = self.mols[state['molidx']]
            pyperclip.copy(molecule.get_xyz(p))
            print('Copied atoms!')

        def paste_atoms():
            data = pyperclip.paste().replace('\\n', '\n')
            lines = [line for line in data.splitlines() if len(line.split()) >= 4]

            mol = plams.Molecule()
            for line in lines:
                el, x, y, z = line.split()[:4]
                mol.add_atom(plams.Atom(symbol=el, coords=(x, y, z)))
            mol.guess_bonds()
            self.add_mol(mol)
            if len(self.mols) > 1:
                state['molidx'] += 1

        def get_dropped_file(path):
            with open(path) as xyz:
                lines = [line for line in xyz.readlines() if len(line.split()) >= 4]

            mol = plams.Molecule()
            for line in lines:
                el, x, y, z = line.split()[:4]
                mol.add_atom(plams.Atom(symbol=el, coords=(x, y, z)))
            mol.guess_bonds()
            self.add_mol(mol)
            if len(self.mols) > 1:
                state['molidx'] += 1

        def delete_atoms():
            if len(self.mols) == 0:
                return
            self.delete_mol(state['molidx'])
            state['molidx'] = max(0, state['molidx']-1)

        if state['keys'][pg.K_ESCAPE]:
            state['run'] = False
        # if state['keys'][pg.K_SPACE]:
        #   self.show_fig = True
        else:
            self.show_fig = False

        for e in state['events']:
            if e.type == pg.QUIT:
                state['run'] = False

            elif e.type == pg.MOUSEBUTTONDOWN:
                if e.button == 4:
                    state['zoom'] = -state['dT'] * self.camera_z * 10
                elif e.button == 5:
                    state['zoom'] = state['dT'] * self.camera_z * 10

            elif e.type == pg.DROPFILE:
                get_dropped_file(e.file)


        # print(state['zoom'], state['dT'])
        self.camera_z += state['zoom']

        move = pg.mouse.get_rel()
        # if state['keys'][pg.K_LCTRL] or state['keys'][pg.K_RCTRL]:
        if pg.mouse.get_pressed()[2]:
            self.camera_position[0] += move[0] / 100
            self.camera_position[1] += move[1] / 100

        if pg.mouse.get_pressed()[0]:
            state['rot'] = np.array([move[1] / 150, move[0] / 150])

        if state['keys'][pg.K_LEFT] and not state['prev_keys'][pg.K_LEFT]:
            state['molidx'] -= 1
            state['molidx'] = state['molidx'] % len(self.mols)
            stop_mode_animation()

        if state['keys'][pg.K_RIGHT] and not state['prev_keys'][pg.K_RIGHT]:
            state['molidx'] += 1
            state['molidx'] = state['molidx'] % len(self.mols)
            stop_mode_animation()

        if state['keys'][pg.K_SPACE] and not state['prev_keys'][pg.K_SPACE] and 'normalmode' in self.molinfo[state['molidx']]:
            if state['normalmode_animation']:
                stop_mode_animation()
            else:
                start_mode_animation()

        if (state['keys'][pg.K_RCTRL] or state['keys'][pg.K_LCTRL]) and state['keys'][pg.K_c] and not state['prev_keys'][pg.K_c]:
            copy_atoms()

        if (state['keys'][pg.K_RCTRL] or state['keys'][pg.K_LCTRL]) and state['keys'][pg.K_v] and not state['prev_keys'][pg.K_v]:
            paste_atoms()

        if (state['keys'][pg.K_RCTRL] or state['keys'][pg.K_LCTRL]) and state['keys'][pg.K_x] and not state['prev_keys'][pg.K_x]:
            delete_atoms()

        if (state['keys'][pg.K_RCTRL] or state['keys'][pg.K_LCTRL]) and state['keys'][pg.K_h] and not state['prev_keys'][pg.K_h]:
            self.hide_hydrogens = not self.hide_hydrogens

            # state['normalmode_animation'] = not state['normalmode_animation']
            # if state['normalmode_animation']:
            #   state['normalmode_animation_start_time'] = state['time']
            # else:
            #   state['normalmode_displacement'] = 0
            #   self.mols[state['molidx']].positions = self.mols[state['molidx']].original_pos

    def draw_axes(self, surf):
        ...

    def draw_pixels(self, poss, colors):
        poss = poss + np.random.randn(*poss.shape)/50
        r = (poss - (*self.camera_position, self.camera_z))
        dist_to_cam = np.sqrt(np.sum(r**2, axis=1))
        dist_idx = np.argsort(-dist_to_cam)
        poss_ = self.project(poss)
        # print(dist_idx.shape, poss_.shape, colors.shape)

        for pos, c in zip(poss_[dist_idx], colors[dist_idx]):
            # self.molecule_surf.set_at(pos, c)
            pg.draw.circle(self.molecule_surf, c, pos, 4)
