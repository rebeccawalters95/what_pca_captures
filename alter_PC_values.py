from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
import numpy as np
import tqdm
import math
import os


# calc_ij and print_distance_weights_to_files functions are taken from PathReducer
# (https://github.com/share1992/PathReducer/blob/c034544e691277be5221b341a06ce3a8de5cca9b/dimensionality_reduction_functions.py#L747)
def calc_ij(k, n):
    """
    Calculate indexes i and j of a square symmetric matrix given upper triangle vector index k and matrix side length n

    :param k: vector index
    :param n: side length of resultant matrix M
    :return: i, j as ints
    """
    i = n - 2 - math.floor((np.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2) - 0.5)
    j = k + i + 1 - (n * (n - 1) / 2) + ((n - i) * ((n - i) - 1) / 2)
    return int(i), int(j)


def print_distance_weights_to_files(directory, n_dim, system_name, pca_components, num_atoms,
                                    selected_atom_indexes=None):
    """
    Couple each distance (denoted by two atom indices) and corresponding coefficient for each principal component, and
    sort by descending order

    :param directory: Path to directory to save files to
    :param n_dim: Number of PCs
    :param system_name: Name of the biomolecular system of interest
    :param pca_components: Scikit-learn object containing pca components, e.g. pca_components=pca.components_
    :param num_atoms: Number of atoms involved in PCA
    :param selected_atom_indexes: Indices of a select few atoms, optional, recommended None
    :return: sorted_d, a pd data frame containing all the atom indices and corresponding coefficients in
             descending order
    """
    for n in range(n_dim):
        if selected_atom_indexes:
            distance_vector_indexes = list(pd.DataFrame(list(selected_atom_indexes.values()))[1])
        else:
            distance_vector_indexes = range(len(pca_components[n]))

        d = []
        for k, l in zip(distance_vector_indexes, range(len(pca_components[n]))):
            i, j = calc_ij(k, num_atoms)
            coeff = pca_components[n][l]
            d.append({'Coefficient of Distance': coeff, 'atom 1': i, 'atom 2': j})

        d_df = pd.DataFrame(d)

        sorted_d = d_df.reindex(d_df['Coefficient of Distance'].abs().sort_values(ascending=False).index)
        sorted_d.to_csv(os.path.join(directory, system_name + '_PC%s_components.txt' % (n + 1)), sep='\t', index=None)

    return sorted_d


def pca_fit_transform(topology, trajectory, atom_selection, number_of_components=2):
    """
    Takes an MD trajectory, calculates the distance matrix between all the atoms involved in the atom selection, and
    performs PCA (fit and transform) on it.

    :param str, topology: Path to the topology of the MD simulation
    :param str, trajectory: Path to the trajectory file of the MD simulation
    :param str, atom_selection: Atoms to include in the distance matrix calculation based on
                                MDAnalysis selection language
    :param int, number_of_components: Number of principal components computed, DEFAULT: 2
    :return:
        pca: sklearn IncrementalPCA object
        transformed: numpy.array, transformed data
        no_of_atoms: int, number of atoms involved in PCA
    """
    # i) Preparing for PCA
    pca = IncrementalPCA(n_components=number_of_components)
    # selecting which atoms to perform PCA on
    universe = mda.Universe(topology, trajectory)
    interest = universe.select_atoms(atom_selection).atoms
    print('the number of atoms chosen is:', interest.n_atoms)
    print('the number of frames in the trajectory is:', universe.trajectory.n_frames)

    # specifying inputs to be used throughout the script...
    no_of_atoms = len(interest)  # number of atoms
    no_of_frames_per_batch = number_of_components  # number of batches must be at least the number of principal components
    frames = universe.trajectory.n_frames  # number of frames in trajectory
    distance_matrix = np.zeros(
        (frames, int(no_of_atoms * (no_of_atoms - 1) / 2)))  # creating empty distance matrix to fill

    # ii) Fitting PCA

    for frame_index, _ in tqdm.tqdm(
            enumerate(universe.trajectory),
            total=frames,
            desc=str('Fitting progress'),
            leave=False
    ):
        # calculates the distance array between all atoms of interest in trajectory
        mda.lib.distances.self_distance_array(
            interest.positions,
            box=interest.dimensions,
            # result is stored in distance matrix where i specifies which row in the 'batch' to place the array
            # and : specifies to fill the full column.
            result=distance_matrix[frame_index % no_of_frames_per_batch, :],
            backend='OpenMP',
        )
        # if all of the rows of the distance matrix in the batch are filled, then perform partial_fit, then start
        # again with empty distance matrix
        if (frame_index - 2) % no_of_frames_per_batch == 0:
            pca.partial_fit(distance_matrix)

    print(pca.explained_variance_ratio_)

    # iii) Transforming PCA

    transformed = np.empty(
        (frames, pca.n_components),
        dtype=np.float32,
    )  # this gets the full result, for each frame get the number_of_components

    no_of_atoms = len(interest)
    distance_matrix_transform = np.zeros(
        (1, int(no_of_atoms * (no_of_atoms - 1) / 2)))  # we will re-use this for the flattened distance matrices (
    # per frame)

    frame_progress = tqdm.tqdm(
        enumerate(universe.trajectory),
        total=universe.trajectory.n_frames,
        desc=str('Transforming progress'),
        leave=False,
    )
    for frame_index, _ in frame_progress:
        mda.lib.distances.self_distance_array(
            interest.positions,
            box=interest.dimensions,
            result=distance_matrix_transform[0, :],
            backend='OpenMP',
        )
        # add the transformed data for each frame
        transformed[frame_index, :] = pca.transform(distance_matrix_transform)

    return pca, transformed, no_of_atoms


def plot_pca(transformed_data):
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle('PCA of VR Unbinding Trajectory (input for WESTPA)')

    # 2D representation
    ax2d = fig.add_subplot(1, 2, 1)
    ax2d.set(xlabel='PC1', ylabel='PC2', Title='2D')
    ax2d.scatter(transformed_data[:, 0], transformed_data[:, 1], s=25, alpha=0.60, c=transformed_data[:, 0],
                 cmap='viridis')

    # 3D representation
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax3d.set(xlabel='PC1', ylabel='PC2', zlabel='PC3', Title='3D')
    ax3d.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], s=25, alpha=0.60,
                 c=transformed_data[:, 0],
                 cmap='viridis')

    return plt.show()


def alter_eigenvalues_to_check_pc(output_directory, transformed_data, chosen_pc, increment,
                                  range_of_values, starting_point=0):
    """
    Takes data that has been transformed by PCA and alters the eigenvalues (principal components) by a given increment
    over a range of values. The idea is that, you can take the set of eigenvalues for the first data point, e.g. take
    the eigenvalues of PC 1, 2, and 3 for the first frame in an MD simulation, and move along a principal coordinate,
    e.g. increase/decrease values along principal coordinate 1.

    This allows you to explore the values of a specific principal coordinate, while the values of the other principal
    coordinates remain the same. This means we can see what motions are captured in a specific principal coordinate.

    :param str, output_directory: Path to the directory where new saved transformed data will be saved
    :param numpy.array, transformed_data: Transformed data to alter
    :param int, chosen_pc: Which eigenvalue should be altered
    :param int, increment: Number by which to increase or decrease the eigenvalue by, e.g., -40, 25, -150
    :param int, range_of_values: Range over which the increment should be applied, e.g., range_of_values = 5 and
                            increment = 10, would increase the eigenvalue by 10, 20, 30, 40, 50
    :param int, starting_point: Which point in PC space would you like to start altering the eigenvalues from, e.g.,
                           0 would mean you would access the first point in PC space, returning the first values for
                           each eigenvalue, and -1 would return the eigenvalues for the final point in PC space
    :return:
        list_of_filenames: list, names of the new files in following format:
                           transformed_pc{chosen_pc}_{copy_transformed[chosen_pc - 1]:.3f}
        list_of_filepaths: list, list of paths to files, inc new filename, e.g.,
                           ['path/to/transformed/data/directory/transformed_pc2_1000.npy',
                           'path/to/transformed/data/directory/transformed_pc2_2000.npy'
                           'path/to/transformed/data/directory/transformed_pc2_3000.npy']
    """
    list_of_filenames = []
    list_of_filepaths = []

    copy_transformed = transformed_data[starting_point, :]
    np.save(os.path.join(output_directory, f'transformed_pc{chosen_pc}_{copy_transformed[chosen_pc - 1]:.3f}.npy'),
            copy_transformed)
    list_of_filenames.append(f'transformed_pc{chosen_pc}_{copy_transformed[chosen_pc - 1]:.3f}')
    list_of_filepaths.append(str(os.path.join(output_directory,
                                              f'transformed_pc{chosen_pc}_{copy_transformed[chosen_pc - 1]:.3f}.npy')))

    for value in range(range_of_values):
        copy_transformed[chosen_pc - 1] += increment
        np.save(
            os.path.join(output_directory, f'transformed_pc{chosen_pc}_{copy_transformed[chosen_pc - 1]:.3f}.npy'),
            copy_transformed)
        print(copy_transformed[chosen_pc - 1])
        list_of_filenames.append(f'transformed_pc{chosen_pc}_{copy_transformed[chosen_pc - 1]:.3f}')
        list_of_filepaths.append(str(os.path.join(output_directory,
                                                  f'transformed_pc{chosen_pc}_{copy_transformed[chosen_pc - 1]:.3f}.npy')))
    return list_of_filenames, list_of_filepaths


def back_transform_pca(topology, trajectory, atom_selection, output_directory, pca_object,
                       transformed_data, name_dir):
    """
    Takes the transformed data from the altered eigenvalues, and backtransforms the data to get the
    corresponding atom indices and distance values.

    :param str, topology: Path to the topology of the MD simulation
    :param str, trajectory: Path to the trajectory file of the MD simulation
    :param str, atom_selection: Atoms to include in the distance matrix calculation based on
                                MDAnalysis selection language
    :param str, output_directory: Pathway to output directory where a new directory will be made for each
                                  altered eigenvalue
    :param Sklearn IncrementalPCA object, pca_object: SciKitLearn IncrementalPCA object of original fit/transform of
                                                       data, this is outputted by the pca_fit_transform function
    :param str, transformed_data: Pathway to numpy array containing the transformed data (e.g. altered eigenvalues)
    :param str, name_dir: Name of directory to save results to (created in this function)
    :return:
        np.asarray(indices_for_bonds): Array of atom indices for each distance, e.g., [[0, 1], [0, 2], ..., [19, 20]]
        np.asarray(distances_for_bonds): Array of distances corresponding to each pair of atom indices, e.g.,
                                         [[3.14], [2.56], ..., [4.12]]. N.B. Distances are in Ångstroms
    """

    # selecting atoms involved in pca
    universe = mda.Universe(topology, trajectory)
    interest = universe.select_atoms(atom_selection).atoms
    print('the number of atoms chosen is:', interest.n_atoms)
    print('the number of frames in the trajectory is:', universe.trajectory.n_frames)

    number_of_atoms = len(interest)

    # using pca object and transformed numpy array of eigenvalues
    # for specific point in PC space
    np_transformed_data = np.load(transformed_data)
    print(np_transformed_data)
    # get flattened distance matrix
    distances_for_bonds = pca_object.inverse_transform(np_transformed_data)

    # get atom indices involved in distance matrix
    N = number_of_atoms
    upper = np.triu_indices(N, k=1)
    atom_pairs = np.array(upper).T

    # creating array of atom indices used in pca i.e. no hydrogen
    arr_noH = np.asarray(interest.indices)

    # updating atom pairs to have correct indices i.e. no hydrogen
    indices_for_bonds = arr_noH[atom_pairs]
    print('CHECK: Number of indices for bonds is', len(indices_for_bonds),
          ' and the number of distances is', len(distances_for_bonds))
    print(indices_for_bonds[0], ':', distances_for_bonds[0])

    dir_name = os.path.join(output_directory, name_dir)
    try:
        # create target Directory
        os.mkdir(dir_name)
        print("Directory ", dir_name, " Created ")
        # save array for use in amber simulation to add fake bonds
        np.save(os.path.join(dir_name, 'indices_for_bonds.npy'), indices_for_bonds)
        np.save(os.path.join(dir_name, 'distances_for_bonds.npy'), distances_for_bonds)
    except FileExistsError:
        print("Directory ", dir_name, " already exists")

    return np.asarray(indices_for_bonds), np.asarray(distances_for_bonds)


def write_fake_bonds_to_file(output_directory, indices_for_bonds, distances_for_bonds, verbose=True):
    """
    Takes atom indices and distances from the altered eigenvalues and saves them to a file as 'fake bonds'
    for AMBER (MD) program to use as restraints during energy minimisation

    :param output_directory: str, Path to save output file
    :param indices_for_bonds: np.array, atom indices for each distance, e.g., [[0, 1], [0, 2], ..., [19, 20]]
    :param distances_for_bonds: np.array, distances corresponding to each pair of atom indices, e.g.,
                                [[3.14], [2.56], ..., [4.12]]. N.B. Distances are in Ångstroms
    :param verbose: bool, prints every distance and weight to stdout if True
    :return: Saves 'fakebonds.120' file to chosen output directory
    """
    if verbose:
        print('index 0:', indices_for_bonds[0][0])
        print('index 1:', indices_for_bonds[0][1])
        print('distance:', distances_for_bonds[0])
    else:
        print('Verbose option not chosen, will not write out each index and distance value, carrying on...')

    with open(os.path.join(output_directory, 'fakebonds.120'), 'w') as f:
        for index_for_bond in range(len(indices_for_bonds)):
            index0 = int(indices_for_bonds[index_for_bond][0]) + 1  # +1 to offset zero indexing
            index1 = int(indices_for_bonds[index_for_bond][1]) + 1
            dist = float(distances_for_bonds[index_for_bond])
            dist = round(dist, 1)
            f.write(
                f'\nHarmonic restraints for calcium ion \n &rst \n  iat={index0},{index1},\n  r1={dist - 0.5}, '
                f'r2={dist}, r3={dist}, r4={dist + 0.5}\n  rk2=250000.0, rk3=250000.0,\n /')

    f.close()


# First, must define some variables

mutant = 'HY'  # which mutant of neuraminidase are we looking at?

# path to molecular dynamics trajectory and topology
traj = 'PATH/TO/TRAJECTORY'
top = 'PATH/TO/TOPOLOGY'

num_pcs = 10  # specifying number of components
sel = 'ATOM SELECTION USING MDANALYSIS SELECTION LANGUAGE'  # specifying atom selection to perform PCA on
name = 'PC'  # prefix of file name to save PCA coefficients to

# parameters to change for altering eigenvalues (part iii onwards)
pc_chosen = 3  # which principal component to alter
inc = 100  # increment to increase/decrease eigenvalue
range_vals = 10  # range of values of which the increment will be applied
output_dir = 'PATH/TO/OUTPUT/DIRECTORY'

# -------------------------------------- #
# ------ i) Fit/transform of data ------ #
# -------------------------------------- #

pca, transformed, n_atoms = pca_fit_transform(topology=top, trajectory=traj, atom_selection=sel,
                                              number_of_components=num_pcs)
print('i) The data (distance matrix) has been successfully fit and transformed into PC space')

# -------------------------------------- #
# -------- ii) Plot transformed -------- #
# ---------- data in PC space ---------- #
# -------------------------------------- #

plot_pca(transformed)
print(transformed)
print('ii) Transformed data has been successfully plotted in 2D and 3D space')

# -------------------------------------- #
# ----- iii) Altering eigenvalues ------ #
# -------------------------------------- #

filenames, path_to_altered_nparrays = alter_eigenvalues_to_check_pc(output_dir, transformed, chosen_pc=pc_chosen,
                                                                    increment=inc, range_of_values=range_vals)

print(filenames)
print(path_to_altered_nparrays)
print(f'iii) Done altering eigenvalues along PC {pc_chosen}, by {inc} over a range of {range_vals}')

# -------------------------------------- #
# ------ iv) Back-transform data ------- #
# ------- to get distance matrix ------- #
# -------------------------------------- #

for index, path in enumerate(path_to_altered_nparrays):
    indices, distances = back_transform_pca(topology=top, trajectory=traj, atom_selection=sel,
                                            output_directory=output_dir, pca_object=pca,
                                            transformed_data=path, name_dir=filenames[index])
    print('indices:', indices)
    print('distances', distances)
    write_fake_bonds_to_file(output_directory=os.path.join(output_dir, filenames[index]),
                             indices_for_bonds=indices, distances_for_bonds=distances)

print('iv) Done getting distance matrix back out for each structure with altered eigenvalues, and have written '
      'new \'fake\' bonds to file. These files can be used as restraints during energy minimisation to get structures '
      'of how each PC evolves over time')

# -------------------------------------- #
# -------- v) Print indices and -------- #
# -------- coefficients to file -------- #
# -------------------------------------- #

print_distance_weights_to_files(directory=output_dir, n_dim=num_pcs, system_name=name,
                                pca_components=pca.components_, num_atoms=n_atoms)
print('v) Finally, have printed distance and weights to a file, for future reference')