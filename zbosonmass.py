#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phys20161 Second Assignment: Z boson
Created on Fri Nov 26 21:43:56 2021
This program gives a best estimate on the mass and width of the Z-boson based
on a minimum chi-square fit on the data provided. Specifically, it plots out a
fit of the breit-wigner distribution on the data as well as a contour plot of
the fitted chi square values. Lifetime of the Z-boson is also calculated.
Datasets should be put in a file-folder first before running the script.
@author: chanheichunthomas ID:10727090
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import scipy.constants as pc



ELECTRON_WIDTH = 0.08391 # in GeV
# input initial guesses for optimization fit
# note that mass is the first index (in GeV/c square) and width is the second
# index (in GeV)
GUESS = [70, 20]
CONVERSION_FACTOR = 0.3894 * (10**6)
NO_OF_PARAMETERS = 2
HBAR = pc.hbar
GEV_UNIT = pc.eV * 1e9

# Insert your folder name here (e.g, 'boson')
FOLDER_NAME = 'boson'

# adjust the scale below to control the contour plot
MESH_SCALE = 0.05

# choose SAVEFIG as True if you wish to save your graph as png.file
# type in the name of the figure and file of your filtered data
SAVEFIG = True
FIGNAME = 'Z_boson_optimized_fit.png'
COMBINED_FILENAME = 'Z_boson_combined.csv'

def save_file(data, name):
    '''
    Saves the filtered dataset as another csv file for convenience of use
    later on

    Parameters
    ----------
    data : np.ndarray
    name : string

    Returns
    -------
    Int: 0 (To indicate the file is saved)

    '''
    np.savetxt(name, data, delimiter = ',', header =
               '% energy (GeV), cross section (nb), uncertainties (nb)')
    return 0

def get_folder_data(folder_name):
    '''
    Grabs all the files in the folder and puts them in a list

    Parameters
    ----------
    folder_name : string
        name of your folder holding the datasets

    Returns
    -------
    data_list : list

    '''
    try:
        path = os.getcwd() + '/' + folder_name
        data_list = os.listdir(path)

    except FileNotFoundError:
        print('Your folder could not be found in the current directory.')
        data_list = 0

    return data_list

def extreme_outlier_removal(raw_data):
    '''
    Removes the extreme outliers by assuming the whole data takes on a linear
    line and detect points that exceed three times the standard deviation away
    from the mean

    Parameters
    ----------
    raw_data : ndarray

    Returns
    -------
    data: ndarray

    '''
    line_mean = np.mean(raw_data[:, 1])
    standard_deviation = np.std(raw_data[:, 1])
    difference = np.abs(raw_data[:, 1] - line_mean)
    condition = difference < 3 * standard_deviation
    filtered_data = raw_data[np.where(condition)]
    return filtered_data

def file_read(raw_data):
    '''
    Reads in a raw dataset and removes any nan values from the dataset.
    Any extreme outliers and invalid negative uncertainties were also removed
    from the process.
    Dataset should be in order of energy(Gev), cross section(nb),
    uncertainty(nb)

    Parameters
    ----------
    RAW_DATA : csv file

    Returns
    -------
    data : np.ndarray with 3 columns

    '''
    data = np.genfromtxt(raw_data, delimiter = ',', skip_header = 1)
    valid_values = np.where(~np.isnan(data).any(axis = 1))
    data = data[valid_values]
    valid_errors = np.where(data[:, 2] > 0)
    data = data[valid_errors]
    valid_cross_sections = np.where(data[:, 1] > 0)
    data = data[valid_cross_sections]
    data = extreme_outlier_removal(data)
    return data

def combine_file(filelist):
    '''
    combines files with the same data types and creates a new sorted combined
    file. The new combined file is automatically saved

    Returns
    -------
    sorted_combined_data: ndarray

    '''
    try:
        combined_data = np.zeros((0,3))
        for file in filelist:
            if file.endswith('.csv'):
                read_data = file_read(file)
                combined_data = np.vstack((combined_data, read_data))
        sorted_index = np.argsort(combined_data[:, 0])
        sorted_combined_data = combined_data[sorted_index]
    except TypeError:
        print('cannot stack the datafiles as some files cannot be found')
        sorted_combined_data = None

    return sorted_combined_data

def outlier_removal(data, model_result):
    '''
    Further removes outliers in our data that exceeds three standard deviation
    of the prediction made by the model

    Parameters
    ----------
    data : np.ndarray
    model_result: float tuple (mass, width)

    Returns
    -------
    filtered_data: np.ndarray

    '''
    threshold = 3 * data[:, 2]
    difference = np.abs(data[:, 1] - cross_section_model(model_result[0],
                            data[:, 0], model_result[1]))
    non_outliers = np.where(difference < threshold)
    filtered_data = data[non_outliers]
    return filtered_data

# functions for calculations
def cross_section_model(mass, energy, boson_width):
    '''
    calculates cross section from mass, energy, width based on equation
    of Breit-Wigner expression. Cross section is return in nb.

    Parameters
    ----------
    mass : float
    energy : float
    boson_width : float

    Returns
    -------
    predicted_cross_section : float ndarray
    '''
    energy_square = energy ** 2
    mass_square = mass ** 2
    boson_width_square = boson_width ** 2
    constant = 12 * np.pi / (mass_square)
    denominator_portion_1 = (energy_square - mass_square) ** 2
    denominator_portion_2 = (mass_square) * (boson_width_square)
    denominator = denominator_portion_1 + denominator_portion_2
    numerator =  energy_square * (ELECTRON_WIDTH ** 2)
    predicted_cross_section = constant * numerator / denominator
    predicted_cross_section = predicted_cross_section * CONVERSION_FACTOR
    return predicted_cross_section

def chi_square_function(parameters, xdata, observation, uncertainties):
    '''
    calculates the chi square of a fit of the function

    Parameters
    ----------
    parameters : float tuple (mass, width)
    xdata : np.array
    observation: np.array
    uncertainties : np.array

    Returns
    -------
    chi_square: float

    '''
    mass = parameters[0]
    boson_width = parameters[1]
    prediction = cross_section_model(mass, xdata, boson_width)
    chi_square = np.sum((observation - prediction)**2 / uncertainties**2)
    return chi_square

def chi_square_fit(data):
    '''
    Generates a fit on our data by altering mass and width of Z boson
    and minimizing the chi-square using fmin.

    Parameters
    ----------
    data: np.ndarray

    Returns
    -------
    fit_results: float tuple
                first index: tuple with fitted mass as first
                              index, fitted width as second index)
                second index: chi square value of your fit
                third index: number of iterations made
                fourth index: number of function calls made
    '''

    fit_results = fmin(chi_square_function, GUESS,
                       args = (data[:, 0], data[:, 1], data[:, 2]),
                       full_output=1, disp = 0)
    return fit_results


def reduced_chi_square(chi_square, data):
    '''
    Calculates the reduced chi square of the fit. In this case, the number of
    parameters is 2 for this fit.

    Parameters
    ----------
    chi_square : float
    data : ndarray

    Returns
    -------
    reduced_chi : float
    '''

    dof = len(data) - NO_OF_PARAMETERS
    reduced_chi = chi_square / dof
    return reduced_chi

def find_fit_uncertainty(first_contour):
    '''
    calculates the uncertainty of mass and width based on data from the first
    contour

    Returns
    -------
    mass_error: float
    width_error: float

    '''
    first_contour_x_values = first_contour.vertices[:,0]
    first_contour_y_values = first_contour.vertices[:,1]
    max_x_value = np.max(first_contour_x_values)
    min_x_value = np.min(first_contour_x_values)
    max_y_value = np.max(first_contour_y_values)
    min_y_value = np.min(first_contour_y_values)
    mass_error = np.abs(max_x_value - min_x_value)/2
    width_error = np.abs(max_y_value - min_y_value)/2
    return mass_error, width_error

def find_lifetime_and_error(width, width_error):
    '''
    calculates the lifetime of the Z boson from its width in seconds

    Parameters
    ----------
    width : float

    Returns
    -------
    lifetime : float

    '''
    width = width * GEV_UNIT
    lifetime = HBAR / width
    lifetime_error = width_error * GEV_UNIT / width * lifetime
    return lifetime, lifetime_error

# Functions for creating mesh arrays and plotting graphs
def mesh_arrays(mass, width):
    '''
    generates mesh arrays of 50x50 of mass and width values of Z boson,
    for plotting a contour plot

    Parameters
    ----------
    mass: float
    width: float

    Returns
    -------
    mass_mesh_array: np.ndarray
    width_mesh_array: np.ndarray

    '''
    mass_array = np.linspace(mass - MESH_SCALE, mass + MESH_SCALE)
    width_array = np.linspace(width - MESH_SCALE, width + MESH_SCALE)

    mass_mesh_array = np.empty((0, len(mass_array)))
    width_mesh_array = np.empty((0, len(width_array)))

    for _ in enumerate(width_array):
        mass_mesh_array = np.vstack((mass_mesh_array, mass_array))

    for _ in enumerate(mass_array):
        width_mesh_array = np.vstack((width_mesh_array, width_array))

    width_mesh_array = np.transpose(width_mesh_array)

    return mass_mesh_array, width_mesh_array



def create_chi_square_mesh(mass, width, data):
    '''
    Generates the mesh of chi square value for contour plot

    Parameters
    ----------
    mass : float
    width : float
    data : np.ndarray

    Returns
    -------
    chi_square_mesh : np.ndarray

    '''
    mass_mesh_array, width_mesh_array = mesh_arrays(mass, width)
    chi_square_mesh = np.zeros((50,50))
    for index, mass_value in enumerate(mass_mesh_array[0]):
        for index_2, width_value in enumerate(width_mesh_array[:,0]):
            chi_square = chi_square_function((mass_value, width_value),
                    data[:,0], data[:,1], data[:,2])
            chi_square_mesh[index,index_2] = chi_square
    return chi_square_mesh




def plot_result(fit_results, data, chi_square):
    '''
    plots out the result of the best fit on our data, as well as a contour
    plot of the minimised chi-square. Results of the fit are also annotated
    along with the graph. Graph of the result can also be saved based on the
    user's choice in the beginning of the code.

    Parameters
    ----------
    fit_results : float tuple
                    (mass, width)
    data : np.ndarray
    chi-square: float

    Returns
    -------
    None.

    '''
    fig = plt.figure(figsize = (10,8))
    grid = fig.add_gridspec(2, 2)
    main_ax = fig.add_subplot(grid[0, 0:2])
    contour_ax = fig.add_subplot(grid[1, 0])
    main_ax.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='o',
    label = 'Data')
    main_ax.plot(data[:, 0], cross_section_model(fit_results[0], data[:, 0],
                                            fit_results[1]),
               color = 'red', label = 'Fitted plot')
    main_ax.set_title('Distribution of cross section over mass of boson $m_Z$')
    main_ax.set_xlabel('Mass of boson $m_Z$')
    main_ax.set_ylabel('Cross section (nb)')
    main_ax.grid(linestyle = '--')
    main_ax.axvline(x = fit_results[0], linestyle = '--', label =
               '$m_Z = {0:.4g}$'.format(fit_results[0]), color = 'purple')
    main_ax.legend()

    # plotting the contour plot


    first_contour = plot_contour(fit_results, data, chi_square, contour_ax)

    mass_error, width_error = find_fit_uncertainty(first_contour)

    plot_contour_uncertainties(contour_ax, fit_results[0], fit_results[1],
                               mass_error, width_error)

    reduced_chi = reduced_chi_square(chi_square, data)

    boson_lifetime, lifetime_error = find_lifetime_and_error(
                                            fit_results[1], width_error)

    #annotating the results on the graph

    main_ax.annotate((r'$\chi^2$ = {0:.3f}'.format(chi_square)),
                     (0.6, 0), (0, -50), xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='12')

    main_ax.annotate((r'reduced $\chi^2 = {0:.3f}$'.format(reduced_chi)),
                     (0.6, 0), (0, -70), xycoords='axes fraction', va='top',
                            textcoords='offset points', fontsize='12')

    main_ax.annotate((r'$m_z$ = {0:.4g} ± ${1:.2f}$ GeV/$c^2$'.format(
                     fit_results[0], mass_error)), (0.6, 0), (0, -110),
                     xycoords='axes fraction', va='top',
                     textcoords='offset points', fontsize='12')

    main_ax.annotate((r'$\Gamma_z$ = {0:.4g} ± ${1:.3f}$ GeV'.format(
                     fit_results[1], width_error)), (0.6, 0), (0, -130),
                     xycoords='axes fraction', va='top',
                     textcoords='offset points', fontsize='12')

    main_ax.annotate((r'Lifetime $\tau_z$ = {0:.3g} ± ${1:.1g}$ s'.format(
                     boson_lifetime, lifetime_error)), (0.6, 0), (0, -150),
                     xycoords='axes fraction', va='top',
                     textcoords='offset points', fontsize='12')

    if SAVEFIG:
        plt.savefig(FIGNAME, dpi = 500)
    plt.show()


def plot_contour(fit_results, data, chi_square, contour_ax):
    '''
    Takes in an empty axes and plots out a contour plot using your fit results
    (mass and width), data and chi square value. Function also returns the x,y
    data from the first contour

    Parameters
    ----------
    fit_results : float tuple (mass as first index, width as second index)
    data : ndarray
    chi_square : float
    contour_ax : axes

    Returns
    -------
    contour_data : path

    '''

    coordinates = mesh_arrays(fit_results[0], fit_results[1])
    mapping_value = create_chi_square_mesh(fit_results[0], fit_results[1],
                                           data)
    levels = [chi_square + 1, chi_square + 2, chi_square + 3]
    contour_plot = contour_ax.contour(coordinates[0], coordinates[1],
                                      mapping_value, levels = levels)
    contour_ax.clabel(contour_plot, fontsize = 16)
    contour_ax.scatter(fit_results[0], fit_results[1],
                       label = r'$\chi^2_{fit}$')
    contour_ax.set_title(r'2D Contours of $\chi^2_{fit}$')
    contour_ax.set_xlabel('Mass $M_Z$')
    contour_ax.set_ylabel(r'Width $\Gamma_Z$')
    labels = [r'$\chi^2_{fit}+1$', r'$\chi^2_{fit}+2$',
              r'$\chi^2_{fit}+3$']

    for index, label in enumerate(labels):
        contour_plot.collections[index].set_label(label)
    contour_ax.legend(loc = 'upper left')

    contour_data = contour_plot.collections[0].get_paths()[0]
    return contour_data

def plot_contour_uncertainties(contour_axis, mass, width,
                               mass_error, width_error):
    '''
    Plots out the uncertainty line spanning from the fitted chi square to the
    first contour line in the contour plot

    Parameters
    ----------
    contour_axis : axes environment
    mass : float
    width : float
    mass_error : float
    width_error : float

    Returns
    -------
    None.

    '''
    contour_axis.vlines(x = mass, ymin = width - width_error,
                ymax = width + width_error, linestyle = '--', color = 'red')
    contour_axis.hlines(y = width, xmin = mass - mass_error,
                xmax = mass + mass_error, linestyle = '--', color = 'red')


def main():
    '''
    This funtion is where the main operation of the file is. It first filters
    and combines the file, then calculates the minimised chi-square and best
    fit of the parameters, and finally plots out the graph of the fit. The new
    filtered data is also automatically saved.

    Returns
    -------
    int

    '''
    try:
        data_list = get_folder_data(FOLDER_NAME)
        data = combine_file(data_list)
        result = chi_square_fit(data)
        fit_parameters = result[0]
        data = outlier_removal(data, fit_parameters)
        result = chi_square_fit(data)
        # fit_parameters contains the mass as its first index, width as its
        #second index
        updated_fit_parameters = result[0]
        plot_result(updated_fit_parameters, data, result[1])
        save_file(data, COMBINED_FILENAME)
        if SAVEFIG:
            print('A new file and figure has been saved.')
        else:
            print('A new file has been saved')
    except TypeError:
        print('Program failed. Please check if you have the correct files.')

    return 0



main()
