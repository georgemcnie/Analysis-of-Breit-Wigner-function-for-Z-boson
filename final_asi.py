"""
PHYS20101 Final assignment: Program to calculate mass and width of Z_0 boson

This program will allow the user to calculate the mass, width and lifetime of a
Z_0 boson given some data from an experiment. It calculates this when the Z_0
boson is produced from an electron positron pair, and can take multiple
different types of products.

1) data is read in and validated, and however many data files there are can be
   combined
2) large outliers from data are removed, so eroneous that truly are obvious
   outliers
3) an initial value of mass and width is calculated just from data by trying
   to fit a gaussian to it
4) a minimised chi square fit is done by simultaneously varying mass and width
   of Z_0 boson. This gives the best mass and width to fit a funtion to data
5) data point that are 3* their uncertainty away from fit are anomilies so
   removed. Fit done repeatidly until no more uncertainties found
6) mass, width and lifetime now found
7) Contour plot produced to calculate uncertainties on mass and width
8) Plot produced and saved to show what data is showing




AUTHOR:
George McNie 15/12/2021
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy import constants
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

def general_numerical_validation(input_value):
    """
    A general function used for numerical inputs to check they're positive
    and are either a float or integer
    Returns the input value if conditions are met and false if conditions not
    met

    Args:
        input_value: unknown variable type (due to purpose of function)
    Returns:
        False: boolean
        input_value: float
    Raises:
        ValueError: If argument cannot be converted into a string

    """
    try:
        input_value = float(input_value)
    except ValueError:
        print("******-------Invalid input, please enter a valid number"
              "-------******")
        return False
    else:
        if input_value <= 0:
            print("******-------Invalid input, please enter a valid"
                  " number-------******")
            return False
        return input_value #if gets all right validation returns the number


def choose_products_function():
    """
    A function to allow the user to pick the products the z_0 boson decays into
    after it's been created. Takes an input and keeps running function until
    this input is valif.

    Returns
    -------
    float
        returns the partial width of the decay product in Gev

    """
    products = [['electron and positron', 83.91e-3],
                ['a positive and negative muon', 83.99e-3],
                ['a positive and negative tau particle', 84.08e-3]]

    for i, value in enumerate(products):
        print('{0}  =  {1}'.format(i+1, value[0]))
    chosen_product = input("Please select what Z_0 decays into: ")
    if general_numerical_validation(chosen_product): #checked if number
        chosen_product = float(chosen_product) #don't know if int or float
        chosen_product = round(chosen_product, 0) #round to get an integer
        chosen_product = int(chosen_product) #array parameter needs to be int
        if 1 <= chosen_product <= 3:
            print("You've picked your decay products to be "
                  +products[chosen_product-1][0])
            return products[chosen_product-1][1] # partial width of product
        print("******-------Invalid input, please enter a valid number-----"
              "--******")
        return choose_products_function()
    return choose_products_function()


def start_up():
    """
    prints the start up message and then calls the choose_products_function
    funtion

    Returns
    -------
    choose_products_function : float
        will return the value obtained in the choose_products_function funtion

    """
    print("******************************************************************"
          "****\n*--Welcome to the program that calculates the width and mass "
          "of a Z--*\n*--boson given data from an experiment of a colliding "
          "electron and --*\n*--- positron. There can be many different "
          "products of the z boson---*\n**************************************"
          "********************************")
    return choose_products_function()


def check_3_columns(input_array):
    """
    A function to check that a numpy array has 3 columns
    Used to check that data read in is in correct format

    Parameters
    ----------
    input_array : numpy array
        the array who's shape needs to be checked

    Returns
    -------
    is_three_columns : Boolean
        True when in correct shape (has 3 columns)
        False when in a shape that program cannot compute (not 3 columns)

    """
    _, columns = np.shape(input_array) #row part of shape uncalled
    if columns != 3:
        #returns function as false if anything in the input array isn't 3
        is_three_columns = False
    else:
        is_three_columns = True
    return is_three_columns


def read_data(file_name):
    """
    function to read in data from a data file. A numpy array is created
    corresponding to the file. If the file is in the wrong format or cannot be
    found the whole program stops. If in correct format (anything by 3) then
    numpy array with the values returned.

    Parameters
    ----------
    file_name : string
        the name of the file to be read in

    Returns
    -------
    file_array : numpy array
        array from data in the file

    """
    file_name = str(file_name) #makes sure file_name is a string so no error
                                #when openiing
    try:
        file_1 = open(file_name, "r")
    except FileNotFoundError:
        print('Unable to open file with name{0}'.format(file_name))
        sys.exit()
    file_array = np.genfromtxt(file_1, delimiter=",")
    if check_3_columns(file_array):
        return file_array
    file_1.close()
    print("Your input data file contains too many columns")
    sys.exit()


def combine_arrays(array_original, array_to_be_combined):
    """
    A function to combine two different arrays. The array will just be added
    onto the end of the original array and so they're stored in same array

    Parameters
    ----------
    array_original : numpy array
        the origianl array
    array_to_be_combined : numpy array
        the array to be combined with the original array (must be same size)

    Returns
    -------
    stacked: numpy array
        a combined array of the two parameters

    """
    stacked = array_original
    for current_array in array_to_be_combined:
        stacked = np.vstack((stacked, current_array))
    return stacked


def remove_nan_and_zeros(array_original):
    """
    A function to remove any nans (not a number) and any zeros in a numpy
    array. The nans arise from when data is read in (in a different function).

    Parameters
    ----------
    array_original : numpy array
        the original array which may contain nans and zeros

    Returns
    -------
    array_without_zeros_and_nan : numpy array
        the original array but rows that contain nans and zeros have been
        deleted

    """
    array_without_nan = array_original[~np.isnan(array_original).any(axis=1),
                                       :]
    array_without_zeros_and_nan = array_without_nan[
        ~np.any(array_without_nan == 0, axis=1), :]
    sort_ammended = array_without_zeros_and_nan[
        np.argsort(array_without_zeros_and_nan[:, 0])]
    return sort_ammended


def remove_large_outliers_function(data):
    """
    A function to remove extremely large outliers that clearly doesn't fit
    with the data. Does so by taking the median value and if it's more than 3
    standard deviations away then the point is removed. Median used as mean
    would get skewed by large outliers

    Parameters
    ----------
    data : numpy array
        the data in which the extremely large outlier is being removed from

    Returns
    -------
    data

    """
    median = np.median(data[1])
    standard_deviation = np.std(data[1])
    corrected_data = data
    counter = 0
    removed = False
    for index, array in enumerate(data):
        if index - counter <= len(data):
            #index - counter so don't get index error once line is removed
            if array[1] > median + 3*standard_deviation:
                corrected_data = np.delete(corrected_data, (index - counter),
                                           axis=0)
                counter += 1
                removed = True
    return corrected_data, removed


def gaussian(x_value, amp, cen, wid):
    """
    A gaussian funtion

    Parameters
    ----------
    x_value : float
        the input value of gauss function
    amp : float
        the height of the curves peak
    cen : float
        the center of the curves peak
    wid : float
        the standard distribution or width of the gaussian distribution

    Returns
    -------
    float
        a float of the height the gaussian distribution specified is at,
        at value x

    """
    return amp * np.exp(-(x_value - cen)**2 / wid)


def guess_initial_width_and_mass(data):
    """
    This function guess' the initial mass and width of the z_0 boson. It does
    this by reading in the data and assuming it sort of resembles a gaussian.
    mass is approximated as it's the value centered over the distribuion.
    width is approximated as it's the full-width at half-maximum of the
    distribuion.
    The funtion curve fit finds the best value of center, width and height that
    matches the data. Then the initial guess of mass and gamma are calculated.

    Parameters
    ----------
    data : numpy array
        the data contains 3 columns: energies, cross sections and uncertainties
        on the cross sections

    Returns
    -------
    width_guess : float
        a guess on what the width of the z_boson is
    mass_guess : float
        a guess on what the mass of the z_boson is

    """
    # very rough guess' of gaussian center, width and height
    mean = np.mean(data[:, 0])
    sigma = np.std(data[:, 0])
    difference_of_y = np.amax(data, axis=0)[1]-np.amin(data, axis=0)[1]

    popt, *___ = curve_fit(gaussian, data[:, 0], data[:, 1],
                           p0=[difference_of_y, mean, sigma])
    amplitude, center, width = popt

    gaussian_plot = np.transpose(np.array(gaussian(data[:, 0], amplitude,
                                                   center, width)))
    # finds points at half height of the gaussian plot
    spline = UnivariateSpline(data[:, 0], gaussian_plot-np.max(gaussian_plot)/2,
                              s=0)
    root_1, root_2 = spline.roots() # find the roots of points at half height
    width_guess = abs(root_1-root_2)
    mass_guess = center
    return width_guess, mass_guess


def cross_section_function(energy, mass, width, partial_width_end):
    """
    A function to give the output of the formula for a cross section of the
    e- e+ --> Z^0 --> f f- interaction. Where ff- is the particle antiparticle
    pair the Z boson decays into

          12(pi) * (energy before)^2 * partiaal width start * partial width end
    sigma = -------------------------------------------------------------------
           (mass)^2 * [({energy before}^2 - {mass}^2)^2 + (mass)^2 * (width)^2]

    Parameters
    ----------
    energy : numpy array
        numpy array of energy values in GeV (energies are from data given)
    mass : float
        a value for mass of Z boson
    width : float
        a value for width of Z boson

    Returns
    -------
    cross_section : numpy array
        a numpy array with values of expected crosssection for given values of
        energy

    """
    gamma_ee = 83.91 * 10**-3 #convert to giga
    numerator = 12*np.pi * energy**2 * gamma_ee * partial_width_end
    denominator = mass**2 * (((energy)**2 - (mass)**2)**2 + (mass**2 *
                                                             width**2))
    cross_section = numerator / denominator

    cross_section = cross_section * 0.3894e6
    return cross_section


def chi_square(parameters_to_minimise, observation, observation_uncertainty,
               energies, partial_width):
    """
    calculates the chi square. for each value do observed value minus expected
    value all square divided by the square of the uncertainty

    Parameters
    ----------
    parameters_to_minimise : tuple
        the parameters to put into the expected value function. In this form
        for the f_min in fit_to_parameters function. For this code, expect to
        be mass and width of z boson.
    observation : numpy array
        the values seen in the experiment
    observation_uncertainty : numpy array
        the uncertainty on the value seen in the experiment
    energies : numpy array
        the value of energy for each corresponding observation
    partial_width: float
        the partial width of the products of the decay pathway chosen

    Returns
    -------
    float
        returns the chi square of the parameters

    """
    mass, width = parameters_to_minimise
    return np.sum((observation -
                   cross_section_function(energies, mass, width,
                                          partial_width))**2 /
                  observation_uncertainty**2)


def reduced_chi_square(parameters_to_minimise, observation,
                       observation_uncertainty, energies, partial_width):
    """
    Function to calculate the reduced chi square. Calls the chi square
    function to get normal chi square and then divides by the amount of
    observations - degree of freedom (in this case 2)

    Parameters
    ----------
    parameters_to_minimise : tuple
        the parameters to put into the expected value function. In this form
        for the f_min in fit_to_parameters function. For this code, expect to
        be mass and width of z boson.
    observation : numpy array
        the values seen in the experiment
    observation_uncertainty : numpy array
        the uncertainty on the value seen in the experiment
    energies : numpy array
        the value of energy for each corresponding observation
    partial_width: float
        the partial width of the products of the decay pathway chosen

    Returns
    -------
    reduced_chi : float
        returns the reduced chi square

    """
    chi_squared = chi_square(parameters_to_minimise, observation,
                             observation_uncertainty, energies, partial_width)
    reduced_chi = chi_squared/(len(observation)-len(parameters_to_minimise))
    return reduced_chi


def remove_smaller_outliers(data, mass, width, partial_width):
    """
    Function to remove small outliers from the data. Works by seeing if the
    data point is more than 3 times its uncertainty from the fit. If it is
    then can say

    Parameters
    ----------
    data : numpy array
        is a numpy array with energies, cross sections, uncertainties in first
        second and third columns respectively.
    mass : float
        the mass of the z boson
    width : float
        the width of the z boson
    partial_width: float
        the partial width of the products of the decay pathway chosen

    Returns
    -------
    data_filtered : numpy array
        returns data that is less than 3 times a point's uncertainty away from
        the fit
    points_removed : TYPE
        returns data that is removed, so less than 3 times a point's
        uncertainty away from the fit.

    """
    data_filtered = data[abs(cross_section_function(data[:, 0], mass, width,
                                                    partial_width)
                             - data[:, 1]) < 3*data[:, 2]]
    points_removed = data[abs(cross_section_function(data[:, 0], mass, width,
                                                     partial_width)
                              - data[:, 1]) > 3*data[:, 2]]
    return data_filtered, points_removed


def fit_parameters_to_function(data, mass_given, width_given, partial_width):
    """
    function finds the minimum chi square by varying the parameters of the chi
    square. These parameters are then run through to check if there's any
    outliers corresponding to the new fit with the parameters found.
    If there are no outliers to the fit then the data is optimised to the fit.

    Parameters
    ----------
    data : numpy array
        is a numpy array with energies, cross sections, uncertainties in first
        second and third columns respectively.
    mass_given : float
        the initial mass to try a fit for
    width_given : float
        the initial gamma (or width) to try a fit for
    partial_width: float
        the partial width of the products of the decay pathway chosen

    Returns
    -------
    data_corrected : numpy array
        the new data array without the outliers to the fit
    points_removed : numpy array
        the data removed as it's more than 3 standard deviations from the fit
    mass_found : float
        the mass corresponding to the lowest chi square of fit
    width_found : float
        the gamma (or width) corresponding to the lowest chi square of fit
    is_optimised : Boolean
        will say if the data is optimised (if no outliers to fit identified)

    float
        will return the value for the chi square corresponding to parameters
        found

    """
    minimised = fmin(chi_square, (mass_given, width_given),
                     args=(data[:, 1], data[:, 2], data[:, 0], partial_width),
                     full_output=1)
    # full output gets minimised parameters and the corresponding chi square
    mass_found, width_found = minimised[0]
    data_corrected, points_removed = remove_smaller_outliers(data, mass_found,
                                                             width_found,
                                                             partial_width)
    # if no points are removed after the fit, then data is optimised for fit
    if len(points_removed) == 0:
        is_optimised = True
    else:
        is_optimised = False
    return (data_corrected, points_removed, mass_found, width_found,
            is_optimised, minimised[1])


def mesh_arrays(x_array, y_array):
    """
    a funtion to create mesh arrays from normal numpy arrays. Used in contour
    plots, so x_array is values for data on x axis and y_array is values for
    data on y axis

    Parameters
    ----------
    x_array : numpy array
        array to be meshed
    y_array : numpy array
        array to be meshed

    Returns
    -------
    x_array_mesh : numpy array
        is an array formatted as below

                    columns = HOW MANY VALUES IN x_array
                       _                     _
                      |X1  X2  X3  X4  X5  X6|
                      |X1  X2  X3  X4  X5  X6|  rows = HOW LONG y_array is
                      |X1  X2  X3  X4  X5  X6|
                      -       ect...        -


    y_array_mesh : numpy array
        is an array formatted as below

                    columns = HOW MANY VALUES IN y_array
                       _                     _
                      |Y1  Y2  Y3  Y4  Y5  Y6|
                      |Y1  Y2  Y3  Y4  Y5  Y6|  rows  = HOW LONG x_array is
                      |Y1  Y2  Y3  Y4  Y5  Y6|
                      -       ect...        -

    """
    x_array_mesh = np.empty((0, len(x_array)))

    for _ in y_array:  # _ is an uncalled variable.
        x_array_mesh = np.vstack((x_array_mesh, x_array))

    y_array_mesh = np.empty((0, len(y_array)))

    for _ in x_array:  # _ is an uncalled variable.
        y_array_mesh = np.vstack((y_array_mesh, y_array))

    y_array_mesh = np.transpose(y_array_mesh)

    return x_array_mesh, y_array_mesh


def get_chi_square_array(mass, width, data, partial_width):
    """
    calculates the chi square array from meshed arrays of parameters to be
    fitted
    Firstly gets list for mass and gamma, then gets a mesh array for the chi
    square in which chi square calculated for different values of mass and
    width

    Parameters
    ----------
    mass : float
        mass calculated of the m_z boson
    width : float
        width calculated of the m_z boson
    data : numpy array
        array of the data with columns energies, cross sections and cross
        sections uncertainty
    partial_width: float
        the partial width of the products of the decay pathway chosen

    Returns
    -------
    mass_values_mesh : numpy array
        a mesh of the mass values
    width_values_mesh : numpy array
        a mesh of the width values
    chi_square_array : numpy array
        a mesh of the reduced chi square corresponding to the mass mesh and
        width mesh

    """
    # gets values of mass and width so resolution good on graph plotted
    mass_values = np.linspace(mass-0.001*mass, mass+0.001*mass, 47)
    width_values = np.linspace(width-0.01*width,
                               width+0.01*width, 47)

    mass_values_mesh, width_values_mesh = mesh_arrays(mass_values,
                                                      width_values)

    chi_square_array = np.zeros(np.shape(mass_values_mesh)) # gets a zero
    for row in range(chi_square_array.shape[0]):
        for column in range(chi_square_array.shape[1]):
            chi_square_array[row, column] = chi_square(
                (mass_values_mesh[row, column],
                 width_values_mesh[row, column]), data[:, 1], data[:, 2],
                data[:, 0], partial_width)
    return mass_values_mesh, width_values_mesh, chi_square_array


def calculate_uncertainties(coor_path):
    """
    calculates the uncertainties on a contour path from a contour graph

    used to calculate uncertainties on width and mass using chi square +1
    contour. Calculated on y contour as difference of top and bottom divided by
    2. Calculated on x contour as difference of first edge and last edge
    divided by 2.

    Parameters
    ----------
    coor_path : numpy array
        gives coordinates of contour want to find uncertainties on

    Returns
    -------
    sigma_y : float
        the uncertainty on the y value of contour
    sigma_x : float
        the uncertainty on the x value of contour

    """
    sigma_y = (np.amax(coor_path[:, 1]) - np.amin(coor_path[:, 1]))/2
    sigma_x = (np.amax(coor_path[:, 0]) - np.amin(coor_path[:, 0]))/2
    return sigma_y, sigma_x


def contour_plot(data, mass, width, min_chi, partial_width):
    """
    Creates a contour plot. This is needed to calculate the uncertainty on the
    mass and the width calculated.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    mass : TYPE
        DESCRIPTION.
    width : TYPE
        DESCRIPTION.
    min_chi : TYPE
        DESCRIPTION.
    partial_width : TYPE
        DESCRIPTION.

    Returns
    -------
    coordinates_path : TYPE
        DESCRIPTION.

    """
    mass_mesh, width_mesh, chi_square_meshed = (
        get_chi_square_array(mass, width, data, partial_width))
    fig = plt.figure()

    axis = fig.add_subplot(111)
    plot_chi_plus_one = axis.contour(mass_mesh, width_mesh, chi_square_meshed,
                                     [min_chi+1])
    path = plot_chi_plus_one.collections[0].get_paths()[0]
    coordinates_path = path.vertices # gets the coordinates of the contour

    axis.plot(mass, width, marker="o", label='minimum')

    other_contours = axis.contour(mass_mesh, width_mesh, chi_square_meshed,
                                  [min_chi+2.3, min_chi+5.99, min_chi+9.21],
                                  colors=['red', 'orange', 'blue'],
                                  linestyles=['dashed', 'dashdot', 'dotted'])

    axis.clabel(other_contours, inline=1, fontsize=10)
    axis.set_xlabel(r'Mass  ($ \frac{\rm GeV}{{\rm c}^2} $)')
    axis.set_ylabel("Gamma (GeV)")

    labels = [r'$\chi^2_{{\mathrm{{min.}}}}+1.00$',
              r'$\chi^2_{{\mathrm{{min.}}}}+2.30$',
              r'$\chi^2_{{\mathrm{{min.}}}}+5.99$',
              r'$\chi^2_{{\mathrm{{min.}}}}+9.21$']
    # Want plot legend outside of plot area, need to adjust size of plot so
    # that it is visible
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    for index, label in enumerate(labels):
        axis.collections[index].set_label(label)
    axis.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    axis.hlines(y=width, xmin=mass-0.001*mass, xmax=mass, color='black',
                alpha=0.5)
    axis.vlines(x=mass, ymin=width-0.01*width, ymax=width, color='black',
                alpha=0.5)
    axis.set_title('Contour plot showing the minimised point')
    plt.savefig('contour_plot.png', dpi=300)
    plt.savefig('cross_section_against_enrgy.png', dpi=300,
                bbox_inches='tight') # bounds figure when saved

    plt.show()
    return coordinates_path


def get_lifetime(width):
    """
    function to calculate lifetime of the z_0 boson.

    Parameters
    ----------
    width : float
        the width of the z boson (sometimes referred to as gamma)

    Returns
    -------
    float
        the lifetime of the Z boson

    """
    conversion = (constants.e)*10**9 # means that when inputted in GeV get in J
    return constants.hbar / (width*conversion)


def get_lifetime_uncertainty(width, width_uncertainty):
    """
    function to calculate uncertainty on lifetime of the z_0 boson.

    Parameters
    ----------
    width : float
        the width of the z boson (sometimes referred to as gamma)
    width_uncertainty : float
        the uncertainty on width of the z boson

    Returns
    -------
    float
        returns the uncertainty on the lifetime

    """
    conversion = (constants.e)*10**9 # means that when inputted in GeV get in J
    return (width_uncertainty/width) * constants.hbar / (width*conversion)


def plot_of_data(data, parameters, removed_data, parameter_uncertainties,
                 chi_r):
    """
    Plots the raw data with their uncertainties. Plots the fit calculated that
    goes through data. Shows the removed data points. Shows all values
    calculated with fit. Shows the residuals of the data. Saves as a png file.

    Parameters
    ----------
    data : numpy array
        array of data points to be plotted, has 3 columns energies, cross
        section and cross section uncertainty
    parameters : tuple
       has the values of the parameters for the cross section function
    removed_data : numpy array
       array of data points that have been removed. Has 3 columns energies,
       cross section and cross section uncertainty
    parameter_uncertainties : tuple
        has the uncertainties on the parameters calculated, so the
        uncertainties on mass and width
    chi_r : float
        has the value of the reduced chi square for the fit calculated

    Returns
    -------
    None.

    """

    energies = data[:, 0]
    cross_sections = data[:, 1]
    mass, width, partial_width = parameters
    width_uncert, mass_uncert = parameter_uncertainties

    fig = plt.figure(figsize=(8, 6))
    axes_data = fig.add_subplot(211)
    axes_data.plot(energies, cross_section_function(energies, mass, width,
                                                    partial_width),
                   label='Fit function', color='red', linestyle='--')
    axes_data.plot(removed_data[:, 0], removed_data[:, 1], linestyle='none',
                   marker='*', color='black', label='removed points')
    axes_data.errorbar(energies, cross_sections, data[:, 2], linestyle="none",
                       label='Data', alpha=0.5, fmt='o', ecolor='blue',
                       markersize='2')
    axes_data.legend(loc='upper left')
    axes_data.set_xlabel("Energy (GeV$^{-2}$)", fontname='Arial',
                         fontsize='12')
    axes_data.set_ylabel(r"$\sigma $ (nb)", fontname='Arial', fontsize='12',)
    axes_data.set_title('A plot of Cross section against energy',
                        fontname='Arial', fontsize='14', weight='bold')
    axes_data.grid(dashes=[4, 2], linewidth=1.2)

    axes_data.annotate(('Mass = {0:.4g} ± {1:.2f}'.format(mass, mass_uncert)),
                       (0, 0), (0, -55), xycoords='axes fraction', va='top',
                       textcoords='offset points', fontsize='10')
    axes_data.annotate((r'$\Gamma$ = {0:.4g} ± {1:.4g}'.format(width,
                                                               width_uncert)),
                       (0, 0), (0, -70), xycoords='axes fraction', va='top',
                       textcoords='offset points', fontsize='10')
    axes_data.annotate(('Lifetime = {0:.3g} ± {1: .1g}'.format(
        get_lifetime(width), get_lifetime_uncertainty(width, width_uncert))),
                       (0, 0), (200, -55), xycoords='axes fraction',
                       textcoords='offset points', va='top', fontsize='10')
    axes_data.annotate((r'$\chi^2$ Reduced = {0:.4g}'.format(chi_r)), (0, 0),
                       (200, -70), xycoords='axes fraction',
                       textcoords='offset points', va='top', fontsize='10')

    # residuals plot
    residuals = cross_sections - cross_section_function(energies, mass, width,
                                                        partial_width)
    axes_residuals = fig.add_subplot(414)
    axes_residuals.errorbar(energies, residuals, yerr=data[:, 2],
                            markersize='2', fmt='o', color='k')
    axes_residuals.plot(energies, 0 * energies, color='red')
    axes_residuals.grid(True)
    axes_residuals.set_title('Residuals', fontsize=14, fontname='Arial',
                             weight='bold')

    plt.savefig('cross_section_against_energy.png', dpi=300,
                bbox_inches='tight') # bounds figure when saved
    plt.show()

def main():
    """
    Main bit of code that runs everything

    Returns
    -------
    None.

    """
    partial_width = start_up()
    file_a = read_data("z_boson_data_1.csv")
    file_b = read_data("z_boson_data_2.csv")

# =============================================================================
# Arrays combined
# All nans and zeros removed to give
# This then used to get rid of large outliers
# =============================================================================
    data_no_large_outliers, is_removed = remove_large_outliers_function(
        remove_nan_and_zeros(combine_arrays(file_a, file_b)))

    # runs through data until no outliers found by removing large outliers
    while is_removed:
        data_no_large_outliers, is_removed = remove_large_outliers_function(
            data_no_large_outliers)

    width_start, mass_start = guess_initial_width_and_mass(
        data_no_large_outliers) # guess' the mass and width roughly

    # does the first fit with the guessed parameters
    points, new_anomalies, mass_obtained, width_obtained, is_finished, \
    min_chi_square = fit_parameters_to_function(data_no_large_outliers,
                                                mass_start, width_start,
                                                partial_width)
    all_anomalies = new_anomalies # to keep track of removed points

# =============================================================================
# runs through fitting function until no new anomolies are found, thus have
# optimised fit
# =============================================================================
    while not is_finished:
        points, new_anomalies, mass_obtained, width_obtained, is_finished, \
            min_chi_square = fit_parameters_to_function(points, mass_obtained,
                                                        width_obtained,
                                                        partial_width)
        parameters_obtained = (mass_obtained, width_obtained, partial_width)
        all_anomalies = np.append(all_anomalies, new_anomalies, axis=0)
        new_anomalies = []

    reduce_chi = reduced_chi_square((mass_obtained, width_obtained),
                                    points[:, 1], points[:, 2], points[:, 0],
                                    partial_width)

    # produces a contour plot to calculate uncertainty on mass and width
    coordinate_for_uncert = contour_plot(points, mass_obtained, width_obtained,
                                         min_chi_square, partial_width)
    # plots valid data and the fit, displa
    plot_of_data(points, parameters_obtained, all_anomalies,
                 calculate_uncertainties(coordinate_for_uncert), reduce_chi)


# mainguarding code, allows to run code in another scripy and it won't run
if __name__ == '__main__':
    main()
