import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def rbe(alfax, betax, s, alfai, d):
    v = (np.sqrt((alfax ** 2) - (4 * betax * np.log(s))) - alfai) / (2 * betax * d)
    return v


def find_dose_to_survival(alfa, beta, survival):
    if beta > 0:
        dose1 = (-alfa + np.sqrt(alfa ** 2 - 4 *
                                 beta * np.log(survival))) / (2 * beta)
    else:
        dose1 = -np.log(survival) / alfa
    return dose1


def lq(alfa, beta, D, D2):
    return np.exp(-alfa * D - beta * D2)


def lector_dose_data_FLUKA(dose_data_path):
    ascii_file = open(dose_data_path, 'r')
    contador = 0
    doses = []
    for line in ascii_file:
        processed_line = list(filter(None, line.strip().split(' ')))
        if "Z coordinate" in line:
            # starting point in mm
            starting_point = float(processed_line[3]) * 10
            ending_point = float(processed_line[5]) * 10  # ending point in mm
            nbins = int(processed_line[7])
            bin_width = (ending_point - starting_point) / nbins
        elif contador >= 15:
            doses += [float(i) for i in list(filter(None, line.strip().split(' ')))]
        contador += 1
    depths = [starting_point + bin_width * i - bin_width / 2 for i in range(1, nbins + 1)]
    return depths, doses, nbins, bin_width


def lector_spectrum_data_FLUKA(spectrum_data_path, detector_depths):
    n_depths = len(detector_depths)
    spectrums = []  # tablas de fluencia o corriente [run001_depth_0,...,run001_depth_n,...,run005_depth_0,...,run005_depth_n,...]
    for file in os.listdir(spectrum_data_path):
        total_spectrum = []
        with open(os.path.join(spectrum_data_path, file), 'r') as database:
            contador = 0
            for line in database:
                stripped_split_line = list(filter(None, line.split(' ')))
                try:
                    if stripped_split_line[1] == 'energy' and contador == 0:
                        Emin = float(stripped_split_line[4])*1000  # MeV
                        Emax = float(stripped_split_line[6])*1000  # MeV
                        nbins = int(stripped_split_line[8])
                        binwidth = (Emax-Emin)/nbins
                        Ebins = [Emin+binwidth*i-binwidth /
                                 2 for i in range(1, nbins+1)]
                        contador += 1
                except:
                    continue
                if fully_floatable(stripped_split_line) == True:
                    total_spectrum += [float(i) for i in stripped_split_line]
        spectrum_separated_by_depth = [
            total_spectrum[i*nbins:(i+1)*nbins] for i in range(n_depths)]
        if n_depths != len(total_spectrum)/nbins:
            print("Error: Profundidades dadas no coinciden con espectro de Fluka")
        else:
            spectrums += spectrum_separated_by_depth

    spectrum_count = len(os.listdir(spectrum_data_path))

    spectrum_arrays = []
    spectrum_avg = np.zeros((n_depths, nbins))
    spectrum_variance = np.zeros((n_depths, nbins))

    for i in range(spectrum_count):
        spectrum_arrays.append(
            np.array(spectrums[i*n_depths:(i+1)*n_depths])*6.28318548)
    for i in range(spectrum_count):
        spectrum_avg += spectrum_arrays[i]/spectrum_count
    for i in range(spectrum_count):
        spectrum_variance += (spectrum_arrays[i] -
                              spectrum_avg)**2/spectrum_count

    spectrum_std = np.sqrt(spectrum_variance)

    return Ebins, spectrum_avg, spectrum_std


def fully_floatable(list):
    for element in list:
        try:
            float(element)
        except:
            return False
    return True


def fluka_to_list(spectrum_data_path, dose_data_path, dose_norm_max):
    # Esta función convierte información de fluka (Dosis y espectro a profundidad) a un dataframe
    # que luego se puede pasar directamente a un modelo de Machine learning para predecir survival

    spectrum_norm_path = os.path.normpath(spectrum_data_path)
    dose_data_path_norm = os.path.normpath(dose_data_path)

    doses = []
    doses_error = []
    doses_matrix = []
    for path in os.listdir(dose_data_path_norm):
        # Lectura de dosis desde archivo FLUKA (recorre carpeta con archivos de dosis)
        depths_doses, doses_run, bins_depth, sep = lector_dose_data_FLUKA(os.path.join(dose_data_path_norm, path))
        doses_matrix.append(doses_run)
    doses_matrix = np.transpose(np.array(doses_matrix))
    for i in range(len(depths_doses)):
        doses_at_same_depth = doses_matrix[i][:]
        dose_avg_at_depth = np.average(doses_at_same_depth)
        dose_std_at_depth = np.std(doses_at_same_depth)
        doses.append(dose_avg_at_depth)
        doses_error.append(dose_std_at_depth)
    detector_depths = np.array([0 + i * sep for i in range(0, bins_depth + 1)])   # asumiendo profundidad minima 0, cambiar despues

    dosemax = max(doses)
    doses = [i*dose_norm_max/dosemax for i in doses]
    interp_doses = np.interp(detector_depths, depths_doses, doses)

    # Lectura de espectro
    Ebins, spectrum_avg, spectrum_std = lector_spectrum_data_FLUKA(spectrum_norm_path, detector_depths)

    #### Calculo el promedio de energía en cada profundidad usando Ebins
    # usando las fluencias como pesos

    Edepths = []
    for index, detector_depth in enumerate(detector_depths):
        spectrum_at_detector_depth = spectrum_avg[index]
        Eavg = np.average(Ebins, weights=spectrum_at_detector_depth)
        Edepths.append(Eavg)
    Edepths = np.array(Edepths)

    return Edepths, interp_doses, detector_depths


path_dosis = 'C:\\Users\\mpiav\\Desktop\\interfaz_grafica_adn\\RESPONSE\\wouters2014\\wouters2014_calculation\\wouters2014_dose_data'.replace('\\', '\\\\')
path_espectro = 'C:\\Users\\mpiav\\Desktop\\interfaz_grafica_adn\\RESPONSE\\wouters2014\\wouters2014_calculation\\wouters2014_spectrum_data'.replace('\\', '\\\\')


def fluka_to_list_sin_promediar(spectrum_data_path, dose_data_path, dose_norm_max):
    # Versión alternativa de la función anterior, pero sin promediar las energías de cada depth

    spectrum_norm_path = os.path.normpath(spectrum_data_path)
    dose_data_path_norm = os.path.normpath(dose_data_path)

    doses = []
    doses_error = []
    doses_matrix = []
    for path in os.listdir(dose_data_path_norm):
        # Lectura de dosis desde archivo FLUKA (recorre carpeta con archivos de dosis)
        depths_doses, doses_run, bins_depth, sep = lector_dose_data_FLUKA(os.path.join(dose_data_path_norm, path))
        doses_matrix.append(doses_run)
    doses_matrix = np.transpose(np.array(doses_matrix))
    for i in range(len(depths_doses)):
        doses_at_same_depth = doses_matrix[i][:]
        dose_avg_at_depth = np.average(doses_at_same_depth)
        dose_std_at_depth = np.std(doses_at_same_depth)
        doses.append(dose_avg_at_depth)
        doses_error.append(dose_std_at_depth)
    detector_depths = np.array([0 + i * sep for i in range(0, bins_depth + 1)])   # asumiendo profundidad minima 0, cambiar despues

    dosemax = max(doses)
    doses = [i*dose_norm_max/dosemax for i in doses]
    interp_doses = np.interp(detector_depths, depths_doses, doses)

    # Lectura de espectro
    Ebins, spectrum_avg, spectrum_std = lector_spectrum_data_FLUKA(spectrum_norm_path, detector_depths)

    # En este caso no se promedia para cada depth, se retorna todo para luego obtener survival
    # para cada energía en ese depth, luego se promedian (pesando por la fluencia)

    fluencias = []
    for index, detector_depth in enumerate(detector_depths):
        spectrum_at_detector_depth = spectrum_avg[index]
        fluencias.append(spectrum_at_detector_depth)

    return fluencias, interp_doses, detector_depths, Ebins


def read_set_experimental(file):
    x = []
    y = []
    data = open(file, 'r')
    for line in data:
        line = line.replace(';', '')
        line_list = line.split(' ')
        try:
            x.append(float(line_list[0]))
            y.append(float(line_list[1]))
        except ValueError as error:
            # hay que cambiar las comas (,) por puntos (.) en los datos
            x.append(float(line_list[0].replace(',', '.')))
            y.append(float(line_list[1].replace(',', '.')))
    return x, y


def guardar_datos(x, y, name):
    # recibe arrays x, y, los guarda en .txt
    directory = os.getcwd()
    path_file = os.path.join(directory, f'{name}.txt')
    file = open(path_file, 'w')
    for i in range(len(x)):
        file.write(f'{x[i]}; {y[i]}\n')
    file.close()

    return path_file
