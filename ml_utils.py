import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import pyDOE
import itertools
from sklearn.preprocessing import MinMaxScaler
from numpy import linalg as LA

class MLUtils:
    def __init__(self):
        self.model = tf.keras.models.load_model('model/grad_descent_ml')
        # for input features
        #self.data = pd.read_csv('data/testing_set.csv')
        #input_data = self.data.loc[:, 'uOut':'grid_points']
        self.scaler = pickle.load(open('scaler/scaler.pkl', 'rb'))
        #self.scaler.fit_transform(input_data)
        self.r_a = np.array([0.5, 3.0, 0.5])

    def MapToLinSpace(self, dimension, lowerBound, upperBound):
        return np.multiply(dimension, (upperBound-lowerBound)) + lowerBound

    def GetLatinHypercubeSamples(self, sampleSize, lowerBound, upperBound):
        samples = pyDOE.lhs(1, samples=sampleSize)
        samples = self.MapToLinSpace(samples, lowerBound, upperBound)
        samples = np.concatenate(samples).ravel()
        samples = np.sort(samples)
        return samples

    def getPredictions(self, input_array):
        #dataframe
        dataframe = pd.DataFrame(data=input_array, columns=['uOut', 'rOut_x', 'rOut_y', 'rOut_z', 'g', 'uIn', 'TIn',
                                                            'density', 'mu_c', 'B', 'T_VF'])
        dataframe = dataframe.reindex(dataframe.index.repeat(100)).reset_index(drop=True)
        dataframe['grid_points'] = np.linspace(0.0, 1.0, num=100)
        predictions = self.model.predict(tf.convert_to_tensor(self.scaler.transform(dataframe), dtype=tf.float32))
        return predictions

    def getResidual(self, input_array):
        dataframe = pd.DataFrame(data=input_array, columns=['uOut', 'rOut_x', 'rOut_y', 'rOut_z', 'g', 'uIn', 'TIn',
                                                            'density', 'mu_c', 'B', 'T_VF'])
        dataframe = dataframe.reindex(dataframe.index.repeat(100)).reset_index(drop=True)
        dataframe['grid_points'] = np.linspace(0.0, 1.0, num=100)
        r_b = np.array([dataframe['rOut_x'][0], dataframe['rOut_y'][0], dataframe['rOut_z'][0]])
        L = LA.norm(r_b - self.r_a)
        Re = dataframe['density'][0] * L
        mu = dataframe['mu_c'][0] * np.exp(dataframe['B'][0] / (dataframe['TIn'][0] - dataframe['T_VF'][0]))
        Fr = 1/np.sqrt(dataframe['g'][0] * L)
        tau = (r_b - self.r_a) / L
        tau_g = tau[1] * -1

        dataframe = tf.convert_to_tensor(self.scaler.transform(dataframe), tf.float32)
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(dataframe)
            y_pred = self.model(dataframe)
            u = y_pred[:, 0:1]
            N = y_pred[:, 1:]
        du = tp.gradient(u, dataframe)[:, 11:]
        dN = tp.gradient(N, dataframe)[:, 11:]

        res_u = du - ((Re * N * u)/ (3 * mu))
        res_N = dN - du - ((1 * tau_g) / (Fr * Fr * u))

        res_u_rel = np.abs(res_u / np.maximum(np.abs(du), 1.0e-6))
        res_N_rel = np.abs(res_N / np.maximum(np.abs(dN), 1.0e-6))
        return res_u_rel, res_N_rel

    def getGradients(self, input_array):
        dataframe = pd.DataFrame(data=input_array, columns=['uOut', 'rOut_x', 'rOut_y', 'rOut_z', 'g', 'uIn', 'TIn',
                                                            'density', 'mu_c', 'B', 'T_VF'])
        dataframe = dataframe.reindex(dataframe.index.repeat(100)).reset_index(drop=True)
        dataframe['grid_points'] = np.linspace(0.0, 1.0, num=100)
        dataframe = tf.convert_to_tensor(self.scaler.transform(dataframe), tf.float32)
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(dataframe)
            y_pred = self.model(dataframe)
            u = y_pred[:, 0:1]
            N = y_pred[:, 1:]
        du = tp.gradient(u, dataframe)
        dN = tp.gradient(N, dataframe)
        del tp
        return du.numpy()[:, 11], dN.numpy()[:, 11]


    def getSurfacePredictions(self, col_name, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, nr_samples, der):
        uOut, rOut_x, rOut_y, rOut_z, g, uIn, TIn, density, mu_c, B, T_VF, grid_points, colName = [], [], [], [], [], \
                                                                                                  [], [], [], [], [], \
                                                                                                  [], [], []
        #print('col_name', col_name)
        if col_name == 'velocity_out':
            colName = self.GetLatinHypercubeSamples(nr_samples, 3, 84)
            uOut = colName
            rOut_x = [val1]
            rOut_y = [val2]
            rOut_z = [val3]
            g = [val4]
            uIn = [val5]
            TIn = [val6]
            density = [val7]
            mu_c = [val8]
            B = [val9]
            T_VF = [val10]
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        if col_name == 'rad_r_x':
            colName = self.GetLatinHypercubeSamples(nr_samples, 0, 2.5)
            uOut = [val1]
            rOut_x = colName
            rOut_y = [val2]
            rOut_z = [val3]
            g = [val4]
            uIn = [val5]
            TIn = [val6]
            density = [val7]
            mu_c = [val8]
            B = [val9]
            T_VF = [val10]
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        if col_name == 'rad_r_y':
            colName = self.GetLatinHypercubeSamples(nr_samples, 0, 2.5)
            uOut = [val1]
            rOut_x = [val2]
            rOut_y = colName
            rOut_z = [val3]
            g = [val4]
            uIn = [val5]
            TIn = [val6]
            density = [val7]
            mu_c = [val8]
            B = [val9]
            T_VF = [val10]
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        if col_name == 'rad_r_z':
            colName = self.GetLatinHypercubeSamples(nr_samples, 0, 2.5)
            uOut = [val1]
            rOut_x = [val2]
            rOut_y = [val3]
            rOut_z = colName
            g = [val4]
            uIn = [val5]
            TIn = [val6]
            density = [val7]
            mu_c = [val8]
            B = [val9]
            T_VF = [val10]
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        if col_name == 'gravity':
            colName = self.GetLatinHypercubeSamples(nr_samples, 5, 15)
            uOut = [val1]
            rOut_x = [val2]
            rOut_y = [val3]
            rOut_z = [val4]
            g = colName
            uIn = [val5]
            TIn = [val6]
            density = [val7]
            mu_c = [val8]
            B = [val9]
            T_VF = [val10]
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        if col_name == 'velocity_in':
            colName = self.GetLatinHypercubeSamples(nr_samples, 0.1, 1)
            uOut = [val1]
            rOut_x = [val2]
            rOut_y = [val3]
            rOut_z = [val4]
            g = [val5]
            uIn = colName
            TIn = [val6]
            density = [val7]
            mu_c = [val8]
            B = [val9]
            T_VF = [val10]
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        if col_name == 'temperature':
            colName = self.GetLatinHypercubeSamples(nr_samples, 500, 600)
            uOut = [val1]
            rOut_x = [val2]
            rOut_y = [val3]
            rOut_z = [val4]
            g = [val5]
            uIn = [val6]
            TIn = colName
            density = [val7]
            mu_c = [val8]
            B = [val9]
            T_VF = [val10]
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        if col_name == 'density':
            colName = self.GetLatinHypercubeSamples(nr_samples, 800, 2000)
            uOut = [val1]
            rOut_x = [val2]
            rOut_y = [val3]
            rOut_z = [val4]
            g = [val5]
            uIn = [val6]
            TIn = [val7]
            density = colName
            mu_c = [val8]
            B = [val9]
            T_VF = [val10]
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        if col_name == 'viscosity':
            colName = self.GetLatinHypercubeSamples(nr_samples, 0.01, 0.5)
            uOut = [val1]
            rOut_x = [val2]
            rOut_y = [val3]
            rOut_z = [val4]
            g = [val5]
            uIn = [val6]
            TIn = [val7]
            density = [val8]
            mu_c = colName
            B = [val9]
            T_VF = [val10]
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        if col_name == 'B':
            colName = self.GetLatinHypercubeSamples(nr_samples, 1500, 2500)
            uOut = [val1]
            rOut_x = [val2]
            rOut_y = [val3]
            rOut_z = [val4]
            g = [val5]
            uIn = [val6]
            TIn = [val7]
            density = [val8]
            mu_c = [val9]
            B = colName
            T_VF = [val10]
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        if col_name == 'T_VF':
            colName = self.GetLatinHypercubeSamples(nr_samples, 223.15, 283.15)
            uOut = [val1]
            rOut_x = [val2]
            rOut_y = [val3]
            rOut_z = [val4]
            g = [val5]
            uIn = [val6]
            TIn = [val7]
            density = [val8]
            mu_c = [val9]
            B = [val10]
            T_VF = colName
            grid_points = np.linspace(0.0, 1.0, num=nr_samples)

        feature_ist = [uOut, rOut_x, rOut_y, rOut_z, g, uIn, TIn, density, mu_c, B, T_VF, grid_points]
        dataframe = pd.DataFrame(list(itertools.product(*feature_ist)), columns=['uOut', 'rOut_x', 'rOut_y', 'rOut_z',
                                                                                 'g', 'uIn', 'TIn', 'density', 'mu_c',
                                                                                 'B', 'T_VF', 'grid_points'])
        if der == 0:
            predictions = self.model.predict(tf.convert_to_tensor(self.scaler.transform(dataframe), dtype=tf.float32))

            predictions_u = predictions[:, 0]
            predictions_u = predictions_u.reshape(50, 50)
            predictions_df_u = pd.DataFrame(data=predictions_u, index=colName, columns=grid_points)

            predictions_N = predictions[:, 1]
            predictions_N = predictions_N.reshape(50, 50)
            predictions_df_N = pd.DataFrame(data=predictions_N, index=colName, columns=grid_points)
            return predictions_df_u, predictions_df_N
        else:
            dataframe = tf.convert_to_tensor(self.scaler.transform(dataframe), tf.float32)
            with tf.GradientTape(persistent=True) as tp:
                tp.watch(dataframe)
                y_pred = self.model(dataframe)
                u = y_pred[:, 0:1]
                N = y_pred[:, 1:]
            du = tp.gradient(u, dataframe)
            dN = tp.gradient(N, dataframe)
            del tp

            gradients_u = du.numpy()[:, 11]
            gradients_u = gradients_u.reshape(50, 50)
            gradients_df_u = pd.DataFrame(data=gradients_u, index=colName, columns=grid_points)

            gradients_N = dN.numpy()[:, 11]
            gradients_N = gradients_N.reshape(50, 50)
            gradients_df_N = pd.DataFrame(data=gradients_N, index=colName, columns=grid_points)
            return gradients_df_u, gradients_df_N
