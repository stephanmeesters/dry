'''
Created on 1 May 2018

High level class containing the necessary methods.

@author: Miguel Molina Romero, Technical University of Munich
@contact: miguel.molina@tum.de
@License: LPGL
'''

from utils import generate_synthetic_data
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import load_model as keras_load_model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import nibabel as nb
import numpy as np
import os
from dipy.io.gradients import read_bvals_bvecs

from scipy.ndimage.filters import gaussian_filter

# from pdb import set_trace as bp

import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Dry:
    '''Dry free-water with artificial neural networks (ANN).

    Functions:
        - train_model: trains an ANN model from the given b-values fileself.
        - load_model: loads an existing ANN model from the given file.
        - save_model: saves a new trained model.
        - correct_fwe: Corrects free-water contamination from a list diffusion
                       weighted volumes using the given model.
    '''

    def _get_ann_architecture(self, ninputs, hidden_layers):
        '''Builds the ANN architecture according from the number of inputs and
        hidden layers.'''
        hidden_layers = hidden_layers
        model = Sequential()
        # Input layers
        model.add(Dense(int(ninputs/hidden_layers*1.5), input_dim=ninputs,
                        kernel_initializer="RandomNormal",
                        bias_initializer='RandomNormal', activation="softmax"))
        # Hidden hidden
        for l in np.arange(hidden_layers - 1):
            model.add(Dense(int(ninputs/(hidden_layers*(l+2))*1.5),
                            kernel_initializer="RandomNormal",
                            bias_initializer='RandomNormal',
                            activation="softmax"))
        # Output layer
        model.add(Dense(1, activation="linear"))
        plot_model(model, to_file='model.png', show_shapes=True,
                   show_layer_names=True, rankdir='TB')
        return model

    def train_model(self, bfile):
        '''Creates a new model based on the b-values file and trains it'''
        # Model architecture based on the diffuson protocol (bval file)
        bvals, _ = read_bvals_bvecs(bfile, None)
        unique_bvals = np.unique(bvals)

        # Train the model
        print("Compiling model...")
        rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        mae = 1

        while mae > 0.1:
            # Generate training synthetic data
            model = self._get_ann_architecture(bvals.size, unique_bvals.size)
            model.compile(loss="mean_absolute_error", optimizer=rms)
            tdata = generate_synthetic_data(bfile)
            trainData, testData, traintruth, testtruth = train_test_split(
                tdata['S'], tdata['f'], test_size=0.15, random_state=42)

            model.fit(trainData, traintruth, epochs=10, batch_size=32,
                      verbose=1)
            mae = model.evaluate(testData, testtruth, batch_size=32,
                                 verbose=1)
            print("Loss of the testing dataset: {:.3}".format(mae))
        return model

    def load_model(self, mfile):
        '''Load a previously trained and saved model. This together with
        save_model are useful to store a trained model than can be used later
        on with the same diffusion protocol.'''
        return keras_load_model(mfile)

    def save_model(self, model, mfile):
        '''Saves a trained model. This together with load_model are useful to
        store a trained model than can be used later on with the same diffusion
        protocol.'''
        if not model:
            raise ValueError("Cannot save an empty model.")
        model.save(mfile)

    def fwe(self, dwis, model, bfile, output_folder=None, mask=None):
        '''Runs the diffusion data through the model to correct for free
        water contamination'''

        if not output_folder:
            output_folder = "."

        if not model:
            raise ValueError("A Sequential Keras model is required.")

        if not dwis:
            raise ValueError("Diffusion data is necessary.")

        bvals, _ = read_bvals_bvecs(bfile, None)

        def _process_dwi(model, dwi, bvals, output_folder, mask):
            print("Processing file {}".format(dwi))
            # Load and prepare data
            dwidata, maxdwi, dims, dwinii = self._prepare_dwi_data(dwi)
            maskdata = self._prepare_mask(mask)
            # Predict tissue volume fraction and correct for free-water
            f = model.predict(dwidata)
            f[np.isnan(f)] = 0
            f[f < 0] = 0
            f[f > 1] = 1
            St, Scsf = self._remove_fw_component(dwidata, maxdwi, f, bvals, maskdata)
            # Save tissue volume fraction and corrected dwi
            f = np.reshape(f, dims[0:3])
            St = np.reshape(St, dims)
            niif = nb.Nifti1Image(f, dwinii.affine)
            dwiname = os.path.basename(dwi)
            dwiname = dwiname.split('.')
            output_folder = os.path.join(output_folder, dwiname[0])
            mkdir_p(output_folder)
            nb.save(niif, os.path.join(output_folder,
                                       "tissue_volume_fraction.nii.gz"))
            niiS = nb.Nifti1Image(St, dwinii.affine)
            nb.save(niiS, os.path.join(output_folder, "fwe_dwi.nii.gz"))

            # Save free water map
            Scsf = np.reshape(Scsf, dims)
            niiScsf = nb.Nifti1Image(Scsf, dwinii.affine)
            nb.save(niiScsf, os.path.join(output_folder, "Scsf.nii.gz"))

        for dwi in dwis:
            _process_dwi(model, dwi, bvals, output_folder, mask)

    def fwe_tissue(self, dwi, tissue_volume_fraction, bfile,
                   output_folder=None):
        """Given the DWI volumes and the tissue volume fraction extracts the
        free water component from the diffusion data and saves it. This option
        might be useful when the tisseu volume fraction has been computed with
        only a part of the b-values (<1500 s/mm^2), but the all the diffusion
        volumes need to be corrected. Contrary to fwe method it only accepts
        one dwi file and its corresponding tissue volume fraction at a time."""
        if not output_folder:
            output_folder = "."
        if not dwi:
            raise ValueError("Diffusion data is necessary.")
        if not tissue_volume_fraction:
            raise ValueError("Tissue volume fraction data is necessary.")
        bvals, _ = read_bvals_bvecs(bfile, None)

        dwidata, maxdwi, dims, dwinii = self._prepare_dwi_data(dwi)
        f = nb.load(tissue_volume_fraction).get_data()
        f = np.reshape(f, (np.prod(dims[0:3]), 1))
        St = self._remove_fw_component(dwidata, maxdwi, f, bvals)
        St = np.reshape(St, dims)
        niiS = nb.Nifti1Image(St, dwinii.affine)
        dwiname = os.path.basename(dwi)
        dwiname = dwiname.split('.')
        output_folder = os.path.join(output_folder, dwiname[0])
        mkdir_p(output_folder)
        nb.save(niiS, os.path.join(output_folder, "fwe_dwi.nii.gz"))

    def _prepare_dwi_data(self, dwi):
        """Private method. Concatenates and normalizes dwi into a 2D matrix"""
        dwinii = nb.load(dwi)
        dwidata = dwinii.get_fdata()
        dims = dwidata.shape
        dwidata = np.reshape(dwidata, (np.prod(dims[0:3]), dims[3]))
        maxdwi = np.amax(dwidata, axis=1)
        dwidata = np.divide(dwidata.T, maxdwi.T).T
        return dwidata, maxdwi, dims, dwinii

    def _prepare_mask(self, mask):
        """Private method. Concatenates mask into a 2D matrix"""
        if mask is None:
            return None
        masknii = nb.load(mask)
        maskdata = masknii.get_fdata()
        dims = maskdata.shape
        maskdata = np.reshape(maskdata, (np.prod(dims[0:3]), 1))
        return maskdata

    def _remove_fw_component(self, dwidata, maxdwi, f, bvals, mask):
        """Private method. Computes the free water signal from bvals and
        substract it from the diffusion data"""

        # bp()

        meanb0 = np.mean(dwidata[:,np.where(bvals<10)],2)
        fsmooth = gaussian_filter(f, sigma=.5)

        Scsf = np.exp(-3e-3*bvals)
        Scsf = np.multiply(1-fsmooth, Scsf)
        Scsf = np.multiply(meanb0, Scsf)

        # only apply fw correction within mask
        if mask is not None:
            Scsf = Scsf * mask

        Scsf[np.isnan(Scsf)] = 0
        Scsf[Scsf < 0] = 0

        St = np.divide(dwidata - Scsf, fsmooth)
        St[:,np.where(bvals<10)] = dwidata[:,np.where(bvals<10)] # leave b=0 volumes untouched
        St = np.multiply(St.T, maxdwi.T).T
        St[np.isnan(St)] = 0
        # St[St < 0] = 0
        return (St, Scsf)
