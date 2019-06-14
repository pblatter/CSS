""" 
Computer Simulations of Sensory Systems FS2019

Exercise 1: The Auditory System
Simulation of a cochlea-implants using GammaTones

Authors: Jan Wiegner, Diego Salinas Gallegos, Philippe Blatter
Date: 29.03.2019
Version: 5
"""

# Standard packages
import os
import numpy as np
import matplotlib.pyplot as mpl

# Other imports
from scipy.io.wavfile import write
from sksound.sounds import Sound

# Local modules
import GammaTones

__author__ = 'Jan Wiegner, Diego Salinas Gallegos, Philippe Blatter'
__email__ = 'wiegnerj@student.ethz.ch, sdiego@student.ethz.ch, pblatter@student.ethz.ch'
__status__ = 'v5'


### Define used funcitons

def getParameters():
    """ Gets parameters from user (if default values are not desired)."""

    #dictionnary of all parameters
    parameters = {}

    #defalt values
    parameters['numChannels'] = [20, 'int']         # number of channels
    parameters['loFreq'] = [200, 'int']             # lower bound on frequencies
    parameters['hiFreq'] = [5000, 'int']            # upper bound on frequencies
    parameters['plotChannels'] = [False, 'boolean'] # if it should plot the Gammatone channels
    parameters['block_time'] = [10.0, 'float']          # in ms
    parameters['block_shift'] = [1.0, 'float']          # in ms
    parameters['selectChannels'] = [6, 'int']       # number of channels to activate at a single time

    # print values in console
    print('\nHere are the default values for the parameters:')
    for key, val in parameters.items():
        print('{} = {}'.format(key, val[0]))

    # ask user if he wants to change them
    response = ''
    while(response not in {'y', 'n'}):
        response = input('Do you want to change the parameters? (y/n) ')

    if(response=='n'):
        return parameters
        
    description = """
    numChannels  (int): the number of channels/frequencies/electrodes to simulate
    loFreq (int): the lower bound on the frequencies (in Hz)
    hiFreq (int): the upper bound on frequencies (in Hz)
    plotChannels (boolean): if the Gammatone channels should be plotted
    block_time (float): length of blocks of time over whch to integrate (in ms)
    block_shift (float): length of shift between blocks (in ms)
    selectChannels (int): number of channels/frequencies to activate at a single time

    """

    #ask if he wants to read description
    response = ''
    while(response not in {'y', 'n'}):
        response = input('Do you want to read the description of the parameters? (y/n) ')

    if(response == 'y'):
        print(description)

    #change the parameters
    for key, val in parameters.items():
        user_input = input('{} = '.format(key))
        if(parameters[key][1] == 'int'):
            parameters[key][0] = int(user_input)
        elif(parameters[key][1] == 'float'):
            parameters[key][0] = float(user_input)
        elif(parameters[key][1] == 'boolean'):
            parameters[key][0] = (user_input == 'True')
            
    return parameters




def filterDataGamaTone(data, sampleRate,numChannels,loFreq,hiFreq, plot=False):
    """ Uses the GammaTone toolbox to produce the output of numChannels channels,
        corresponding to numChannels linearly spaced electrode locations.

        arguments: 
            data: data stream
            sample rate: sample rate of the audio file
            numChannels: number of electrodes contained in the implant
            loFreq: lower bound on frequencies
            hiFreq: upper bound on frequencies
            plot: boolean (default=False) value to decide if the user wants to see the plot of the audio file


        returns:
            filtered: numpy array of data filtered by all freqencies in channel_fs
                        (each channel is a row) 
            channel_fs: list of frequencies of all channels
    """
        
    #create fileter parameters from the GammaTone toolbox
    method = 'moore'        #method for finding the parameters
    (forward,feedback,channel_fs,ERB,B) = GammaTones.GammaToneMake(sampleRate,numChannels,loFreq,hiFreq,method)

    #apply filter parameters
    filtered = GammaTones.GammaToneApply(data,forward,feedback)

    if plot: 
        # Show the plots
        mpl.figure(1)

        ax = mpl.subplot(121)
        # Show all frequencies, and label a selection of centre frequencies
        GammaTones.BMMplot(filtered, channel_fs, sampleRate, [0, 3, 6, 12, 15, 19])
        
        
        mpl.figure(1)
        ax = mpl.subplot(122)
        # For better visibility, plot selected center-frequencies in a second plot.
        # Dont plot the centre frequencies on the ordinate.
        GammaTones.BMMplot(filtered[[0, 3, 6, 12, 15, 19],:], channel_fs, sampleRate, '')
        
        
        mpl.show()
        mpl.close()

    return (filtered, channel_fs)



def gammatoneToAmplitude(filtered, samples_in_block, samples_in_shift):
    """ Go from Gammatone channels to amplitudes in time blocks

        Calculates the amplitude of the soundwaves for all time blocks and all frequencies
        of the channels from the Gammatones

        arguments: 
            filtered: numpy array of data filtered by Gammatones (each channel is a row)
            samples_in_block: number of samples in a time block
            samples_in_shift: number by which to shift to 

        returns:
            summed: 
    """

    nb_blocks = np.ceil(len(filtered[0]) / samples_in_shift).astype('int')
    nb_channels = len(filtered)

    summed = np.empty((nb_channels, nb_blocks)) #preallocate space

    for i in range(nb_channels):
        channel = filtered[i]
        summed[i] = [ sum(channel[x:x+samples_in_block]**2) for x in range(0, len(channel), samples_in_shift)]

    # Stimulation Intensity -> Stimulation Amplitude
    summed = np.sqrt(summed)

    return summed

def n_largest_channels(summed, n=6):
    """ Implements the n-out-of-m strategy

        Option to use the n-out-of-m strategy used in real cochlear implants: at any given time, 
        only those n (typically 6) of all m available (typically around 20) electrodes are 
        activated, which have the largest stimulation.

        arguments: 
            summed: filtered data of dimensions [n_electrodes, data_length]
            n: number of channels that should be considered -> n has to be smaller than the number of rows

        returns:
            n_out_of_m: modified data matrix only consisting of the columns with the n largest values per
                        column.

    """

    rows, columns = summed.shape

    # n has to be smaller than the number of rows
    assert n <= rows

    n_out_of_m = np.zeros(summed.shape)

    for col in range(columns):
        column = summed[:, col]
        indices = np.argsort(column)[:-n]
        tmp = column
        tmp[indices] = 0
        n_out_of_m[:, col] = tmp

    return n_out_of_m
    



def generateSound(amps_samples, channel_fs, sampleRate):
    """ Generate sound data for given amplitudes and frequencies

        arguments: 
            amps_samples:
            channel_fs:
            sampleRate: sample rate for the given amp_samples

        returns:
            res_data: the generated sound data
    """

    samples_to_gen = len(amps_samples[0]) 
    nb_channels = len(amps_samples)
    duration = samples_to_gen / sampleRate # in s

    
    t = np.linspace(0.0, duration, samples_to_gen)  #  Produces length of samples

    sines = amps_samples * np.sin(2 * np.pi * np.outer(channel_fs, t) )
    ySum = np.sum(sines, axis=0)


    # Normalize data, so that it is in playable amplitude
    res_data = 10* ySum / np.linalg.norm(ySum)

    return res_data



def main():
    """ main function that starts the entire process"""

    ### Choose and Import File

    inSound = Sound()

    rate = inSound.rate
    data = inSound.data
    dataLength = len(data)
    
    info = inSound.get_info()
    head, filename = os.path.split(info[0])  # get filename of input
    
    # Decide output directory and filename
    outDir = r'out'
    outFile = os.path.join(outDir, 'out_'+filename)

    # Check if data has multiple channels, if yes use only one
    if(len(data.shape) > 1):
        data = data[:,0]


    ### Set All Parameters

    #get parameters from user dialogue
    params = getParameters()

    numChannels = params['numChannels'][0]      # number of Channels
    loFreq = params['loFreq'][0]                # lower bound on frequencies
    hiFreq = params['hiFreq'][0]                # upper bound on frequencies
    plotChannels = params['plotChannels'][0]    # if it should plot the Gammatone channels
    block_time = params['block_time'][0]        # in ms
    block_shift = params['block_shift'][0]      # in ms
    selectChannels = params['selectChannels'][0] # number of channels to activate at a single time


    ### Filter input file

    filtered, channel_fs = filterDataGamaTone(data, rate, numChannels, loFreq, hiFreq, plotChannels)


    ### Gammatones -> Stimulation Amplitude for time block

    samples_in_block = np.floor(block_time * rate / 1000).astype('int')
    samples_in_shift = np.floor(block_shift * rate / 1000).astype('int')

    summed = gammatoneToAmplitude(filtered, samples_in_block, samples_in_shift)

    # only activate the n electrodes that have the largest stimulation
    amps = n_largest_channels(summed, n=selectChannels)

    
    #### Sound reconstruction

    # for each timeblock we need to duplicate enough samples to fill it at sample rate
    amps_samples = np.repeat(amps, samples_in_shift, axis=1)
    #trim end to get same length as input
    amps_samples = amps_samples[:,:dataLength] 

    # from amplitude samples and frequencies, reconstruct sound
    res_data = generateSound(amps_samples, channel_fs, rate)


    ### Write to output file
    write(outFile, rate, res_data)
    print('Wrote file to: \n' + outFile)


# run if this is the file executed
if __name__ == '__main__':
    main()
