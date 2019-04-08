#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Author: Snehal Vartak
# (based on skeleton code by D. Crandall, Oct 2017)
#

'''
Report--

First we need to calculate the emission, transition and initial state probabilities for the data.
The initial state probabilities for the training characters are calculated from a text training file.
Initial Probability of a Character = number of times the character is the first alphabet of a word/ total number of words

The transition probability is modeled as,
P(S_i|S_{i-1}) = number of times state i-1 is followed by state i/number of occurences of state i-1

The emission probabilities are modeled as below:
Here each observation is a group of pixels. 
    
Emission prob = P(all observed pixels | letter) = P(p1|a) P(p2|a)...P(pn|a)
and the pixel is either black or white i.e 1 or 0 

So, the Emission prob can be written as -

P(all observed pixels | letter) = (1)^(number of black pixels) *(0)^(no. of white pixels)
 
We also know that m% of the pixels are noisy, that means using naive bayes we can assume 
that the noisy pixels of the observed image will match the reference letter pixel (100-m)%
 
If we assume 10% noise, then probability of pixels matching is 0.9 which can be written as
  
P(all observed pixels| ref letter) = (0.9)^(number of matched pixels) * (0.1)^(number of mismatched pixels)

For the simplified model in Fig 1b, each observation is only dependent on its state.
Hence, s_i = argmax {si}P(Si=si|W) which is proportional to argmax{si} P(W|Si=si)

For the HMM of Figure 1(a) with variable elimination, we need to sum out all the state 
variables one by one, and then choose the states with the maximum probability for each observation.
Since I implemented the viterbi algorithm before variable elimination, Viterbi can be converted to Variable 
elimination by swapping the max with sum. (taken from source http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Handouts/recitations/HMM-inference.pdf)

For HMM of Fig 1a with MAP inference (Viterbi), we compute 
(q 1, . . . , q N) = arg max{q1,...,qN} P(Qi = qi|O) which is proportional to P(Q0=q0) *(product_{t=0 to T} P(Qt+1|Q_t)) * (product_{t=0 to T} P(Ot|Qt))

'''
from __future__ import division
from PIL import Image, ImageDraw, ImageFont
import sys
import math
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print (im.size)
    print (int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#print "\n".join([ r for r in train_letters['a'] ])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print "\n".join([ r for r in test_letters[2] ])


##### MY CODE STARTS HERE ###################
def init_trans(filename):
    '''
    This function reads the text training file and calculates the initial and transition probabilities of the states.
    
    Input Paramters-
    filename -- String containing the file name of text training file
    
    Return Values-
    init - List of initial state probabilities for the states indexed as per the TRAIN_LETTERS 
    trans- 2D array of transition probabilities
    trans[i,j] can be read as the P(Q_t = j | Q_{t-1} = i) i.e the probability of transitioning from state i to state j
    
    These are calculated as,
    state t = a and state t-1 = b then
    P(Q_t = a| Q_{t-1}=b) = number of times state b is followed by state a/ number of occurences of b
    '''
    train_txt = open("sample.txt",'r')
    lines = train_txt.readlines()

    file_temp = []
    word_temp=[]
    for line in lines:
        for word in line.split():
            word_temp.append(word)
        for a in line:
            file_temp.append(a)
            
    #List to store the number of occurences each alphabet at the start of a word
    init_temp = [0]*len(TRAIN_LETTERS)
    
    for a in range(len(TRAIN_LETTERS)):
        for w in word_temp:
            if TRAIN_LETTERS[a] == w[0]:
                init_temp[a] +=1
                
        for a in line:
            file_temp.append(a) #used below when calculating the transition probabilities

    train_txt.close()
    
    #init store the logged initial state probabilities smoothed using laplace smoothing
    #index of init is decided by TRAIN_LETTERS
    init = [(i+1)/(len(word_temp)+2) for i in init_temp] #Laplace Smoothing 
    
    #initialize a 2D array to 0 to store transition probabilities
    trans = np.zeros(shape=(len(TRAIN_LETTERS), len(TRAIN_LETTERS)))
    
    #Iterate over the training file to calculate the transition probabilities       
    for i in range(0,len(file_temp)):
        if file_temp[i] in TRAIN_LETTERS and file_temp[i+1] in TRAIN_LETTERS:
            t_index = TRAIN_LETTERS.index(file_temp[i])
            t1_index = TRAIN_LETTERS.index(file_temp[i+1])
            trans[t_index,t1_index] += 1
    #Calculate the denominator from the transition probability formula below
    #P(Q_t = a| Q_{t-1}=b) = number of times state b is followed by state a/ number of occurences of b
    rows_sum = np.sum(trans,axis=1)
    
    #Update the 2D array trans with logged transition probabilities
    for i in range(0,len(TRAIN_LETTERS)):
        for j in range(0,len(TRAIN_LETTERS)):
            trans[i,j] = (trans[i,j] + 1)/(rows_sum[i]+2) #Laplace Smoothing
       
    return init,trans   

def calc_emission_prob():
    ''' 
    This funtion calculates the emission probabilities of a observation given a state.
    Here each observation is a group of pixels. 
    
    Emission prob = P(all observed pixels | letter) = P(p1|a) P(p2|a)...P(pn|a)
    and the pixel is either black or white i.e 1 or 0 
    
    So, the Emission prob can be written as -
    
    P(all observed pixels | letter) = (1)^(number of black pixels) *(0)^(no. of white pixels)
 
    We also know that m% of the pixels are noisy, that means using naive bayes we can assume 
    that the noisy pixels of the observed image will match the reference letter pixel (100-m)%
 
    If we assume 10% noise, then probability of pixels matching is 0.9 which can be written as
 
 
    P(all observed pixels| ref letter) = (0.9)^(number of matched pixels) * (0.1)^(number of mismatched pixels)
    
    This is the basis for the calculation of emission probabilities below.
    
    Input Parameters -
    test_letters - Observed pixels of characters
    train_letters -  Actual pixels of training characters
    noise - Tune the noise parameter based to get the best possible prediction
    
    Return -
    emission - 2D numpy array, where rows denote the states and columns denote the observations
    emission[i,j] -Read as Probability that we observe j given that the state is i
    '''
    
    noise = 0.42
    num_i = len(train_letters)
    num_j = len(test_letters)
    emissions = np.zeros(shape=(num_i,num_j))
    for letter in train_letters:
        for j in range(num_j):
            val = train_letters.get(letter)
            obs = test_letters[j]
            mis = 0
            match = 0
            for m in range(25):
                for n in range(14):
                    if val[m][n] != obs[m][n]:
                        mis += 1 #the number of mismatched pixels
                    else:
                        match += 1 #the number of matched pixels
            emissions[TRAIN_LETTERS.index(letter)][j] = (math.pow(1-noise,match)) * (math.pow(noise,mis))
    
    return emissions
            
         
def simplified(test_letters,init,ems):
    '''
    This function calculates the simplified 
    '''
    init =init
    emission = ems
    obs = test_letters
    
    rows = len(TRAIN_LETTERS)
    cols = len(obs)
    y = np.zeros(cols)
    temp_results = np.zeros(shape=(rows,cols))
    
    for i in range(0,cols):
        for j in range(0,rows):
            temp_results[j,i] = emission[j,i]
    y = np.argmax(temp_results, axis=0)
    
    return y


def hmm_ve(test_letters,init,ems,trans):
    ''' This function solves 1(a) using variable elimination'''
    init =init
    emission = ems
    transition = trans
    obs = test_letters
    rows = len(TRAIN_LETTERS)
    cols = len(obs)
    out = np.zeros(shape=(rows,cols))
    
    # for t = 0
    for i in range(rows):
        out[i,0] = init[i] * emission[i,0]
    # replace the max in viterbi to sum
    for i in range(cols):
        for j in range(rows):
            temp =0
            for k in range(rows):
                temp += init[k]*transition[k,i]
            out[j,i] = temp* emission[j,i]
    
    #Get the indexes of the max probabilities from each column of out
    indexes = np.argmax(out,axis=0)
    
    return indexes


def viterbi(test_letters,init,trans,ems):
    '''
    This function performs the viterbi decoding to calculate the most likely path.
    
    Input Parameters -
    obs - the observed data from test_letters
    init - initial state probabilities
    trans - state transition probabilities
    ems - emission probabilities
    
    Return Value-
    most_likely - List of Most likely States for observed data
    '''
    #Set the input variables
    obs = test_letters
    init = init
    transition = trans
    emission = ems
    
    #get the length of observation data and the length of training images to create a 2D array v_i(t) for the viterbi calculations
    t = len(obs)
    rows = len(train_letters)
    
    #Initialize a 2D array v_it with (rows,t) to store the maximum probabilities for each (State,Observation)
    v_it = np.zeros(shape=(rows,t))
    
    #Store the index of the best states at (row,t)
    max_path =  np.empty(shape=(rows,t),dtype=int)
    
    #At t=0, v_it is calculated as prior(i)*emission(t|i). The for loop will iteratively calculate v_it for t=0 at each state i
    #Since all the probabilities are logged, we calculate v_it at t=0 as logged_prior(i) + logged_emission(t|i)
    for i in range(rows):
        v_it[i,0] = math.log(init[i]) + math.log(emission[i,0])
    
    #For j=1 to t, recursively calculate the probability of the most probable path ending at state i at time t+1
    for j in range(1,t):
        for i in range(rows):
            temp_val = [] #Store the intermidiate probabilities coming in from each state at each j
            for k in range(rows):
                temp_val.append(v_it[k,j-1] + math.log(transition[k,i] ) +math.log(emission[i,j]))
            #Get the max probability for Observation j    
            max_state = max(temp_val)
            v_it[i,j] = max_state #Store it in our 2D array
            
            #Get the index for the state that generates the max probability for Observation j and store it to backtrack the most likely path
            max_path[i,j] = temp_val.index(max_state)
    
    #Create a 1D array to store the indexes to TRAIN_LETTERS for the most likely path
    most_likely = np.zeros(t,dtype=int)
    
    #Get the state with the maximum probability at the last observation and backtrack from there
    most_likely[-1] = np.argmax(v_it[:,-1])
    
    for i in range(1,t)[::-1]: # iterate in reverse order
        most_likely[i-1] = max_path[most_likely[i],i]
    return most_likely

###### main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
init, trans = init_trans(train_txt_fname)
ems = calc_emission_prob()
simple = simplified(test_letters,init,ems)
print ("Simple test: "+"".join([TRAIN_LETTERS[i] for i in simple]))
output = viterbi(test_letters,init,trans,ems)
print ("Final HMM vetebri: " + "".join([TRAIN_LETTERS[i] for i in output]))
