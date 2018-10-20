#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 20:26:33 2018

@author: michel
"""

import random
import matplotlib.pyplot as plt
import numpy as np
   

"""
Problem description:
    
    A man has the flu and needs to take pills to get better.
    
    There are 2 types of pills in his medecine cabinet. 
    pill-a: which is just vitamines and has no effect on the flu virus.
    pill-b: which contains an agent killing the virus and needs to be taken
            for 4 consecutive days in order to defeat the virus.
    
    Unfortunately all the pills have been mixed together in the same box 
    and there is no way to distinguish between them.
    
    The man decides to randomly take a pill each day and wait until he gets better.
    
    We want to know how long it will take the man to be virus free.
    We want to repeat the simulation for 1000 people and figure out:
        - The mean value of the recovery time
        - The standard variation of the recovery time
    
"""

def generate_box_of_pills(nb_pill_in_box):
    pills = ['pill-a', "pill-b"]
    box = []
    for i in range(nb_pill_in_box):
        box.append(random.choice(pills))
    return box


def run_experiment(box, goal):
    pill_b_consecutive_counter = 0
    total_nb_pills = len(box)
    for i in range(total_nb_pills):
        pill_taken = box.pop()
        if (pill_taken =='pill-a'):
            pill_b_consecutive_counter = 0
        elif (pill_taken =='pill-b'):
            pill_b_consecutive_counter += 1
            if pill_b_consecutive_counter == goal:
                return i
        else:
            print("Hum, don't know about that pill. Skippint it!")
            
    return None
    

def run_test_campaign(nb_patients):        
    test_campaign_results = []
    for i in range(nb_patients):
        patient_box = generate_box_of_pills(200)    
        days = run_experiment(patient_box, nb_consecutive_days_for_pill_b_success)        
        test_campaign_results.append(days)
        print("Patient: ", i, " - Nb of days taken to be curred:", days)

    return test_campaign_results


def find_mean(campain_results):
    total_days=0
    for i in range(len(campain_results)):
        if campain_results[i-1] != None:
            total_days += campain_results[i-1]
    return int(total_days/len(campain_results))


def remove_uncurred_people_from_campain_results(campain_results):
    currated_results = []
    for i in range(len(campain_results)):
        if campain_results[i] != None:
            #print("campain_result[", i, "]=", campain_results[i])
            currated_results.append(campain_results[i])
    return currated_results
    

def find_nb_uncurred_people(campain_results):
    nb_uncurred = 0
    for i in range(len(campain_results)):
        if campain_results[i-1] == None:
            nb_uncurred += 1
    return nb_uncurred


def find_standard_deviation(campaign_results):
    data = np.array(campaign_results)
    std = data.std()
    return std 


def find_min_max(campaign_results):
    data = np.array(campaign_results)
    min = data.min()
    max = data.max()
    return (min, max)


def summarize(data, nb_pills_in_box, nb_patients, max_bins):
    summarized_data = []
    for i in range(max_bins):
        counter = 0
        for j in range(nb_patients):
            if data[j]==i:
                counter += 1
        summarized_data.append(counter)
    return summarized_data

    
# Important parameters
nb_consecutive_days_for_pill_b_success = 2
nb_patients = 10000
nb_pills_in_box = 200

# Test the code for 1 patient
box = generate_box_of_pills(nb_pills_in_box)
recovery_time = run_experiment(box, nb_consecutive_days_for_pill_b_success)
print("box:", box)
print("recovery time: ", recovery_time)

# Run test campaign
campaign_results = run_test_campaign(nb_patients)
print("campaign results:", campaign_results)
print("uncurred people:", find_nb_uncurred_people(campaign_results))

currated_results = remove_uncurred_people_from_campain_results(campaign_results)
min, max = find_min_max(currated_results)
print("currated results:", currated_results)
print("mean nb of days to be cured:", find_mean(currated_results))
print("min days:{}, max days:{}".format(min, max))

print("std:", find_standard_deviation(currated_results))

num_bins = 21
plt.hist(currated_results, num_bins, facecolor='red')

plt.title("Days to recover")
plt.xlabel("Patient number (number)")
plt.ylabel("Days to recover (number)")
plt.show()

print()
summarized_data = summarize(currated_results, nb_pills_in_box, nb_patients, 20)
print("summarized data:", summarized_data)

#plt.hist(summarized_data)
plt.hist(np.array(summarized_data), facecolor='blue')
plt.title("Summarized data. \n Shows how many patients had similar number of days to recover")
plt.xlabel("Days to recover (number)")
plt.ylabel("Number of patients (number)")
plt.show()