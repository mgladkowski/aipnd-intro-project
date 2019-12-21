#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_results.py
#                                                                             
# PROGRAMMER: Michael Gladkowski
# DATE CREATED: 2019-11-30
# REVISED DATE: 2019-12-03
# PURPOSE: Create a function print_results that prints the results statistics
#          from the results statistics dictionary (results_stats_dic). It 
#          should also allow the user to be able to print out cases of misclassified
#          dogs and cases of misclassified breeds of dog using the Results 
#          dictionary (results_dic).  
#         This function inputs:
#            -The results dictionary as results_dic within print_results 
#             function and results for the function call within main.
#            -The results statistics dictionary as results_stats_dic within 
#             print_results function and results_stats for the function call within main.
#            -The CNN model architecture as model wihtin print_results function
#             and in_arg.arch for the function call within main. 
#            -Prints Incorrectly Classified Dogs as print_incorrect_dogs within
#             print_results function and set as either boolean value True or 
#             False in the function call within main (defaults to False)
#            -Prints Incorrectly Classified Breeds as print_incorrect_breed within
#             print_results function and set as either boolean value True or 
#             False in the function call within main (defaults to False)
#         This function does not output anything other than printing a summary
#         of the final results.
##
# TODO 6: Define print_results function below, specifically replace the None
#       below by the function definition of the print_results function. 
#       Notice that this function doesn't to return anything because it  
#       prints a summary of the results using results_dic and results_stats_dic
# 

import datetime
import os


def print_results(results_dic, results_stats_dic, model, 
                  print_incorrect_dogs = False, print_incorrect_breed = False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats_dic - Dictionary that contains the results statistics (either
                   a  percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - Indicates which CNN model architecture will be used by the 
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """    
    
    # summary of results
    
    print("\n----------------------------------------------")
    print("Results for CNN architecture : {}".format(model.upper()))
    print("")
    print("Number of images         : {:,d}".format(results_stats_dic['n_images']))
    print("Number of dog images     : {:,d}".format(results_stats_dic['n_dogs_img']))
    print("Number of non-dog images : {:,d}".format(results_stats_dic['n_notdogs_img']))
    print("")
    print("Percent dogs correct     : {:6.2f}".format(results_stats_dic['pct_correct_dogs']))
    print("Percent breeds correct   : {:6.2f}".format(results_stats_dic['pct_correct_breed']))
    print("Percent not dogs correct : {:6.2f}".format(results_stats_dic['pct_correct_notdogs']))
    print("Percent matches correct  : {:6.2f}".format(results_stats_dic['pct_match']))

    # tabular layout for details
    
    layout = "{:<28} {:<20} {:<28}"

    # display incorrectly classified as dog or not dog

    if print_incorrect_dogs:
        
        print("\n\nThe following images were incorrectly classified as 'dog' or 'not dog':\n")
        print(layout.format("Filename", "Pet label", "Classifier label"))
        print(layout.format("--------", "-----------", "----------------"))
        
        # iterate results where label and classifier disagree is dog or not
        
        for key, value in filter(lambda item: sum(item[1][3:])==1, results_dic.items()):
            print(layout.format(key, value[0], value[1][:30]))   # filename, image label, classifier label

            
        
    # display incorrectly classified dog breeds

    if print_incorrect_breed:
        
        print("\n\nThe following dogs have incorrectly classified breeds:\n")
        print(layout.format("Filename", "Pet label", "Classifier label"))
        print(layout.format("--------", "-----------", "----------------"))

        # iterate results where label and classifier agree is dog but labels mismatch
        
        for key, value in filter(lambda item: item[1][2]==0 and sum(item[1][3:])==2, results_dic.items()):
            print(layout.format(key, value[0], value[1][:30]))   # filename, image label, classifier label



            
def save_to_final_results(results_stats, tot_time, model, folder):
    """
    Appends current statistics to a file
    
    Parameters:
      results_stats - Dictionary that contains the results statistics (either
                      a  percentage or a count) where the key is the statistic's 
                      name (starting with 'pct' for percentage or 'n' for count)
                      and the value is the statistic's value 
      tot_time      - Elapsed runtime in seconds
      model         - Indicates which CNN model architecture was used
                      values must be either: resnet alexnet vgg (string)
      folder        - The images folder processed by this run
      
    Returns:
           None     - saves raw data to the following filename : model_folder.dat
                    - aggregates summary data into filename    : final_folder.txt
    """    

    # data will be saved to these files
    
    filename = 'final_' + folder.replace('_','-').replace('/','') + '.txt'

    # ensure consistent tabular formatting
    
    layout_t = "{:<20}{:<10}\n"
    layout_h = "{:>30}{:>10}{:>10}{:>10}{:>10}{:>8}\n"
    layout_r = "{:<18}{:>12}{:>10.1f}{:>10.1f}{:>10.1f}{:>10.1f}{:>8}\n"
    
    
    # create the final summary file if needed
   
    if not os.path.exists(filename):
        
        # create the header
        
        with open(filename, 'w') as f:
            
            f.write("Use a Pre-trained Image Classifier to Identify Dog Breeds\n\n")
            f.write("Final Results Table\n")
            f.write("\n\n")
            f.write(layout_t.format("Folder", folder))
            f.write(layout_t.format("Total images", results_stats['n_images']))
            f.write(layout_t.format("Dog images", results_stats['n_dogs_img']))
            f.write(layout_t.format("Not-dog images", results_stats['n_notdogs_img']))
            f.write("\n\n")
            f.write(layout_h.format("", "% not-dog", "% dogs", "% breeds", "% match", "time"))
            f.write(layout_h.format("CNN Model Architecture", "correct", "correct", "correct", "labels", "elapsed"))
            f.write("\n")
            
    with open(filename, 'a') as f:
        
        f.write(layout_r.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                model.title(),
                                results_stats['pct_correct_notdogs'],
                                results_stats['pct_correct_dogs'],
                                results_stats['pct_correct_breed'],
                                results_stats['pct_match'],
                                str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"+str(int((tot_time%3600)%60)) 
                               ))
        
