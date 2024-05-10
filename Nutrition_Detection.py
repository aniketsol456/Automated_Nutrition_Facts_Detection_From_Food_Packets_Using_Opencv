##To kill previous work we are done in colab

import os
os.kill(os.getpid(), 9)

##Installing and Importing Libraries and Load an Image"""

##Installing the libraries
!pip install pytesseract
!pip install Levenshtein
!sudo apt install tesseract-ocr

##Importing the libraries
import re
import cv2
import csv
import pytesseract
import Levenshtein
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

"""##Wrapping Effected Image Solution"""

# Load the image
image_path = '/content/Farari_chevdo.jpg'
image = cv2.imread(image_path)
assert image is not None, "File could not be read, check the path."

# Display the image with matplotlib to pick coordinates
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Select Points')
plt.show()

# Define manually observed source points (update these points based on your actual observations)
pts1 = np.float32([
    [50, 50],   # top-left corner
    [830, 50],  # top-right corner
    [710, 870], # bottom-right corner
    [75, 860]   # bottom-left corner
])

# Define destination points - creating a rectangle
pts2 = np.float32([
    [45, 45],
    [900, 0],
    [900, 900],
    [0, 900]
])

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(pts1, pts2)

# Perform the transformation
transformed_image = cv2.warpPerspective(image, M, (900, 900))

# Show the original and transformed images
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)), plt.title('Corrected Perspective')
plt.show()

##Load an Image

Image = cv2.imread("/content/balaji.jpg")
Image.shape

##Pre-processing Steps for an Further Process

##If needed thenn process for Image resizing process
Image2 = cv2.resize(Image,(0,0),fx=0.6,fy=0.6)
cv2_imshow(Image2)
Image2.shape

##Convert Colour to Grayscale Image
Gray_scale_image =  cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)

##Performing OTSU thresholding
Rect , Threshold1 = cv2.threshold(Gray_scale_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

##Specify a Structure or shape of Kernal
Rect_Kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(18,18))

##Applying Dilation on An Image
Dilation = cv2.dilate(Threshold1,Rect_Kernal,iterations=30)

##Finding contours which involved into an image
Contours,Heirarchy = cv2.findContours(Dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

##Making a copy of Original Image
Image2 = Image.copy()

##All images showing
cv2_imshow(Image)
cv2_imshow(Gray_scale_image)
cv2_imshow(Image2)

# A text file is created and flushed
file = open("recognized.txt", "w+")
file.write("")
file.close()

##Extracting a Nutrition from images

##Making a Function for looping for identified contours and after that extract the text from particular contours with the help
##of the pytessract library
for cnt in Contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # Drawing a rectangle on copied image
    rect = cv2.rectangle(Image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Cropping the text block for giving input to OCR
    cropped = Image2[y:y + h, x:x + w]
    # Open the file in append mode
    file = open("recognized.txt", "a")
    # Apply OCR on the cropped image
    # custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(cropped)
    # Appending the text into file
    file.write(text)
    file.write("\n")
    # Close the file
    file.close

##By Levenshtein Distance Correcting a word which are not Detecting properly

##Now,For Perfecting The word which are detecting a worng
correct_word_list = ["nutritional information*","per","daily","value","rda","energy","kcal","protein","carbrohydrates",
                     "total sugars","added sugars","total fat","saturated fat","trans fat","cholestrol","sodium"]

##Defining a function for a Finding the closet word from detected recognize.txt file and swap them with correct word.
# Read recognized.txt file
with open("recognized.txt", "r") as file:
    recognized_text = file.readlines()

# Define a function to find the closest word from the list of correct words
def find_closest_word(word):
    min_distance = float('inf')
    closest_word = ""
    for correct_word in correct_word_list:
        distance = Levenshtein.distance(word, correct_word)
        if distance < min_distance:
            min_distance = distance
            closest_word = correct_word
    return closest_word

# Define a threshold for Levenshtein distance
threshold = 2

# Open final.txt file for writing
with open("final.txt", "w") as file:
    # Iterate through each line in recognized text
    for line in recognized_text:
        # Split the line into words
        words = line.split()
        # Iterate through each word in the line
        for word in words:
            # Check if the word is in the correct_word list
            if word.lower() in correct_word_list:
                file.write(word.lower() + " ")
            else:
                # Find the closest word from the correct_word list
                closest = find_closest_word(word.lower())
                # If the Levenshtein distance is below threshold, replace the word
                if Levenshtein.distance(word.lower(), closest) <= threshold:
                    file.write(closest + " ")
                else:
                    file.write(word.lower() + " ")
        file.write("\n")

##Doing a Mapping detected value with a actual value

nutrient_thresholds = {
    "energy": 2000,  # kcal
    "total fat": 70,  # g
    "saturated fat": 20,  # g
    "cholesterol": 300,  # mg
    "sodium": 2300,  # mg
    "carbohydrates": 300,  # g
    "fiber": 25,  # g
    "sugars": 50,  # g
    "protein": 50,  # g
}

def parse_nutrient_values(recognized_lines):
    nutrient_data = {}
    for line in recognized_lines:
        line = line.lower()
        for nutrient in nutrient_thresholds:
            if nutrient in line:
                match = re.search(rf"{nutrient}.*?(\d+)", line)
                if match:
                    nutrient_data[nutrient] = int(match.group(1))
    return nutrient_data

def evaluate_healthiness(nutrient_data):
    evaluations = []
    for nutrient, value in nutrient_data.items():
        if value > nutrient_thresholds[nutrient]:
            evaluations.append(f"{nutrient} exceeds healthy limit.")
        else:
            evaluations.append(f"{nutrient} within healthy limit.")
    return evaluations

def find_closest_word(word, correct_words):
    """
    Finds the closest match to a given word within a list of correct words.

    Args:
        word: The word to find the closest match for.
        correct_words: A list of correct words.

    Returns:
        The closest match to the given word.
    """

    closest_word = None
    closest_distance = float('inf')

    # Iterate through the list of correct words
    for correct_word in correct_words:
        # Calculate the Levenshtein distance between the given word and the current correct word
        distance = Levenshtein.distance(word, correct_word)

        # If the distance is smaller than the current closest distance, update the closest word and distance
        if distance < closest_distance:
            closest_word = correct_word
            closest_distance = distance

    # Return the closest word
    return closest_word

with open("report.txt", "w") as file:
    # Writing corrected text
    for line in recognized_text:
        words = line.split()
        for word in words:
            closest = find_closest_word(word.lower(), correct_word_list)
            if Levenshtein.distance(word.lower(), closest) <= threshold:
                file.write(closest + " ")
            else:
                file.write(word.lower() + " ")
        file.write("\n")

    # Parsing nutrient data
    nutrient_data = parse_nutrient_values(recognized_text)

    # Writing health evaluation
    evaluations = evaluate_healthiness(nutrient_data)
    file.write("\nNutrition Evaluation:\n")
    for eval in evaluations:
        file.write(eval + "\n")

