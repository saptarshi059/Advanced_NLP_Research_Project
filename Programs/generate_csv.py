#python3 generate_csv.py -trf Desktop/Advanced_NLP_Research_Project/corpus/train/en -tsf Desktop/Advanced_NLP_Research_Project/corpus/test/en -tsl Desktop/Advanced_NLP_Research_Project/corpus/test/truth-en.txt 

import os
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import re
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract XML data and organize into CSV files.')

parser.add_argument('--train_folder','-trf', type=str, required = True, help='Path to training files.')
parser.add_argument('--test_folder', '-tsf', type=str, required = True, help='Path to test files.')
parser.add_argument('--test_labels','-tsl', type=str, required = True, help='Path to truth labels of test files.')

args = parser.parse_args()

train_path = Path.home().joinpath(args.train_folder)
test_path = Path.home().joinpath(args.test_folder)
test_labels = Path.home().joinpath(args.test_labels)

train_df = pd.DataFrame()
test_df = pd.DataFrame()

def find_category(gender,age_group):
	if gender == 'male' and age_group == '10s':
		return 0
	elif gender == 'male' and age_group == '20s':
		return 1
	elif gender == 'male' and age_group == '30s':
		return 2
	elif gender == 'female' and age_group == '10s':
		return 3
	elif gender == 'female' and age_group == '20s':
		return 4
	elif gender == 'female' and age_group == '30s':
		return 5

def clean_text(text):
	text = re.sub('<.*?>', ' ', text) #Removing HTML tags.
	text = re.sub('\n', ' ', text) #Removing new lines.
	text = re.sub('\s+', ' ', text) #Removing extra spaces.
	text = re.sub('[\W]+', ' ', text) #Removing non-word characters.
	return text

def extract_train_xml():
	with os.scandir(train_path) as entries:
		conversation_list = []
		category_list = []
		count = 0
		for xml_file in tqdm(entries):
			dom = ET.parse(xml_file)

			author_details = dom.getroot().attrib
			category = find_category(author_details['gender'], author_details['age_group'])
			
			conversations = dom.findall('conversations/conversation')
			for conversation in conversations:
				if conversation.text != None:
					conversation_list.append(clean_text(conversation.text))
					category_list.append(category)

		train_df['conversation'] = conversation_list
		train_df['category'] = category_list

	print('Saving Training File...')
	train_df.to_csv('train_custom.csv')

def extract_test_xml():
	label_lines = open(test_labels,'r').readlines()
	with os.scandir(test_path) as entries:
		conversation_list = []
		category_list = []
		count = 0
		for xml_file in tqdm(entries):
			file_name = re.findall('(.*?)_',xml_file.name)[0]
			dom = ET.parse(xml_file)
			
			count = 0
			conversations = dom.findall('conversations/conversation')
			for conversation in conversations:
				if conversation.text != None:
					conversation_list.append(clean_text(conversation.text))
					count += 1

			for line in label_lines:
				if file_name in line:
					category = find_category(re.findall('male|female',line)[0],re.findall('10s|20s|30s',line)[0])
					break

			for i in range(count):
				category_list.append(category)

		test_df['conversation'] = conversation_list
		test_df['category'] = category_list
	
	print('Saving Test File...')
	test_df.to_csv('test_custom.csv')

extract_train_xml()
extract_test_xml()