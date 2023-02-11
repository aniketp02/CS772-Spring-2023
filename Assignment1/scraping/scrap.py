import requests
from bs4 import BeautifulSoup
import os
import nltk
from nltk.corpus import gutenberg



# Hyperparameter for number of sentences
K = 150



def remove_bracket_substrings(text):			#Removes substring within brackets along with brackets from the given string
	bracket = 0
	bracketless_text = ""
	for char in text:
		if(char == '(' or char == '[' or char == '{'):
			bracket += 1
			continue
		if(bracket == 0):
			bracketless_text += char
			continue
		if(char == ')' or char == ']' or char == '}'):
			bracket -= 1
	return bracketless_text



def main():
	# Download NLTK's Gutenberg Corpus in nltk_data folder in home directory
	print()
	nltk.download("gutenberg")					#https://www.nltk.org/api/nltk.downloader.html



	# Show the books included in the corpus
	books = gutenberg.fileids()					#https://www.nltk.org/book/ch02.html
	print("\nThe following books are present in the corpus:\n", books, sep="")

	# Create a data string containing all the words in the corpus
	data = " ".join(gutenberg.words())			#https://www.nltk.org/api/nltk.corpus.html



	# Read Analogy_dataset.txt
	analogy_file = None
	if(os.name == "posix"):
		analogy_file = open("../Dataset/Analogy_dataset.txt")
	elif(os.name == "nt"):
		analogy_file = open(r"..\Dataset\Analogy_dataset.txt")

	analogy_string = ""
	for i in analogy_file:
		analogy_string += i
	analogy_string = analogy_string.strip()								#Removes the leading and trailing whitespace
	# print(analogy_string)
	analogy_words = analogy_string.split()								#Forms a list from the string by separating elements at comma and newline
	# print(analogy_words)

	# Create a list of analogy word pairs
	analogy_word_pairs = []
	for i in range(len(analogy_words)//2):
		if([analogy_words[2*i], analogy_words[2*i+1]] not in analogy_word_pairs):
			analogy_word_pairs.append([analogy_words[2*i], analogy_words[2*i+1]])
	print("\nThe following words are present in the analogy dataset:\n", analogy_word_pairs, sep="")



	# Scrap data from web using BeautifulSoup
	scrap = ""
	for pair in analogy_word_pairs:										#https://www.geeksforgeeks.org/implementing-web-scraping-python-beautiful-soup/
		url0 = f"https://en.wikipedia.org/wiki/{pair[0]}"
		url1 = f"https://en.wikipedia.org/wiki/{pair[1]}"
		r0 = requests.get(url0)											#Stores the HTML content from url0 into r0
		r1 = requests.get(url1)
		# print(r0.content)												#Raw HTML content (string type) printed
		# print(r1.content)
		soup0 = BeautifulSoup(r0.content, "html5lib")					#BeautifulSoup object created with html5lib as the parser
		soup1 = BeautifulSoup(r1.content, "html5lib")
		# print(soup0.prettify())										#Parse tree gets printed
		# print(soup1.prettify())

		# Useful text can be found in the <p> elements of the html content of the above URLs
		p0 = soup0.findAll('p')											#https://www.crummy.com/software/BeautifulSoup/bs4/doc/
		p1 = soup1.findAll('p')

		n_lines = 0
		lines = ""
		# l = []
		for p in p0:
			# First we need to remove all the substrings within brackets (including the brackets) as they might cause issues later
			p_text = remove_bracket_substrings(p.text)
			
			para = p_text.split('. ')
			if('.' in para[-1]):										#The para contains text if the last element (line) contains ".\n"
				for line in para:
					if(line != para[-1]):
						line += '. '
					else:
						line = line[:-1] +' '
					# l.append(line)
					if(line in lines):
						continue
					lines += line
					n_lines += 1
					if(n_lines == K-10):
						if(pair[1] not in lines):
							p01 = soup0.findAll('p', string=pair[1])
							if(len(p01) != 0):
								for ap in p01:
									ap_text = remove_bracket_substrings(ap.text)

									apara = ap_text.split('. ')
									if('.' in apara[-1]):
										for line in apara:
											if(line != apara[-1]):
												line += '.'
											else:
												line = line[:-1] + ' '
											# l.append(line)
											lines += line
											n_lines += 1
											if(n_lines == K):
												break
									if(n_lines == K):
										break
							else:
								continue
					if(n_lines == K):
						break
				if(n_lines == K):
					break
		# print(len(l))
		scrap += lines

		n_lines = 0
		lines = ""
		for p in p1:
			p_text = remove_bracket_substrings(p.text)

			para = p_text.split('. ')
			if('.' in para[-1]):
				for line in para:
					if(line != para[-1]):
						line += '. '
					else:
						line = line[:-1] + ' '
					if(line in lines):
						continue
					lines += line
					n_lines += 1
					if(n_lines == K-10):
						if(pair[0] not in lines):
							p10 = soup1.findAll('p', string=pair[0])
							if(len(p10) != 0):
								for ap in p10:
									ap_text = remove_bracket_substrings(ap.text)

									apara = ap_text.split('. ')
									if('.' in apara[-1]):
										for line in apara:
											if(line != apara[-1]):
												line += '.'
											else:
												line = line[:-1] + ' '
											lines += line
											n_lines += 1
											if(n_lines == K):
												break
									if(n_lines == K):
										break
							else:
								continue
					if(n_lines == K):
						break
				if(n_lines == K):
					break
		scrap += lines
		# break



	# # Analogy_dataset contains only data related to countries, cities, continents, states, and currencies.
	# # Hence, some more data has to be collected which should be related to analogy tasks.
	senses = ["Eye", "Ear", "Nose", "Tongue", "Skin"]
	animals = [["Lion", "Lioness", "Cub"], ["Dog", "Bitch", "Puppy"], ["Cat", "Cat", "Kitten"], ["Cow", "Bull", "Calf_(animal)"], ["Cock", "Hen", "Chick"], ["Fish", "Fish", "Fry_(fish)"], ["Fox", "Vixen", "Cub"], ["Tiger", "Tigress", "Cub"], ["Horse", "Mare", "Foal"]]

	for organ in senses:
		url = f"https://en.wikipedia.org/wiki/{organ}"
		r = requests.get(url)
		soup = BeautifulSoup(r.content, "html5lib")
		p_element = soup.findAll('p')

		n_lines = 0
		lines = ""
		for p in p_element:
			p_text = remove_bracket_substrings(p.text)
			para = p_text.split('. ')
			if('.' in para[-1]):
				for line in para:
					if(line != para[-1]):
						line += '. '
					else:
						line = line[:-1] +' '
					if(line in lines):
						continue
					lines += line
					n_lines += 1
					if(n_lines == K):
						break
			if(n_lines == K):
				break
		scrap += lines

	for family in animals:
		url = f"https://en.wikipedia.org/wiki/{family[0]}"
		r = requests.get(url)
		soup = BeautifulSoup(r.content, "html5lib")
		p_element = soup.findAll('p')

		n_lines = 0
		lines = ""
		for p in p_element:
			p_text = remove_bracket_substrings(p.text)
			para = p_text.split('. ')
			if('.' in para[-1]):
				for line in para:
					if(line != para[-1]):
						line += '. '
					else:
						line = line[:-1] + ' '
					if(line in lines):
						continue
					lines += line
					n_lines += 1
					if(n_lines == K-20):
						f01 = soup.findAll('p', string=family[1])
						if(len(f01) != 0):
							for ap in f01:
								ap_text = remove_bracket_substrings(ap.text)
								apara = ap_text.split('. ')
								if('.' in apara[-1]):
									for line in apara:
										if(line != apara[-1]):
											line += '.'
										else:
											line = line[:-1] + ' '
										lines += line
										n_lines += 1
										if(n_lines == K-10):
											break
								if(n_lines == K-10):
									break
					if(n_lines == K-10):
						f02 = soup.findAll('p', string=family[2])
						if(len(f02) != 0):
							for ap in f02:
								ap_text = remove_bracket_substrings(ap.text)
								apara = ap_text.split('. ')
								if('.' in apara[-1]):
									for line in apara:
										if(line != apara[-1]):
											line += '.'
										else:
											line = line[:-1] + ' '
										lines += line
										n_lines += 1
										if(n_lines == K):
											break
								if(n_lines == K):
									break
					if(n_lines == K):
						break
				if(n_lines == K):
					break
		scrap += lines



	# print()
	# print(data)
	# print()
	# print(scrap)
	# print()

	final_data = data + scrap
	final_data = final_data.replace('\n', ' ')



	# Writing final_data to a txt file
	if(os.name == "posix"):
		write_file = open("../Data/final_data.txt", 'w')
	elif(os.name == "nt"):
		write_file = open(r"..\Data\final_data.txt", 'w')
	write_proc = write_file.write(final_data)
	write_file.close()
	# # print(write_proc)



if __name__ == "__main__":
	main()



# References:
# https://www.nltk.org/api/nltk.downloader.html
# https://www.nltk.org/book/ch02.html
# https://www.nltk.org/api/nltk.corpus.html
# https://www.geeksforgeeks.org/implementing-web-scraping-python-beautiful-soup/
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/
# https://en.wikipedia.org/wiki/
