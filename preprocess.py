from tqdm import tqdm #Visualizing implementation of loops
import io

#Pre-trained dataset
path = 'glove.840B.300d.txt'

#Reading pre-trained dataset
with io.open(path, encoding="utf8") as f:
	lines = [line for line in f]


updated = []

#Cleaning the garbage values
for line in tqdm(lines, total=len(lines)):
	entries = line.rstrip().split(" ")
	if entries[0].isalpha():
		updated.append(line)

updated = ''.join(updated)

#New file in which processed data will be stored
new_file = open('glove.840B1.300d.txt','w')

#writing preprocessed data in a new file
new_file.write(str(new2))
print("Content written")



