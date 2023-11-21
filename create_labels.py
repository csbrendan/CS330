import re

def create_labels(input_filename, output_filename):
    cauc = r"\b(american|caucasian|white|anglo|european)s?\b"
    afro = r"\b(african|afro|afro-american|black)s?\b"
    asian = r"\b(asian|chinese|japanese)s?\b"
    latin = r"\b(latino|spanish|mexican|hispanic)s?\b"
    #all = r"\b(african|afro|arab|american|asian|caucasian|latino|black|white|spanish|mexican|anglo|hispanic|european|chinese|native|japanese|jewish|jew)s?\b"
    
    male = r"\b(he|him|his|man|men|boy|boys)\b"
    female = r"\b(woman|women|girl|girls|she|her|hers)\b"

    # Dictionary to keep track of counts
    subgroup_counts = {'male': 0, 'female': 0, 'caucasian': 0, 'afro': 0, 'asian': 0, 'latin': 0, 'neither': 0}

    with open(input_filename, "r") as infile, open(output_filename, "w") as outfile:
        for line in infile:
            labels = []
            if re.search(male, line, flags=re.IGNORECASE):
                labels.append('1') 
                subgroup_counts['male'] += 1
            if re.search(female, line, flags=re.IGNORECASE):
                labels.append('2') 
                subgroup_counts['female'] += 1
            if re.search(cauc, line, flags=re.IGNORECASE):
                labels.append('3') 
                subgroup_counts['caucasian'] += 1
            if re.search(afro, line, flags=re.IGNORECASE):
                labels.append('4') 
                subgroup_counts['afro'] += 1
            if re.search(asian, line, flags=re.IGNORECASE):
                labels.append('5')
                subgroup_counts['asian'] += 1
            if re.search(latin, line, flags=re.IGNORECASE):
                labels.append('6')
                subgroup_counts['latin'] += 1                                                
            if not labels:
                labels.append('0')  # Neither
                subgroup_counts['neither'] += 1

            outfile.write(','.join(labels) + '\n')

    return subgroup_counts

if __name__ == "__main__":
    input_file = '/Users/brendanmurphy/Desktop/CS330 META/PROJECT/project_code/train.txt'
    output_file = '/Users/brendanmurphy/Desktop/CS330 META/PROJECT/project_code/train_labels.txt'
    counts = create_labels(input_file, output_file)
    print("Subgroup Counts:", counts)

#Subgroup Counts: {'male': 1597, 'female': 1127, 'caucasian': 456, 'afro': 254, 'asian': 51, 'latin': 60, 'neither': 672}