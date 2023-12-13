import re

def create_labels(input_filename, output_filename):
    male_pattern = r"\b(he|him|his|man|men|boy|boys)\b"
    female_pattern = r"\b(woman|women|girl|girls|she|her|hers)\b"

    # Dictionary to keep track of counts
    subgroup_counts = {'male': 0, 'female': 0, 'both': 0}

    with open(input_filename, "r") as infile, open(output_filename, "w") as outfile:
        for line in infile:
            is_male = re.search(male_pattern, line, flags=re.IGNORECASE)
            is_female = re.search(female_pattern, line, flags=re.IGNORECASE)

            if is_male and is_female:
                label = '0'
                subgroup_counts['both'] += 1
            elif is_male:
                label = '1'
                subgroup_counts['male'] += 1
            elif is_female:
                label = '2'
                subgroup_counts['female'] += 1
            else:
                label = ''

            outfile.write(label + '\n')

    return subgroup_counts

if __name__ == "__main__":
    input_file = '/Users/brendanmurphy/Desktop/CS330 META/PROJECT/project_code/train_gender.txt'
    output_file = '/Users/brendanmurphy/Desktop/CS330 META/PROJECT/project_code/train_gender_labels.txt'
    counts = create_labels(input_file, output_file)
    print("Subgroup Counts:", counts)

#Subgroup Counts: {'male': 798, 'female': 790]
