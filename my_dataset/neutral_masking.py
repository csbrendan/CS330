import re

def neutralize_text(text):

    race_terms = r"\b(african|afro|arab|american|asian|caucasian|latino|black|white|spanish|mexican|anglo|hispanic|european|chinese|native|japanese|jewish|jew|native american)s?\b"
    #race_terms = r"\b(american|asian|caucasian|latino|black|white|spanish|mexican|anglo|hispanic|european|chinese|native|japanese|jewish|native)\b"
    text = re.sub(race_terms, "PERSON", text, flags=re.IGNORECASE)

    # Replace gendered pronouns with neutral pronouns
    gender_pronouns = r"\b(he|him|his|she|her|hers|they|them|theirs)\b"
    text = re.sub(gender_pronouns, "PERSON", text, flags=re.IGNORECASE)

    # Replace gendered words with neutral alternatives (including plurals)
    gender_terms = r"\b(man|men|boy|boys|woman|women|girl|girls)\b"
    text = re.sub(gender_terms, "PERSON", text, flags=re.IGNORECASE) 



    return text

def update_training_file(input_filename, output_filename):
    with open(input_filename, "r") as f:
        text = f.read()

    neutralized_text = neutralize_text(text)

    with open(output_filename, "w") as f:
        f.write(neutralized_text)

if __name__ == "__main__":
    training_file = "train.txt"
    output_file = "train_neutralized.txt"
    update_training_file(training_file, output_file)
