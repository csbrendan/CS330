import json
import matplotlib.pyplot as plt
import pandas as pd


# Load the JSON file to examine its structure
file_path = 'evaluation_results.json'

with open(file_path, 'r') as file:
    evaluation_results = json.load(file)

# Displaying the structure of the JSON file for better understanding
evaluation_results.keys(), {key: list(evaluation_results[key].keys()) for key in evaluation_results.keys()}


#Plotting only the bar charts for gender, race, and profession comparisons
fig, axes = plt.subplots(1, 3, figsize=(18, 6))


# Extract and process data for gender comparison
data_gender = {
    "Metric": ["LM Score", "SS Score", "ICAT Score"],
    "LL Neutral Masking": [evaluation_results['LL_NM_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['gender']['LM Score'], 
                evaluation_results['LL_NM_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['gender']['SS Score'], 
                evaluation_results['LL_NM_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['gender']['ICAT Score']],
    "LL CDA": [evaluation_results['LL_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['gender']['LM Score'], 
                evaluation_results['LL_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['gender']['SS Score'], 
                evaluation_results['LL_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['gender']['ICAT Score']],
    "Base Model": [evaluation_results['stereoset_m-GPT2LMHeadModel_c-gpt2']['intrasentence']['gender']['LM Score'], 
                   evaluation_results['stereoset_m-GPT2LMHeadModel_c-gpt2']['intrasentence']['gender']['SS Score'], 
                   evaluation_results['stereoset_m-GPT2LMHeadModel_c-gpt2']['intrasentence']['gender']['ICAT Score']]
}

df_gender_corrected = pd.DataFrame(data_gender)

# Extract and process data for race comparison
data_race = {
    "Metric": ["LM Score", "SS Score", "ICAT Score"],
    "LL Neutral Masking": [evaluation_results['LL_NM_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['race']['LM Score'], 
                evaluation_results['LL_NM_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['race']['SS Score'], 
                evaluation_results['LL_NM_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['race']['ICAT Score']],
    "LL CDA": [evaluation_results['LL_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['race']['LM Score'], 
                evaluation_results['LL_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['race']['SS Score'], 
                evaluation_results['LL_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['race']['ICAT Score']],
    "Base Model": [evaluation_results['stereoset_m-GPT2LMHeadModel_c-gpt2']['intrasentence']['race']['LM Score'], 
                   evaluation_results['stereoset_m-GPT2LMHeadModel_c-gpt2']['intrasentence']['race']['SS Score'], 
                   evaluation_results['stereoset_m-GPT2LMHeadModel_c-gpt2']['intrasentence']['race']['ICAT Score']]
}

df_race_corrected = pd.DataFrame(data_race)



# Extract and process data for profession comparison
data_profession = {
    "Metric": ["LM Score", "SS Score", "ICAT Score"],
    "LL Neutral Masking": [evaluation_results['LL_NM_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['profession']['LM Score'], 
                evaluation_results['LL_NM_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['profession']['SS Score'], 
                evaluation_results['LL_NM_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['profession']['ICAT Score']],
    "LL CDA": [evaluation_results['LL_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['profession']['LM Score'], 
                evaluation_results['LL_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['profession']['SS Score'], 
                evaluation_results['LL_stereoset_m-CDAGPT2LMHeadModel_c-gpt2_s-42']['intrasentence']['profession']['ICAT Score']],
    "Base Model": [evaluation_results['stereoset_m-GPT2LMHeadModel_c-gpt2']['intrasentence']['profession']['LM Score'], 
                   evaluation_results['stereoset_m-GPT2LMHeadModel_c-gpt2']['intrasentence']['profession']['SS Score'], 
                   evaluation_results['stereoset_m-GPT2LMHeadModel_c-gpt2']['intrasentence']['profession']['ICAT Score']]
}

df_profession_corrected = pd.DataFrame(data_profession)

# Now df_profession_corrected is ready for plotting




# Gender comparison bar chart
df_gender_corrected.plot(kind='bar', x='Metric', ax=axes[0])
axes[0].set_title('Gender Comparison - Bar Chart')
axes[0].set_ylabel('Scores')

# Race comparison bar chart
df_race_corrected.plot(kind='bar', x='Metric', ax=axes[1])
axes[1].set_title('Race Comparison - Bar Chart')
axes[1].set_ylabel('Scores')

# Profession comparison bar chart
df_profession_corrected.plot(kind='bar', x='Metric', ax=axes[2])
axes[2].set_title('Profession Comparison - Bar Chart')
axes[2].set_ylabel('Scores')

plt.tight_layout()
plt.show()
