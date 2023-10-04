# Validators Code

Minor Bioinformatics Internship Repository, Julian REE370
Description Internship: 
Creating an online survey with a conversational interface and comparing its answers to that of traditional online survey. Additionally, comparing the influence of verbal answering instead of typing.

There are three conditions: 
- Traditional Survey(A),
- Chatbot survey typed(B),
- Chatbot survey spoken(C).

Measurements:
- Informativeness
  - Information Theory 
- Readability
  - Flesch Reading Ease 
- Valence
  - Sentimental Analysis 
- Wordcount
  - Quantity words
- Self-Disclosure
  - Object analysis, and manual topic selection
- Time
  - Objective time and Subjective time 
- User Experience
  - Closed questions with Likert Scale
- Prior Experience with Chatbot
  - Closed questions with Likert Scale

Preprocessing files
- process_sonar.ipynb: Process the SoNAR corpus frequency file to use for informativeness scoring
- preprocess_conditions: Extract, calculate, and preprocess the relevant information from the survey outputs
Statistical analysis files
- process_data.Rmd: Visualize and perform statistical analysis on the preprocessed data (except self-disclosure)
- analyze_object_analysis.Rmd: Visualize and perform statistical analysis on the object_analysis output file for the self-disclosure measurement


![alt text](https://github.com/Goblok0/Validators/blob/bc9fc9d167da53009579b6cbbe9ef8322a21e717/Flowchart_Validators_Code.png)
