# Validators Code

Minor Bioinformatics Internship Repository, Julian REE370
Description Internship: 
Creating an online survey with a conversational interface and comparing its answers to that of traditional online survey. Additionally, comparing the influence of verbal answering instead of typing.

There are three conditions: 
- Tradition Survey(A),
- Chatbot survey typed(B),
- Chatbot survey spoken(C).

Measurements:
- Informativeness
- Readability
- Valence
- wordcount
- self-disclosure
- time
- user experience
- prior experience with chatbot

Preprocessing files
- process_sonar.ipynb: Process the SoNAR corpus frequency file to use for informativeness scoring
- preprocess_conditions: Extract, calculate, and preprocess the relevant information from the survey outputs
- process_data.Rmd: Visualize and perform statistical analysis on the preprocessed data (except self-disclosure)
- analyze_object_analysis.Rmd: Visualize and perform statistical analysis on the object_analysis output file for the self-disclosure measurement


![alt text](https://github.com/Goblok0/Validators/blob/92b37fae2ae3cfc4b4d9a8c316cb7e40f96b4a68/Flowchart_Validators.png)
