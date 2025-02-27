## Text-Normalization
# Goal 
The goal of the task was to analyse the dataset provided, that included raw data and their normalised representation, and implement a scalable solution able to generalise on this normalisation and produce the same pattern of results with unseen new data.

# Dataset Analysis
We run a series of functions to view the dataset and perform a statistical review on its contents. Two plots created for this purpose are provided . We also extract some .json files with useful data.

# Approach 1 - Heuristics
After careful examination of the data we try to provide some rules for the raw text normalisation. We create two heuristic functions, one that is based on punctuation and domain-specific keywords and achieves a prediction accuracy around 0.65 and another that is based on the preservation rate of a character depending on the script it belongs to, based on the observations from the original dataset, that achieves similar results. For example for the first function we would replace punctuation characters and remove keywords while for the second function we would work on the text character by character, identifying in which script it belongs to and if the script stats that we have collected shows that most frequently this scripts character’s scripts are omitted we will remove it.

# Approach 2 - T5 transformers
Since our heuristics attempt did not yield very good results we tried a machine learning method. After preprocessing and formatting the data accordingly we train a T5 transform model to ‘learn’ the normalisation patterns it can find and repeat them on its predictions. We evaluate a series of transformer models from hugging face, although due to limited resources the results were limited and maybe not representative. Despite that we created models with up to 80% accuracy.

# Approach 3 - Combination
We saw that we can train a model using our data, so we tried to improve its observations by implementing the previous heuristic functions. By implementing one or both of the heuristics before the input of the model we observe a slight improvement. This approach can be better evaluated if its tested on a better trained model, which we did not do due to time constraints.

# Conclusion
The normalisation of the raw data as set in the dataset we worked on is feasible mainly through training a machine learning model which is pretrained on similar text tasks. For optimal solution we can use heuristics complementary to the models.

# Further Work
Scaling this task we can try a more elaborative search on T5 model capabilities, with longer trainings on bigger models (as we used mainly the small ones due to training locally) . The combination with heuristics should be explored although the can be redundant if a model’s performance skyrockets.
