=====================================
Chenyu You (cy346)
=====================================

------------------------------------------
Part A
------------------------------------------

In the original paper, they find it beneficial to reverse the order of the input when feeding it to the model.

Q1. Write why you think that would help. (hint -- see the original paper for their explanation.)
Answer:
The benefits to reverse the order of input are as follows. First, it introduces many short term dependencies in the data that make the optimization problem much easier. Normally, when we concatenate a source sentence with a target sentence, the distance between a word in the source sentence and its corresponding word target sentence is far. By reversing the words in the source sentence, the average distance between corresponding words in the source and target language remain unchanged. But the first few words in source sentence are becoming very close to the first few words in target sentence, so it greatly reduce the problem's minimal time lag. Furthermore, backpropagation help with the connection between source sentence and target sentence, which in turn results in substantially imrpoved overall performance.


Q2. Can you think of any additional data augmentation which could be done to improve translation results?
Answer:
1. Thesaurus: replacing words or phrases with their synonyms
2. Word Embeddings + cosine similarity: Find similar words for replacement. 

Outputs
---------------------------------------------------------------------------------------------------
Epoch: 01 | Time: 0m 31s
	Train Loss: 4.766 | Train PPL: 117.492
	 Val. Loss: 4.847 |  Val. PPL: 127.421
Epoch: 02 | Time: 0m 31s
	Train Loss: 3.920 | Train PPL:  50.394
	 Val. Loss: 4.233 |  Val. PPL:  68.901
Epoch: 03 | Time: 0m 31s
	Train Loss: 3.418 | Train PPL:  30.506
	 Val. Loss: 4.006 |  Val. PPL:  54.950
Epoch: 04 | Time: 0m 31s
	Train Loss: 3.141 | Train PPL:  23.123
	 Val. Loss: 3.836 |  Val. PPL:  46.349
Epoch: 05 | Time: 0m 31s
	Train Loss: 2.887 | Train PPL:  17.936
	 Val. Loss: 3.732 |  Val. PPL:  41.762

| Test Loss: 3.734 | Test PPL:  41.855 |


------------------------------------------
Part B
------------------------------------------

Q1. Find a German sentence from Google Translate (or any other source) and try translating it to English with our system. Add the model's translation as well as Google's translation to your README. We are training for a very short time and on very little data, so we can't expect good results, but see if what parts our model is able to correctly translate.
Answer:
German sentence: Am Ende geht die Sonne auf
Google's translation: In the end the sun rises
Model's translation: ['the', 'train', 'is', 'the', 'the', 'on', 'the', '.', '.']


Outputs
---------------------------------------------------------------------------------------------------
Epoch: 01 | Time: 0m 35s
	Train Loss: 4.460 | Train PPL:  86.447
	 Val. Loss: 4.239 |  Val. PPL:  69.365
Epoch: 02 | Time: 0m 35s
	Train Loss: 3.565 | Train PPL:  35.336
	 Val. Loss: 3.844 |  Val. PPL:  46.730
Epoch: 03 | Time: 0m 35s
	Train Loss: 3.135 | Train PPL:  22.998
	 Val. Loss: 3.781 |  Val. PPL:  43.872
Epoch: 04 | Time: 0m 35s
	Train Loss: 2.873 | Train PPL:  17.685
	 Val. Loss: 3.621 |  Val. PPL:  37.370
Epoch: 05 | Time: 0m 35s
	Train Loss: 2.622 | Train PPL:  13.762
	 Val. Loss: 3.617 |  Val. PPL:  37.228
