# Positive-Engagement-Classification-for-YouTube-Titles
This program creates a locally hosted app that will attempt to classify the predicted positive engagements (likes/dislikes+likes) that a trending video title would receive on YouTube. There are three main classification categories: 'Above Average', 'Average', and 'Below Average'


## Repo structure 

```
├── README.md                         <- You are here
│
├── cnn_predict.py                    <- Script that generates classifications based on the saved CNN model file and Dictionary
│   
├── best_cnn.model                    <- Model file for the best performing CNN found during experimentation
│   
├── cnn.dict                          <- CNN dictionary file used for classifications
│   
├── api.py                            <- Script that hosts a local flask app for predictions
│  
├── experiemnts.ipynb                 <- Jupyter notebook full of experiments to find the optimal model
│   
