# Feedback Prize - Evaluating Student Writing

This is the solution for 2nd rank in Kaggle competition: Feedback Prize - Evaluating Student Writing. The competition can be found here: https://www.kaggle.com/competitions/feedback-prize-2021/

## Datasets required

Download folds from https://www.kaggle.com/datasets/ubamba98/feedbackgroupshufflesplit1337

Download and convert LSG Roberta from https://github.com/ccdv-ai/convert_checkpoint_to_lsg/blob/main/convert_roberta_checkpoint.py

## Models and Training

* Deberta large, Deberta xlarge, Deberta v2 xlarge,  Deberta v3 large, Funnel transformer large are trained using __trainer.py__

* Deberta large with LSTM head and jaccard loss is trained using __debertabilstm_trainer.py__

* Longformer large with LSTM head is trained using __longformerwithbilstm_trainer.py__

* LSG Roberta is trained with __lsgroberta_trainer.py__

* YOSO is trained with __yoso_trainer.py__

## Inference

After training all the models, the outputs were pushed to Kaggle Datasets.

And the final inference kernel can be found here: https://www.kaggle.com/code/cdeotte/2nd-place-solution-cv741-public727-private740?scriptVersionId=90301836


Solution writeup: https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389