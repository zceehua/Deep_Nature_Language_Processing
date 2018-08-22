# Ensemble for text classification  
 
This project is for those who would like to start on text classification problems and those whoe would like to have a quick insight into model ensembling. The dataset was crawled from BBC news website(only a small part provided here).  

**Requirements:**  

* Python 3  
* TensorFlow >= 1.4  
* scikit-learn >=0.18.1 
* nltk  
* hyperopt  
* xgboost >=0.6

**How to run:**  
  ```
  python main.py --mode test
  ```

**Models Used:**  

1. [CNN for Sentence Classification](https://arxiv.org/abs/1408.5882)  
2. RandomForestClassifier  
3. GradientBoostingClassifier  
4. ExtraTreesClassifier  
5. LogisticRegression  

While more models could be used for expritments.Also, parameter optimization is enabled.The final ensemble model gives an accuracy around 93% on test set while this could vary a bit.

**Data set:**
Totally 2501 samples are from BBC news containing 8 categories. The dataset is not balanced, more data will be added to slove this problem.The performance might be further improved if more data is avaliable
