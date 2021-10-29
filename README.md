# Disaster Response Pipeline

Author: Horst Osberger

This is a project for the UDACITY Nanodegree "Data Scientist"


## Table of content
* [Description of the project](#Chap1)
* [About the data and the created model](#Chap2)
* [About this repository](#Chap3)
* [File descriptions](#Chap4)
* [Installation instructions](#Chap5)
* [Project conclusion](#Chap6)
* [License](#Chap7)


## Description of the project <a name=Chap1></a>

After a natural disaster or a terrorist attack, people often use social media or other services to write message to inform and warn other people. 
In such a case, thousands of thousands such messages pop up during a very short time. This information could be helpful for aid organizations
or governments to provide the right help at the right place. However, due to the massive amount of messages, it is hard to filter out
meaningful information without wasting too much time.  

The goal of this project is to provide a multi-class classifier to categorize such disaster messages. 

Starting with raw csv-files that contain messages about disasters from different sources (e.g., social media), we define
pipelines that clean/transform the information and build a model to classify the disaster messages in certain categories. More detailed: 
1. ETL pipeline that cleans, transforms, and stores the raw message data. 
2. ML pipeline that creates a model that can classify new disaster messages. 
3. A web-app that nicely visualizes the data and that can be used to classify new disaster messages.  

See the screenshots below to get an idea how the web-app looks like.  

![Start screen](./screenshots/start_screen.PNG?raw=true "Start screen")

![Request](./screenshots/request_1.PNG?raw=true "Request")


## About the data and the created model <a name=Chap2></a>

The raw csv-files are provided by [Figure Eight](https://www.figure-eight.com/) ([appen](https://appen.com/), respectively).  
Thanks for that!!

Having a look on the data distribution, see the screenshot above, one can see that many categories, as e.g. `water`, are purely 
represented. To make the data more balanced, we added the possibility to apply data augmentation on the data. 
In rough words, for each category, the provided functionality creates new messages by tokenizing existing messages and
reusing the most descriptive words for the selected category. 
For details on the implementation, see the class `TextAugmentation()` in [./models/functions.py](./models/functions.py).

Furthermore, the imbalanced data needs to be considered when having a look on the evaluation measures of the created model. 
The current best model achieved here attains the following **mean evaluation metrics over all categories**
on the test dataset (20% of the whole data):

|               | overall mean  |
| ------------- |--------------:|
| **Accuracy**  | 0.946163      |
| **Precision** | 0.545690      |
| **Recall**    | 0.245347      |
| **F1-Score**  | 0.299345      |

The accuracy indicates that the results are very good. However, this is misleading due to the following reason. 
For each category, a classifier is trained with two classes, the class "category" itself and a class "others"
that contains all other categories. Assuming that the selected category, e.g. `water`, has a low number of samples in
the dataset. Then, if such a classifier is tested on a test set, there are much less samples for the class "category"
than for the class "others", which leads to a high accuracy even if samples labeled as "category" are
classified as "others". In other words: A wrong classification of almost all messages in the class "category" can still lead
to high values for the accuracy and the precision.  
Therefore, the main metric we should consider here is the `recall`, telling us that there are still a lot of 
false negatives. 

In this project, we tried several models and pipelines to find a model with a high recall. For details, see
the [helper_ml_pipeline jupyter notebook](./models/helper_ml_pipeline.ipynb). The best working pipeline/model
is used in the final [training script train_classifier.py](./models/train_classifier.py). 


## About this repository <a name=Chap3></a>

This is a Python project. The raw data containing disaster messages is provided by `.csv`-files, see the folder `/data`.  

This project uses the following python libraries:
* [argparse](https://docs.python.org/3/library/argparse.html): Parser for command-line options
* [collections](https://docs.python.org/3/library/collections.html): Container database library
* [flask](https://flask.palletsprojects.com/en/2.0.x/): Python based web framework
* [joblib](https://pypi.org/project/joblib/): Library to read/write `.pkl` files
* [nltk](https://www.nltk.org/): Natural Language Toolkit library
* [os](https://docs.python.org/3/library/os.html): Miscellaneous operating system interfaces
* [pandas](https://pandas.pydata.org/): Library to handle datasets
* [plotly](https://plotly.com/): Interactive, open-source, and browser-based graphing library for Python
* [re](https://docs.python.org/3/library/re.html): Library to handle regular expressions
* [scikit-learn](https://scikit-learn.org/stable/): Machine Learning library 
* [sqlalchemy](https://www.sqlalchemy.org/): Library to handle SQL databases
* [sys](https://docs.python.org/3/library/sys.html): System specific params and functions
* [tqdm](https://tqdm.github.io/): Library for nice progress bar visualization 
* [warnings](https://docs.python.org/3/library/warnings.html): Warning control library


## File descriptions <a name=Chap4></a>
├── data\
│ ├── DisasterResponse.db \
│ ├── disaster_categories.csv \
│ ├── disaster_messages.csv\
│ ├── helper_etl_pipeline.ipynb >>> Helper Jupyter Notebook explaining steps for process_data.py \
│ └── process_data.py >>> ETL pipeline - a Python script that loads the messages and categories datasets\
                          merges the two datasets,cleans the data,stores it in a SQLite database\
├── models\
│ ├── functions.py >>> File containing helper functions \
│ ├── classifier.pkl >>> Pretrained model\
│ ├── helper_ml_pipeline.ipynb >>> Helper Jupyter Notebook explaining steps for train_classifier.py \
│ └── train_classifier.py >>> ML pipeline - a Python script that builds a text processing and machine learning pipeline\
                              which trains and tunes a model using GridSearchCV, and then exports the final model as classifier.pkl\
├── screenshots\
│ ├── request_1.PNG \
│ ├── start_screen.PNG \

├── web_app\
│ ├── static\
│ │ ├── logos\
│ │ │ ├─ githublogo.png\
│ │ │ └─ linkedinlogo.png\
│ ├── templates\
│ │ ├─ go.html\
│ │ └─ master.html\
│ └── run.py

├── requirements.txt\
├── README.md

## Installation instructions <a name=Chap5></a>

0. Installation requirements: Python 3.7.3 or higher
1. Install the required Python packages using a virtual environment and the `requirements.txt` file. E.g., for Linux systems run
```console
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```
2. Build a model by running the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        ```console
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains and saves the classifier  
        ```console
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```  
        Note that this call will take a long time since training data is augmented. To avoid this, use e.g.,  
        ```console
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl --skip_data_augmentation
        ```  
        For further helpful commands, call  
        ```console
        python models/train_classifier.py --help
        ```  
3. Switch to the app's directory to run your web app.
    ```
    cd app
    python run.py
    ```

4. Go to localhost:3001/ or http://0.0.0.0:3001/


## Project conclusion <a name=Chap6></a>

In this project, a baseline multi-class classifier and several approaches to improve this baseline classifier has been
trained and tested, see [helper_ml_pipeline jupyter notebook](./models/helper_ml_pipeline.ipynb). 
Unfortunately, most experiments did not lead to a significant improvement of the baseline classifier. Therefore,
further investigations should be done to make the model better.  
One idea might be to use Support Vector Classification that showed good results for the recall at least
for the category `water`. However, the training for all categories takes a long time.

In general, I liked the task and enjoyed to play around with different classifiers and the text data to find a good model.  
To be honest, I'm not sure if I would use scikit-learn pipelines in the future after this project, because
they seem a bit restrictive to me. 
E.g., I was a bit disappointed when I found out that data augmentation cannot be handled within the pipelines,
because `.fit_transform()` functions do not return the labels `y` that also need to be considered during augmentation. 
I believe that writing a pipeline "by hand" is not that hard and would provide more flexibility.  
In addition, when using grid search for training, I missed some visualization/feedback on how long the training
will take. This would have been helpful. 

## License <a name=Chap7></a>

The code is free for any usage. For the license of the raw data, please check the license of data provided by [Figure Eight](https://www.figure-eight.com/)/[appen](https://appen.com/). 