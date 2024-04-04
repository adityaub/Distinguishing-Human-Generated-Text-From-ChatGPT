import pandas as pd, numpy as np, re

from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
import pickle
import nltk
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

root = tk.Tk()
root.title("Distinguishing Human Generated Text From \n ChatGPT  Generated Text Using Machine Learning")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
# ++++++++++++++++++++++++++++++++++++++++++++

image2 = Image.open('img4.jpeg')

image2 = image2.resize((w, h))

background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)

label_l1 = tk.Label(root, text="Distinguishing Human Generated Text From \n ChatGPT  Generated Text Using Machine Learning ",font=("Times New Roman", 30, 'bold'),
                    background="#000000", fg="white", width=50, height=2)
label_l1.place(x=0, y=0)


frame_alpr = tk.LabelFrame(root, text=" Url", width=300, height=450, bd=5, font=('times', 14, ' bold '),bg="white")
frame_alpr.grid(row=0, column=0)
frame_alpr.place(x=60, y=160)


# lbl = tk.Label(root, text="Milk Quality detection", font=('times', 35,' bold '), height=1, width=32,bg="violet Red",fg="Black")
# lbl.place(x=300, y=10)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++




def Data_Preprocessing():

    data = pd.read_csv("data.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    
    data['category'] = le.fit_transform(data['category'])
    
    data['text'] = le.fit_transform(data['text'])
    
    data['Label'] = le.fit_transform(data['Label'])


  

    """Feature Selection => Manual"""
    x = data.drop(['Label'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Label']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=1 )

    

    load = tk.Label(root, font=("Tempus Sans ITC", 15, "bold"), width=50, height=2, background="green",
                    foreground="white", text="Data Loaded=>Splitted into 80% for Training & 20% for Testing")
    load.place(x=200, y=80)


def Train():
    
    result = pd.read_csv(r"combined_dataset.csv",encoding = 'unicode_escape')

    result.head()
        
    result['headline_without_stopwords'] = result['domain'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    
    
    def pos(headline_without_stopwords):
        return TextBlob(headline_without_stopwords).tags
    
    
    os = result.headline_without_stopwords.apply(pos)
    os1 = pd.DataFrame(os)
    #
    os1.head()
    
    os1['pos'] = os1['headline_without_stopwords'].map(lambda x: " ".join(["/".join(x) for x in x]))
    
    result = result = pd.merge(result, os1, right_index=True, left_index=True)
    result.head()
    result['pos']
    result_train, result_test, label_train, label_test = train_test_split(result['pos'], result['label'],
                                                                              test_size=0.10, random_state=1)
    
    tf_vect = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, sublinear_tf=False)
    
    X_train_tf = tf_vect.fit_transform(result_train)
    X_test_tf = tf_vect.transform(result_test)
    
    
    
    clf = svm.SVC(C=10, gamma=0.001, kernel='linear')   
    clf.fit(X_train_tf, label_train)
    pred = clf.predict(X_test_tf)
    
    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(tf_vect, fin)
    with open('mlmodel.pickle', 'wb') as f:
        pickle.dump(clf, f)
    
    pkl = open('mlmodel.pickle', 'rb')
    clf = pickle.load(pkl)
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    
    X_test_tf = tf_vect.transform(result_test)
    pred = clf.predict(X_test_tf)
    
    print(metrics.accuracy_score(label_test, pred))
    
    print(confusion_matrix(label_test, pred))
    
    print(classification_report(label_test, pred))

       
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(label_test, pred)))
    print("Accuracy : ",accuracy_score(label_test, pred)*100)
    accuracy = accuracy_score(label_test, pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(label_test, pred) * 100)
    repo = (classification_report(label_test, pred))
    
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=405,y=100)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as SVM_MODEL.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=405,y=320)
    
    dump (clf,"SVM_MODEL.joblib")
    print("Model saved as SVM_MODEL.joblib")

    print("chatgpt.joblib")
    
entry = tk.Entry(frame_alpr,width=24,bg="#ff9999",font=("Times New Roman",16,"italic"))
entry.insert(0,"Enter text here...")
entry.place(x=5,y=90)
    
def Test():
    predictor = load("SVM_MODEL.joblib")
    Given_text = entry.get()
    #Given_text = "the 'roseanne' revival catches up to our thorny po..."
    vec = open('vectorizer.pickle', 'rb')
    tf_vect = pickle.load(vec)
    X_test_tf = tf_vect.transform([Given_text])
    y_predict = predictor.predict(X_test_tf)
    print(y_predict[0])
    if y_predict[0]==0:
        label4 = tk.Label(root,text ="Human",width=24,height=2,bg='#ff9999',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=400,y=500)
    elif y_predict[0]==1:
        label4 = tk.Label(root,text ="BOT",width=35,height=2,bg='Red',fg='black',font=("Tempus Sanc ITC",14))
        label4.place(x=400,y=500)



def call_file():
    import Check_Milk_Quality
    Check_Milk_Quality.Train()
    
def Main():
       from subprocess import call
       call(['python','Check.py'])



def window():
    root.destroy()

button1 = tk.Button(frame_alpr,command=Data_Preprocessing,text="Data_Preprocessing",bg="#ff9999",fg="black",width=24,font=("Times New Roman",15,"italic","bold"))
button1.place(x=5,y=30)



button2 = tk.Button(frame_alpr,command= Train,text=" TrainSVM",bg="#ff9999",fg="black",width=24,font=("Times New Roman",15,"italic","bold"))
button2.place(x=5,y=140)



button2 = tk.Button(frame_alpr,command=Test,text="Test",bg="#ff9999",fg="black",width=24,font=("Times New Roman",15,"italic","bold"))
button2.place(x=5,y=190)


button3 = tk.Button(frame_alpr,text="Exit",command=window,bg="red",fg="black",width=24,font=("Times New Roman",15,"italic","bold"))
button3.place(x=5,y=250)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
